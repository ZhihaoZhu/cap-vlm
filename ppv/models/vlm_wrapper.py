from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
)

from ppv.models.config import PPVModelConfig

logger = logging.getLogger(__name__)

QWEN2_VL_FAMILIES = ("Qwen2-VL", "Qwen2.5-VL")

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _is_qwen2_vl(model_name: str) -> bool:
    return any(tag in model_name for tag in QWEN2_VL_FAMILIES)


def _load_model_and_processor(
    model_name: str,
    torch_dtype: torch.dtype,
    attn_implementation: str,
    max_image_tokens: int | None = None,
):
    if _is_qwen2_vl(model_name):
        from transformers import Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    if max_image_tokens is not None and hasattr(processor, "image_processor"):
        ip = processor.image_processor
        if hasattr(ip, "max_pixels"):
            ip.max_pixels = max_image_tokens * 28 * 28

    return model, processor


def _get_vision_parameters(model: nn.Module) -> list[nn.Parameter]:
    vision_keywords = ("visual", "vision_tower", "vision_model", "vit", "image_encoder")
    params = []
    for name, param in model.named_parameters():
        name_lower = name.lower()
        if any(kw in name_lower for kw in vision_keywords):
            params.append(param)
    return params


def _get_vision_param_names(model: nn.Module) -> set[str]:
    vision_keywords = ("visual", "vision_tower", "vision_model", "vit", "image_encoder")
    names = set()
    for name, _ in model.named_parameters():
        name_lower = name.lower()
        if any(kw in name_lower for kw in vision_keywords):
            names.add(name)
    return names


class VLMWrapper(nn.Module):
    def __init__(
        self,
        model_name: str,
        max_seq_length: int = 32768,
        max_image_tokens: int = 4096,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        vision_encoder_trainable: bool = True,
        llm_trainable: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.max_image_tokens = max_image_tokens
        self._torch_dtype_str = torch_dtype
        self._torch_dtype = DTYPE_MAP.get(torch_dtype, torch.bfloat16)
        self.attn_implementation = attn_implementation
        self.vision_encoder_trainable = vision_encoder_trainable
        self.llm_trainable = llm_trainable

        self.model, self.processor = _load_model_and_processor(
            model_name,
            torch_dtype=self._torch_dtype,
            attn_implementation=attn_implementation,
            max_image_tokens=max_image_tokens,
        )

        if not vision_encoder_trainable:
            self.freeze_vision_encoder()

        if not llm_trainable:
            self._set_llm_requires_grad(False)

        trainable = self.num_parameters(trainable_only=True)
        total = self.num_parameters(trainable_only=False)
        logger.info(
            "Loaded %s -- %.1fM / %.1fM trainable params",
            model_name,
            trainable / 1e6,
            total / 1e6,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        forward_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if pixel_values is not None:
            forward_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            forward_kwargs["image_grid_thw"] = image_grid_thw
        forward_kwargs.update(kwargs)
        return self.model(**forward_kwargs)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        gen_kwargs: dict = {
            "input_ids": input_ids,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        if pixel_values is not None:
            gen_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            gen_kwargs["image_grid_thw"] = image_grid_thw
        gen_kwargs.update(kwargs)
        return self.model.generate(**gen_kwargs)

    def get_param_groups(
        self,
        lr: float,
        weight_decay: float,
        vision_lr_scale: float = 0.1,
    ) -> list[dict]:
        vision_names = _get_vision_param_names(self.model)

        decay_vision, no_decay_vision = [], []
        decay_llm, no_decay_llm = [], []

        no_decay_keywords = ("bias", "layernorm", "layer_norm", "rmsnorm", "embedding")

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            name_lower = name.lower()
            is_no_decay = any(kw in name_lower for kw in no_decay_keywords)
            is_vision = name in vision_names

            if is_vision:
                (no_decay_vision if is_no_decay else decay_vision).append(param)
            else:
                (no_decay_llm if is_no_decay else decay_llm).append(param)

        vision_lr = lr * vision_lr_scale
        groups = []
        if decay_vision:
            groups.append({"params": decay_vision, "lr": vision_lr, "weight_decay": weight_decay})
        if no_decay_vision:
            groups.append({"params": no_decay_vision, "lr": vision_lr, "weight_decay": 0.0})
        if decay_llm:
            groups.append({"params": decay_llm, "lr": lr, "weight_decay": weight_decay})
        if no_decay_llm:
            groups.append({"params": no_decay_llm, "lr": lr, "weight_decay": 0.0})
        return groups

    def freeze_vision_encoder(self) -> None:
        for param in _get_vision_parameters(self.model):
            param.requires_grad = False
        self.vision_encoder_trainable = False
        logger.info("Vision encoder frozen.")

    def unfreeze_vision_encoder(self) -> None:
        for param in _get_vision_parameters(self.model):
            param.requires_grad = True
        self.vision_encoder_trainable = True
        logger.info("Vision encoder unfrozen.")

    def _set_llm_requires_grad(self, requires_grad: bool) -> None:
        vision_names = _get_vision_param_names(self.model)
        for name, param in self.model.named_parameters():
            if name not in vision_names:
                param.requires_grad = requires_grad

    def num_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())

    def save_pretrained(self, path: str) -> None:
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)
        meta = {
            "model_name": self.model_name,
            "max_seq_length": self.max_seq_length,
            "max_image_tokens": self.max_image_tokens,
            "torch_dtype": self._torch_dtype_str,
            "attn_implementation": self.attn_implementation,
            "vision_encoder_trainable": self.vision_encoder_trainable,
            "llm_trainable": self.llm_trainable,
        }
        (save_dir / "ppv_config.json").write_text(json.dumps(meta, indent=2))
        logger.info("Saved model to %s", save_dir)

    @classmethod
    def from_pretrained(cls, path: str) -> VLMWrapper:
        save_dir = Path(path)
        meta_file = save_dir / "ppv_config.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
        else:
            meta = {"model_name": path}

        wrapper = cls(
            model_name=path,
            max_seq_length=meta.get("max_seq_length", 32768),
            max_image_tokens=meta.get("max_image_tokens", 4096),
            torch_dtype=meta.get("torch_dtype", "bfloat16"),
            attn_implementation=meta.get("attn_implementation", "flash_attention_2"),
            vision_encoder_trainable=meta.get("vision_encoder_trainable", True),
            llm_trainable=meta.get("llm_trainable", True),
        )
        return wrapper
