from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class PPVModelConfig:
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    max_seq_length: int = 32768
    max_image_tokens: int = 4096
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    vision_encoder_trainable: bool = True
    llm_trainable: bool = True
    resume_from: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> PPVModelConfig:
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    def to_dict(self) -> dict:
        return asdict(self)
