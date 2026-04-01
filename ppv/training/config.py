"""Training configuration for PPV-CPT."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2-VL-7B-Instruct"
    vision_encoder_trainable: bool = True
    llm_trainable: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    max_seq_length: int = 32768
    max_image_tokens: int = 4096
    resume_from: Optional[str] = None


@dataclass
class DataConfig:
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 4
    per_device_batch_size: int = 2
    total_tokens: int = 200_000_000_000
    mixture: dict = field(default_factory=lambda: {
        "apc": 0.40,
        "vstp": 0.40,
        "hvc": 0.10,
        "general_vl": 0.10,
    })
    sft_datasets: Optional[list] = None


@dataclass
class LoggingConfig:
    project: str = "ppv-cpt"
    run_name: str = "ppv-cpt"
    log_interval: int = 10
    save_interval: int = 1000
    eval_interval: int = 2000


@dataclass
class DeepSpeedConfig:
    stage: int = 2
    offload_optimizer: bool = False
    offload_param: bool = False


@dataclass
class PPVTrainingConfig:
    stage: str = "1"

    # Model
    model: ModelConfig = field(default_factory=ModelConfig)

    # Data
    data: DataConfig = field(default_factory=DataConfig)

    # Training hyperparameters
    seed: int = 42
    learning_rate: float = 2e-5
    min_learning_rate: float = 2e-6
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    bf16: bool = True
    tf32: bool = True
    num_train_epochs: int = 1
    output_dir: str = "checkpoints/stage1"

    # Logging
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # DeepSpeed
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dict_to_config(raw: dict) -> PPVTrainingConfig:
    """Convert a flat/nested dict from YAML into PPVTrainingConfig."""
    model_cfg = ModelConfig(**raw.get("model", {}))
    data_cfg = DataConfig(**raw.get("data", {}))
    logging_cfg = LoggingConfig(**raw.get("logging", {}))
    ds_cfg = DeepSpeedConfig(**raw.get("deepspeed", {}))

    training = raw.get("training", {})

    return PPVTrainingConfig(
        stage=str(raw.get("stage", "1")),
        model=model_cfg,
        data=data_cfg,
        seed=training.get("seed", 42),
        learning_rate=training.get("learning_rate", 2e-5),
        min_learning_rate=training.get("min_learning_rate", 2e-6),
        lr_scheduler=training.get("lr_scheduler", "cosine"),
        warmup_ratio=training.get("warmup_ratio", 0.03),
        weight_decay=training.get("weight_decay", 0.1),
        adam_beta1=training.get("adam_beta1", 0.9),
        adam_beta2=training.get("adam_beta2", 0.95),
        adam_epsilon=training.get("adam_epsilon", 1e-8),
        gradient_accumulation_steps=training.get("gradient_accumulation_steps", 8),
        gradient_checkpointing=training.get("gradient_checkpointing", True),
        max_grad_norm=training.get("max_grad_norm", 1.0),
        bf16=training.get("bf16", True),
        tf32=training.get("tf32", True),
        num_train_epochs=training.get("num_train_epochs", 1),
        output_dir=training.get("output_dir", "checkpoints/stage1"),
        logging=logging_cfg,
        deepspeed=ds_cfg,
    )


def load_config(config_path: str) -> PPVTrainingConfig:
    """Load a YAML config, resolving `defaults` inheritance."""
    config_path = Path(config_path)

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    defaults = raw.pop("defaults", [])
    base_raw: dict = {}
    for default_name in defaults:
        base_file = config_path.parent / f"{default_name}.yaml"
        with open(base_file) as f:
            base_raw = _deep_merge(base_raw, yaml.safe_load(f))

    merged = _deep_merge(base_raw, raw)
    return _dict_to_config(merged)
