#!/usr/bin/env python3
"""Main training entry point for PPV-CPT."""

import argparse
import logging
import os
import sys

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from ppv.data import PPVMixtureDataset
from ppv.models import VLMWrapper
from ppv.training import CPTTrainer, load_config

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="PPV-CPT Training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the stage config YAML (e.g., configs/stage1.yaml)",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=None,
        help="Training stage (1 or 2). Inferred from config if not set.",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--per-device-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    return parser.parse_args()


def apply_overrides(config, args):
    """Apply CLI overrides to the loaded config."""
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.per_device_batch_size is not None:
        config.data.per_device_batch_size = args.per_device_batch_size
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.seed = args.seed
    if args.wandb_project is not None:
        config.logging.project = args.wandb_project
    if args.resume_from is not None:
        config.model.resume_from = args.resume_from
    if args.stage is not None:
        config.stage = str(args.stage)
    return config


def main():
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args)

    set_seed(config.seed)

    log_with = None if args.no_wandb else "wandb"
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=log_with,
        mixed_precision="bf16" if config.bf16 else "no",
    )

    if log_with == "wandb" and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.logging.project,
            config={"stage": config.stage, "lr": config.learning_rate},
            init_kwargs={"wandb": {"name": config.logging.run_name}},
        )

    accelerator.print(f"Training stage: {config.stage}")
    accelerator.print(f"Output dir: {config.output_dir}")
    accelerator.print(f"Num processes: {accelerator.num_processes}")

    if config.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load model
    model_name_or_path = config.model.name
    if config.stage == "2" and config.model.resume_from:
        model_name_or_path = config.model.resume_from
        accelerator.print(f"Stage 2: loading model from stage 1 checkpoint: {model_name_or_path}")

    model = VLMWrapper(
        model_name_or_path=model_name_or_path,
        torch_dtype=config.model.torch_dtype,
        attn_implementation=config.model.attn_implementation,
        vision_encoder_trainable=config.model.vision_encoder_trainable,
        llm_trainable=config.model.llm_trainable,
    )
    tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Build dataset
    train_dataset = PPVMixtureDataset(
        mixture_weights=config.data.mixture,
        max_seq_length=config.model.max_seq_length,
        max_image_tokens=config.model.max_image_tokens,
        tokenizer=tokenizer,
    )
    accelerator.print(f"Dataset size: {len(train_dataset)} samples")

    # Create trainer
    trainer = CPTTrainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        accelerator=accelerator,
    )

    # Resume if requested via CLI (takes precedence over config)
    resume_path = args.resume_from
    if resume_path and os.path.isdir(resume_path):
        trainer.resume_from_checkpoint(resume_path)

    # Train
    metrics = trainer.train()

    if accelerator.is_main_process:
        logger.info("Final metrics: %s", metrics)

    if log_with == "wandb":
        accelerator.end_training()


if __name__ == "__main__":
    main()
