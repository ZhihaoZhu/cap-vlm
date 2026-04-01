"""CPTTrainer: main continual pre-training loop for PPV-CPT."""

import json
import logging
import math
import os
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

from ppv.training.config import PPVTrainingConfig
from ppv.training.lr_schedule import get_cosine_schedule_with_min_lr

logger = logging.getLogger(__name__)


class CPTTrainer:
    """Continual pre-training trainer using HuggingFace Accelerate."""

    def __init__(
        self,
        config: PPVTrainingConfig,
        model,
        train_dataset: Dataset,
        tokenizer,
        accelerator: Accelerator,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.train_dataset = train_dataset

        self.global_step = 0
        self.tokens_seen = 0
        self.epoch = 0
        self.best_loss = float("inf")

        self._setup_dataloader()
        self._setup_optimizer()
        self._setup_scheduler()
        self._prepare_with_accelerate()

    def _setup_dataloader(self):
        cfg = self.config
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=cfg.data.per_device_batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
            drop_last=True,
        )

    def _setup_optimizer(self):
        cfg = self.config
        param_groups = self.model.get_param_groups(cfg.learning_rate, cfg.weight_decay)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=cfg.learning_rate,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_epsilon,
        )

    def _setup_scheduler(self):
        cfg = self.config
        total_tokens = cfg.data.total_tokens
        tokens_per_step = (
            cfg.data.per_device_batch_size
            * cfg.gradient_accumulation_steps
            * self.accelerator.num_processes
            * cfg.model.max_seq_length
        )
        self.num_training_steps = max(1, total_tokens // tokens_per_step)
        num_warmup_steps = int(self.num_training_steps * cfg.warmup_ratio)
        min_lr_ratio = cfg.min_learning_rate / cfg.learning_rate

        self.scheduler = get_cosine_schedule_with_min_lr(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.num_training_steps,
            min_lr_ratio=min_lr_ratio,
        )

    def _prepare_with_accelerate(self):
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.scheduler
        )

    def train(self) -> dict:
        """Run the main training loop. Returns a dict of final metrics."""
        cfg = self.config
        total_tokens_target = cfg.data.total_tokens

        self.accelerator.print(
            f"Starting training: {self.num_training_steps} steps, "
            f"{total_tokens_target / 1e9:.1f}B tokens target"
        )

        self.model.train()
        running_loss = 0.0
        step_count_for_avg = 0
        train_start = time.time()
        step_start = time.time()

        for epoch in range(cfg.num_train_epochs):
            self.epoch = epoch
            for batch in self.train_dataloader:
                if self.tokens_seen >= total_tokens_target:
                    break

                loss = self._train_step(batch)
                running_loss += loss
                step_count_for_avg += 1

                if (self.global_step + 1) % cfg.gradient_accumulation_steps == 0:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), cfg.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                self.global_step += 1
                batch_tokens = self._count_tokens(batch)
                self.tokens_seen += batch_tokens * self.accelerator.num_processes

                if self.global_step % cfg.logging.log_interval == 0 and self.global_step > 0:
                    elapsed = time.time() - step_start
                    avg_loss = running_loss / max(step_count_for_avg, 1)
                    tokens_per_sec = (
                        batch_tokens
                        * cfg.logging.log_interval
                        * self.accelerator.num_processes
                        / max(elapsed, 1e-6)
                    )
                    current_lr = self.scheduler.get_last_lr()[0]

                    metrics = {
                        "train/loss": avg_loss,
                        "train/lr": current_lr,
                        "train/tokens_seen": self.tokens_seen,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/epoch": epoch,
                        "train/global_step": self.global_step,
                    }

                    if self.accelerator.is_main_process:
                        try:
                            import wandb

                            if wandb.run is not None:
                                wandb.log(metrics, step=self.global_step)
                        except ImportError:
                            pass

                    self.accelerator.print(
                        f"step {self.global_step}/{self.num_training_steps} | "
                        f"loss {avg_loss:.4f} | lr {current_lr:.2e} | "
                        f"{tokens_per_sec:.0f} tok/s | "
                        f"{self.tokens_seen / 1e9:.2f}B tokens"
                    )
                    running_loss = 0.0
                    step_count_for_avg = 0
                    step_start = time.time()

                if (
                    cfg.logging.save_interval > 0
                    and self.global_step % cfg.logging.save_interval == 0
                    and self.global_step > 0
                ):
                    self.save_checkpoint(self.global_step)

                if (
                    cfg.logging.eval_interval > 0
                    and self.global_step % cfg.logging.eval_interval == 0
                    and self.global_step > 0
                ):
                    self._run_eval()

            if self.tokens_seen >= total_tokens_target:
                self.accelerator.print(
                    f"Reached target token count: {self.tokens_seen / 1e9:.2f}B"
                )
                break

        self.save_checkpoint(self.global_step, final=True)

        total_time = time.time() - train_start
        final_metrics = {
            "total_steps": self.global_step,
            "total_tokens": self.tokens_seen,
            "total_time_hours": total_time / 3600,
            "avg_tokens_per_sec": self.tokens_seen / max(total_time, 1),
        }

        if self.accelerator.is_main_process:
            metrics_path = Path(cfg.output_dir) / "training_metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(final_metrics, f, indent=2)

        self.accelerator.print(f"Training complete: {final_metrics}")
        return final_metrics

    def _train_step(self, batch: dict) -> float:
        """Single forward/backward pass. Returns the scalar loss value."""
        with self.accelerator.accumulate(self.model):
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch.get("pixel_values"),
                labels=batch["labels"],
            )
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            self.accelerator.backward(loss)
        return loss.detach().float().item()

    def _count_tokens(self, batch: dict) -> int:
        """Count non-padding tokens in a batch."""
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            return int(attention_mask.sum().item())
        return int(batch["input_ids"].numel())

    def _run_eval(self):
        """Placeholder for periodic evaluation during training."""
        self.model.eval()
        self.accelerator.print(f"[step {self.global_step}] Evaluation hook (not configured)")
        self.model.train()

    def save_checkpoint(self, step: int, final: bool = False):
        """Save model, optimizer, scheduler, and trainer state."""
        self.accelerator.wait_for_everyone()
        cfg = self.config

        if final:
            save_dir = Path(cfg.output_dir) / "final"
        else:
            save_dir = Path(cfg.output_dir) / f"checkpoint-{step}"

        save_dir.mkdir(parents=True, exist_ok=True)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(str(save_dir))

        if self.tokenizer is not None and self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(str(save_dir))

        self.accelerator.save_state(str(save_dir / "accelerator_state"))

        if self.accelerator.is_main_process:
            trainer_state = {
                "global_step": self.global_step,
                "tokens_seen": self.tokens_seen,
                "epoch": self.epoch,
                "best_loss": self.best_loss,
            }
            state_path = save_dir / "trainer_state.json"
            with open(state_path, "w") as f:
                json.dump(trainer_state, f, indent=2)

        self.accelerator.print(f"Checkpoint saved to {save_dir}")

    def resume_from_checkpoint(self, path: str):
        """Resume training from a previously saved checkpoint."""
        ckpt_dir = Path(path)

        accel_state_dir = ckpt_dir / "accelerator_state"
        if accel_state_dir.exists():
            self.accelerator.load_state(str(accel_state_dir))
            self.accelerator.print(f"Loaded accelerator state from {accel_state_dir}")

        trainer_state_path = ckpt_dir / "trainer_state.json"
        if trainer_state_path.exists():
            with open(trainer_state_path) as f:
                state = json.load(f)
            self.global_step = state["global_step"]
            self.tokens_seen = state["tokens_seen"]
            self.epoch = state.get("epoch", 0)
            self.best_loss = state.get("best_loss", float("inf"))
            self.accelerator.print(
                f"Resumed from step {self.global_step}, "
                f"{self.tokens_seen / 1e9:.2f}B tokens seen"
            )

        skip_steps = self.global_step
        if skip_steps > 0:
            self.accelerator.print(f"Skipping {skip_steps} dataloader steps...")
            skipped = 0
            for _ in self.train_dataloader:
                skipped += 1
                if skipped >= skip_steps:
                    break
