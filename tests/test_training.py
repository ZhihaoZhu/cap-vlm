"""Tests for PPV training infrastructure."""

import tempfile
from pathlib import Path

import pytest
import torch

from ppv.training.lr_schedule import get_cosine_schedule_with_min_lr


class TestCosineScheduleWithMinLR:
    def test_warmup_phase(self):
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10))], lr=1e-4)
        scheduler = get_cosine_schedule_with_min_lr(
            optimizer, num_warmup_steps=100, num_training_steps=1000, min_lr_ratio=0.1
        )
        # At step 0, LR should be near 0 (start of warmup)
        assert scheduler.get_last_lr()[0] < 1e-4

    def test_peak_lr(self):
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10))], lr=1e-4)
        scheduler = get_cosine_schedule_with_min_lr(
            optimizer, num_warmup_steps=100, num_training_steps=1000, min_lr_ratio=0.1
        )
        # Step through warmup
        for _ in range(100):
            optimizer.step()
            scheduler.step()
        # At end of warmup, LR should be at peak
        lr = scheduler.get_last_lr()[0]
        assert abs(lr - 1e-4) < 1e-6

    def test_min_lr_at_end(self):
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10))], lr=1e-4)
        scheduler = get_cosine_schedule_with_min_lr(
            optimizer, num_warmup_steps=100, num_training_steps=1000, min_lr_ratio=0.1
        )
        # Step through entire schedule
        for _ in range(1000):
            optimizer.step()
            scheduler.step()
        # At end, LR should be at min_lr
        lr = scheduler.get_last_lr()[0]
        assert abs(lr - 1e-5) < 1e-7  # min_lr = 1e-4 * 0.1

    def test_lr_never_below_min(self):
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10))], lr=1e-4)
        scheduler = get_cosine_schedule_with_min_lr(
            optimizer, num_warmup_steps=10, num_training_steps=100, min_lr_ratio=0.1
        )
        min_lr = 1e-4 * 0.1
        for _ in range(100):
            optimizer.step()
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            assert lr >= min_lr - 1e-8
