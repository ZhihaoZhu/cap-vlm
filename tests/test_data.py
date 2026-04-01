"""Tests for PPV data loading and mixing."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from ppv.data.apc import APCDataset
from ppv.data.vstp import VSTDataset
from ppv.data.hvc import HVCDataset
from ppv.data.mixtures import PPVMixtureDataset


def _make_apc_sample():
    return {
        "image": "test_scene.jpg",
        "task": "Find the search button on the page",
        "perception_chain": [
            {
                "region": "top navigation bar",
                "observation": "A horizontal bar with logo, menu items, and icons",
                "reasoning": "Navigation bars typically contain search functionality",
                "next_focus": "right side of navigation bar",
            },
            {
                "region": "right side of navigation bar",
                "observation": "A magnifying glass icon next to a text input field",
                "reasoning": "This is the search button/field",
                "next_focus": None,
            },
        ],
    }


def _make_vstp_sample():
    return {
        "before_image": "before.jpg",
        "action": "Click the 'Sort by Price' dropdown",
        "state_changes": "Products reordered by ascending price. Sort dropdown now shows 'Price: Low to High'.",
    }


def _make_hvc_sample():
    return {
        "image": "page.jpg",
        "task": "Navigate to user settings",
        "steps": [
            {
                "hypothesis": "Clicking the gear icon will open settings",
                "expected_outcome": "A settings page with user preferences",
                "actual_outcome": "A settings page with user preferences",
                "assessment": "correct",
                "corrective_plan": None,
            },
        ],
    }


@pytest.fixture
def apc_data_file(tmp_path):
    path = tmp_path / "apc_data.jsonl"
    samples = [_make_apc_sample() for _ in range(10)]
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return str(path)


@pytest.fixture
def vstp_data_file(tmp_path):
    path = tmp_path / "vstp_data.jsonl"
    samples = [_make_vstp_sample() for _ in range(10)]
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return str(path)


@pytest.fixture
def hvc_data_file(tmp_path):
    path = tmp_path / "hvc_data.jsonl"
    samples = [_make_hvc_sample() for _ in range(10)]
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return str(path)


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.randint(0, 1000, (128,)),
        "attention_mask": torch.ones(128, dtype=torch.long),
    }
    tokenizer.pad_token_id = 0
    tokenizer.model_max_length = 32768
    return tokenizer


@pytest.fixture
def mock_image_processor():
    processor = MagicMock()
    processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
    return processor


class TestAPCDataset:
    def test_length(self, apc_data_file, mock_tokenizer, mock_image_processor):
        ds = APCDataset(apc_data_file, mock_tokenizer, mock_image_processor, max_seq_length=2048)
        assert len(ds) == 10

    def test_format_conversation(self, apc_data_file, mock_tokenizer, mock_image_processor):
        ds = APCDataset(apc_data_file, mock_tokenizer, mock_image_processor, max_seq_length=2048)
        sample = ds.samples[0]
        text = ds.format_conversation(sample)
        assert "search button" in text.lower() or "Search" in text
        assert "Step 1" in text or "step 1" in text.lower() or "focus" in text.lower()


class TestVSTDataset:
    def test_length(self, vstp_data_file, mock_tokenizer, mock_image_processor):
        ds = VSTDataset(vstp_data_file, mock_tokenizer, mock_image_processor, max_seq_length=2048)
        assert len(ds) == 10

    def test_format_conversation(self, vstp_data_file, mock_tokenizer, mock_image_processor):
        ds = VSTDataset(vstp_data_file, mock_tokenizer, mock_image_processor, max_seq_length=2048)
        sample = ds.samples[0]
        text = ds.format_conversation(sample)
        assert "Sort by Price" in text
        assert "reorder" in text.lower() or "price" in text.lower()


class TestHVCDataset:
    def test_length(self, hvc_data_file, mock_tokenizer, mock_image_processor):
        ds = HVCDataset(hvc_data_file, mock_tokenizer, mock_image_processor, max_seq_length=2048)
        assert len(ds) == 10

    def test_format_conversation(self, hvc_data_file, mock_tokenizer, mock_image_processor):
        ds = HVCDataset(hvc_data_file, mock_tokenizer, mock_image_processor, max_seq_length=2048)
        sample = ds.samples[0]
        text = ds.format_conversation(sample)
        assert "hypothesis" in text.lower() or "Hypothesis" in text
        assert "correct" in text.lower()


class TestPPVMixtureDataset:
    def test_mixture_length(
        self, apc_data_file, vstp_data_file, hvc_data_file, mock_tokenizer, mock_image_processor
    ):
        apc_ds = APCDataset(
            apc_data_file, mock_tokenizer, mock_image_processor, max_seq_length=2048
        )
        vstp_ds = VSTDataset(
            vstp_data_file, mock_tokenizer, mock_image_processor, max_seq_length=2048
        )
        hvc_ds = HVCDataset(
            hvc_data_file, mock_tokenizer, mock_image_processor, max_seq_length=2048
        )
        mixture = PPVMixtureDataset(
            datasets={"apc": apc_ds, "vstp": vstp_ds, "hvc": hvc_ds},
            ratios={"apc": 0.4, "vstp": 0.4, "hvc": 0.2},
        )
        assert len(mixture) == 30

    def test_mixture_sampling(
        self, apc_data_file, vstp_data_file, hvc_data_file, mock_tokenizer, mock_image_processor
    ):
        apc_ds = APCDataset(
            apc_data_file, mock_tokenizer, mock_image_processor, max_seq_length=2048
        )
        vstp_ds = VSTDataset(
            vstp_data_file, mock_tokenizer, mock_image_processor, max_seq_length=2048
        )
        hvc_ds = HVCDataset(
            hvc_data_file, mock_tokenizer, mock_image_processor, max_seq_length=2048
        )
        mixture = PPVMixtureDataset(
            datasets={"apc": apc_ds, "vstp": vstp_ds, "hvc": hvc_ds},
            ratios={"apc": 0.4, "vstp": 0.4, "hvc": 0.2},
        )
        sample = mixture[0]
        assert sample is not None
