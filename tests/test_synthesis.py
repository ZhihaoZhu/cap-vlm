"""Tests for PPV data synthesis pipelines."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ppv.synthesis.apc_synthesizer import APCSynthesizer
from ppv.synthesis.vstp_synthesizer import VSTSynthesizer
from ppv.synthesis.hvc_synthesizer import HVCSynthesizer
from ppv.synthesis.annotator import SceneAnnotator


class TestSceneAnnotator:
    @patch("ppv.synthesis.annotator.AutoProcessor")
    @patch("ppv.synthesis.annotator.AutoModelForCausalLM")
    def test_init(self, mock_model_cls, mock_proc_cls):
        annotator = SceneAnnotator(model_name="test-model", device="cpu")
        assert annotator is not None

    @patch("ppv.synthesis.annotator.AutoProcessor")
    @patch("ppv.synthesis.annotator.AutoModelForCausalLM")
    def test_decompose_scene_returns_regions(self, mock_model_cls, mock_proc_cls):
        annotator = SceneAnnotator(model_name="test-model", device="cpu")
        # Mock the VLM call to return a region list
        annotator._call_vlm = MagicMock(
            return_value=json.dumps(
                [
                    {
                        "bbox": [0, 0, 100, 50],
                        "description": "Navigation bar",
                        "element_type": "navigation",
                    },
                    {
                        "bbox": [0, 50, 100, 200],
                        "description": "Main content area",
                        "element_type": "content",
                    },
                ]
            )
        )
        regions = annotator.decompose_scene("test.jpg")
        assert len(regions) == 2
        assert regions[0]["element_type"] == "navigation"


class TestAPCSynthesizer:
    def test_filter_rejects_short_chains(self, tmp_path):
        synth = APCSynthesizer.__new__(APCSynthesizer)
        synth.output_dir = str(tmp_path)

        short_sample = {
            "perception_chain": [{"region": "center", "observation": "text"}],
            "task": "Find button",
        }
        assert synth._filter(short_sample) is False

    def test_filter_accepts_valid_chains(self, tmp_path):
        synth = APCSynthesizer.__new__(APCSynthesizer)
        synth.output_dir = str(tmp_path)

        valid_sample = {
            "perception_chain": [
                {"region": "top", "observation": "nav bar", "reasoning": "check nav"},
                {"region": "center", "observation": "content", "reasoning": "look at content"},
                {"region": "bottom", "observation": "footer", "reasoning": "found it"},
            ],
            "task": "Find contact info",
        }
        assert synth._filter(valid_sample) is True


class TestVSTSynthesizer:
    def test_filter_rejects_trivial_transitions(self, tmp_path):
        synth = VSTSynthesizer.__new__(VSTSynthesizer)
        synth.output_dir = str(tmp_path)

        trivial = {
            "state_changes": "",
            "action": "click",
        }
        assert synth._filter(trivial) is False

    def test_filter_accepts_meaningful_transitions(self, tmp_path):
        synth = VSTSynthesizer.__new__(VSTSynthesizer)
        synth.output_dir = str(tmp_path)

        meaningful = {
            "state_changes": "The product list reordered by price. A new filter badge appeared.",
            "action": "Click sort by price",
        }
        assert synth._filter(meaningful) is True


class TestHVCSynthesizer:
    def test_filter_rejects_no_steps(self, tmp_path):
        synth = HVCSynthesizer.__new__(HVCSynthesizer)
        synth.output_dir = str(tmp_path)

        empty = {"steps": [], "task": "test"}
        assert synth._filter(empty) is False

    def test_filter_accepts_valid_chain(self, tmp_path):
        synth = HVCSynthesizer.__new__(HVCSynthesizer)
        synth.output_dir = str(tmp_path)

        valid = {
            "steps": [
                {
                    "hypothesis": "Clicking X opens settings",
                    "expected_outcome": "Settings page",
                    "actual_outcome": "Settings page",
                    "assessment": "correct",
                }
            ],
            "task": "Open settings",
        }
        assert synth._filter(valid) is True
