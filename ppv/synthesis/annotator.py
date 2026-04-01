"""VLM-based scene annotator for visual decomposition and task generation."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


class SceneAnnotator:
    """Wraps a vision-language model to analyse UI/visual scenes.

    Provides primitives consumed by every synthesizer: region detection,
    task generation, state description, and state comparison.
    """

    def __init__(self, model_name: str, device: str = "cuda") -> None:
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the VLM and processor on first use."""
        if self.model is not None:
            return
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name, trust_remote_code=True, device_map=self.device
            )
            logger.info("Loaded VLM %s on %s", self.model_name, self.device)
        except Exception:
            logger.warning("Could not load VLM %s; falling back to stub mode", self.model_name)
            self.model = None
            self.processor = None

    def _call_vlm(self, prompt: str, image_path: str | None = None, temperature: float = 0.3) -> str:
        """Send a prompt (optionally with an image) through the VLM.

        Falls back to a placeholder string when the model is unavailable so that
        the pipeline structure can be exercised without GPU resources.
        """
        self._ensure_loaded()

        if self.model is None or self.processor is None:
            return self._stub_response(prompt)

        image = Image.open(image_path).convert("RGB") if image_path else None
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):].strip()
        return decoded

    def decompose_scene(self, image_path: str) -> list[dict]:
        """Decompose a visual scene into annotated regions.

        Returns a list of dicts each containing:
            bbox: [x1, y1, x2, y2]  (normalised 0-1)
            description: str
            element_type: str  (e.g. "button", "text", "image", "chart")
        """
        prompt = (
            "Analyse this UI screenshot and identify every distinct interactive or "
            "informational element. For each element return a JSON object with keys "
            '"bbox" ([x1,y1,x2,y2] normalised 0-1), "description" (short text), '
            'and "element_type" (one of: button, text, image, chart, input, icon, '
            "container, link, menu, other). Return a JSON array."
        )
        raw = self._call_vlm(prompt, image_path)
        return self._parse_json_array(raw)

    def generate_tasks(self, image_path: str, regions: list[dict], num_tasks: int = 5) -> list[str]:
        """Generate diverse agentic tasks that could be performed on this scene."""
        region_summary = "\n".join(
            f"- [{r['element_type']}] {r['description']}" for r in regions[:20]
        )
        prompt = (
            f"Given a UI screenshot with these elements:\n{region_summary}\n\n"
            f"Generate {num_tasks} diverse, realistic user tasks that someone might "
            "want to accomplish on this page. Return one task per line, no numbering."
        )
        raw = self._call_vlm(prompt, image_path)
        tasks = [line.strip() for line in raw.strip().splitlines() if line.strip()]
        return tasks[:num_tasks]

    def describe_state(self, image_path: str) -> str:
        """Generate a structured description of the current visual state."""
        prompt = (
            "Describe the current state of this UI screenshot in a structured way. "
            "Include: (1) page type/purpose, (2) key visible content, (3) interactive "
            "elements and their states (enabled/disabled, selected, etc.), (4) any "
            "notifications or overlays."
        )
        return self._call_vlm(prompt, image_path)

    def compare_states(self, before_path: str, after_path: str) -> str:
        """Describe what changed between two consecutive states."""
        before_desc = self.describe_state(before_path)
        after_desc = self.describe_state(after_path)
        prompt = (
            f"Before state:\n{before_desc}\n\nAfter state:\n{after_desc}\n\n"
            "Describe precisely what changed between these two UI states. "
            "Focus on element additions/removals, content changes, layout shifts, "
            "and state transitions (e.g. button became disabled)."
        )
        return self._call_vlm(prompt)

    def _parse_json_array(self, text: str) -> list[dict]:
        """Best-effort extraction of a JSON array from VLM output."""
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return [
                        r for r in parsed
                        if isinstance(r, dict) and "bbox" in r and "element_type" in r
                    ]
            except json.JSONDecodeError:
                pass
        return self._fallback_regions()

    @staticmethod
    def _fallback_regions() -> list[dict]:
        """Return a minimal placeholder region list when parsing fails."""
        return [
            {
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "description": "full screen",
                "element_type": "container",
            }
        ]

    @staticmethod
    def _stub_response(prompt: str) -> str:
        """Deterministic stub when no VLM is available."""
        if "JSON array" in prompt:
            return json.dumps([
                {"bbox": [0.1, 0.1, 0.3, 0.05], "description": "navigation bar", "element_type": "menu"},
                {"bbox": [0.2, 0.2, 0.4, 0.35], "description": "search input", "element_type": "input"},
                {"bbox": [0.5, 0.5, 0.7, 0.55], "description": "submit button", "element_type": "button"},
                {"bbox": [0.1, 0.6, 0.9, 0.9], "description": "main content area", "element_type": "container"},
            ])
        if "tasks" in prompt.lower() or "task" in prompt.lower():
            return (
                "Find the search bar and search for recent news\n"
                "Click the submit button to confirm the form\n"
                "Navigate to the settings page using the menu\n"
                "Check the notification area for unread messages\n"
                "Scroll through the main content area to find product details"
            )
        if "changed" in prompt.lower() or "compare" in prompt.lower():
            return (
                "The submit button changed from enabled to disabled. "
                "A success notification appeared at the top of the page. "
                "The form fields were cleared after submission."
            )
        return (
            "This is a web application page with a navigation bar at the top, "
            "a search input field, a submit button, and a main content area below."
        )
