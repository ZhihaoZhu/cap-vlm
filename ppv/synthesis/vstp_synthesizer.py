"""Visual State Transition Prediction (VSTP) data synthesizer.

Generates (state_before, action, state_after, transition_description) tuples
from multiple sources: web pages, GUI recordings, video, and synthetic HTML.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import tempfile
import uuid
from pathlib import Path

from ppv.synthesis.annotator import SceneAnnotator
from ppv.synthesis.base import BaseSynthesizer

logger = logging.getLogger(__name__)

SYNTHETIC_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><style>
body {{ font-family: sans-serif; padding: 20px; }}
button {{ padding: 8px 16px; cursor: pointer; }}
.counter {{ font-size: 24px; margin: 20px 0; }}
input {{ padding: 8px; margin: 4px; }}
</style></head>
<body>
<h1>{title}</h1>
<div class="counter" id="display">{display_text}</div>
{elements}
</body></html>"""

_ELEMENT_TEMPLATES = [
    '<button id="btn-{i}" onclick="this.textContent=\'Clicked!\'">{label}</button>',
    '<input id="input-{i}" type="text" placeholder="{label}" />',
    '<select id="select-{i}"><option>Option A</option><option>Option B</option></select>',
    '<input id="check-{i}" type="checkbox" /> <label>{label}</label>',
    '<a id="link-{i}" href="#">{label}</a>',
]


class VSTSynthesizer(BaseSynthesizer):
    """Synthesize Visual-State-Transition-Prediction training samples."""

    def __init__(
        self,
        annotator_model: str,
        output_dir: str,
        num_workers: int = 4,
        interaction_source: str = "synthetic",
    ) -> None:
        super().__init__(annotator_model, output_dir, num_workers)
        if interaction_source not in ("web", "gui", "video", "synthetic"):
            raise ValueError(f"Unknown interaction_source: {interaction_source}")
        self.interaction_source = interaction_source
        self.annotator = SceneAnnotator(annotator_model)

    def _generate_single(self, source: str) -> dict | None:
        dispatch = {
            "web": self._generate_from_web,
            "gui": self._generate_from_gui,
            "video": self._generate_from_video,
            "synthetic": self._generate_from_synthetic,
        }
        try:
            return dispatch[self.interaction_source](source)
        except Exception:
            logger.debug(
                "VSTP generation failed for %s (source=%s)",
                source,
                self.interaction_source,
                exc_info=True,
            )
            return None

    def _generate_from_web(self, url: str) -> dict | None:
        """Render a live URL, apply a random action, capture before/after."""
        before_path = self._render_url(url)
        if before_path is None:
            return None

        action = self._pick_random_web_action()
        after_path = self._apply_web_action(url, action)
        if after_path is None:
            return None

        return self._build_sample(before_path, after_path, action, meta={"url": url})

    def _generate_from_gui(self, dataset_entry: str) -> dict | None:
        """Load a pre-existing GUI screenshot pair and action annotation.

        Expects *dataset_entry* to be a path to a JSON file with keys:
        ``before_image``, ``after_image``, ``action``.
        """
        entry_path = Path(dataset_entry)
        if not entry_path.is_file():
            return None
        with open(entry_path) as fh:
            data = json.load(fh)

        before = data.get("before_image")
        after = data.get("after_image")
        action = data.get("action", {})
        if not before or not after:
            return None

        return self._build_sample(before, after, action, meta={"gui_entry": dataset_entry})

    def _generate_from_video(self, video_path: str) -> dict | None:
        """Extract consecutive frames around an action boundary in a video.

        Uses OpenCV to sample frames.  The caller should supply a video path;
        frame selection is done internally.
        """
        frames = self._extract_boundary_frames(video_path)
        if frames is None or len(frames) < 2:
            return None

        before_path, after_path = frames[0], frames[1]
        action = {"type": "video_transition", "source_video": video_path}
        return self._build_sample(before_path, after_path, action, meta={"video": video_path})

    def _generate_from_synthetic(self, source: str) -> dict | None:
        """Programmatically generate HTML, render, apply an action, re-render."""
        html_before, action = self._generate_synthetic_html(source)
        before_path = self._render_html(html_before)

        html_after = self._apply_synthetic_action(html_before, action)
        after_path = self._render_html(html_after)

        return self._build_sample(
            before_path, after_path, action, meta={"synthetic_seed": source}
        )

    def _build_sample(
        self,
        before_path: str,
        after_path: str,
        action: dict | str,
        meta: dict | None = None,
    ) -> dict:
        if isinstance(action, str):
            action = {"type": "generic", "description": action}

        before_desc = self.annotator.describe_state(before_path)
        after_desc = self.annotator.describe_state(after_path)
        transition_desc = self.annotator.compare_states(before_path, after_path)

        return {
            "id": str(uuid.uuid4()),
            "type": "vstp",
            "before_image": before_path,
            "after_image": after_path,
            "before_state": before_desc,
            "after_state": after_desc,
            "action": action,
            "transition_description": transition_desc,
            "meta": meta or {},
        }

    def _filter(self, sample: dict) -> bool:
        transition = sample.get("transition_description", "")
        if not transition or len(transition) < 20:
            return False

        trivial_markers = ["no change", "nothing changed", "identical", "same as before"]
        if any(m in transition.lower() for m in trivial_markers):
            return False

        before_state = sample.get("before_state", "")
        after_state = sample.get("after_state", "")
        if before_state == after_state:
            return False

        return True

    def _render_html(self, html: str) -> str:
        """Render HTML to a screenshot and return the file path.

        Attempts to use Playwright for real rendering; falls back to saving the
        HTML and returning a deterministic stub path.
        """
        tmp_dir = Path(tempfile.gettempdir()) / "ppv_vstp_renders"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        html_hash = hashlib.md5(html.encode()).hexdigest()[:12]
        screenshot_path = str(tmp_dir / f"{html_hash}.png")

        if os.path.exists(screenshot_path):
            return screenshot_path

        try:
            from playwright.sync_api import sync_playwright

            html_path = str(tmp_dir / f"{html_hash}.html")
            with open(html_path, "w") as fh:
                fh.write(html)

            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1280, "height": 720})
                page.goto(f"file://{html_path}")
                page.wait_for_load_state("networkidle")
                page.screenshot(path=screenshot_path)
                browser.close()

            return screenshot_path
        except Exception:
            logger.debug("Playwright unavailable; saving HTML as stub screenshot path")
            stub_path = str(tmp_dir / f"{html_hash}_stub.html")
            with open(stub_path, "w") as fh:
                fh.write(html)
            return stub_path

    def _render_url(self, url: str) -> str | None:
        """Render a live URL to a screenshot."""
        try:
            from playwright.sync_api import sync_playwright

            tmp_dir = Path(tempfile.gettempdir()) / "ppv_vstp_renders"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            path = str(tmp_dir / f"web_{url_hash}.png")

            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1280, "height": 720})
                page.goto(url, wait_until="networkidle", timeout=15000)
                page.screenshot(path=path)
                browser.close()
            return path
        except Exception:
            logger.debug("Cannot render URL %s", url, exc_info=True)
            return None

    def _apply_web_action(self, url: str, action: dict) -> str | None:
        """Apply an action on a live page and screenshot the result."""
        try:
            from playwright.sync_api import sync_playwright

            tmp_dir = Path(tempfile.gettempdir()) / "ppv_vstp_renders"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            action_hash = hashlib.md5(
                (url + json.dumps(action, sort_keys=True)).encode()
            ).hexdigest()[:12]
            path = str(tmp_dir / f"web_after_{action_hash}.png")

            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1280, "height": 720})
                page.goto(url, wait_until="networkidle", timeout=15000)

                selector = action.get("selector", "button")
                action_type = action.get("type", "click")
                if action_type == "click":
                    page.click(selector, timeout=5000)
                elif action_type == "fill":
                    page.fill(selector, action.get("value", "test"), timeout=5000)
                page.wait_for_timeout(500)
                page.screenshot(path=path)
                browser.close()
            return path
        except Exception:
            logger.debug("Cannot apply web action on %s", url, exc_info=True)
            return None

    def _extract_boundary_frames(self, video_path: str) -> list[str] | None:
        """Extract two frames around a detected change boundary in a video."""
        try:
            import cv2
        except ImportError:
            logger.debug("OpenCV not available for video frame extraction")
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 10:
            cap.release()
            return None

        tmp_dir = Path(tempfile.gettempdir()) / "ppv_vstp_frames"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        mid = total_frames // 2
        paths: list[str] = []
        for offset in (-2, 2):
            frame_idx = max(0, min(mid + offset, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None
            path = str(tmp_dir / f"{Path(video_path).stem}_{frame_idx}.png")
            cv2.imwrite(path, frame)
            paths.append(path)

        cap.release()
        return paths

    @staticmethod
    def _pick_random_web_action() -> dict:
        actions = [
            {"type": "click", "selector": "button", "description": "click a button"},
            {"type": "click", "selector": "a", "description": "click a link"},
            {"type": "fill", "selector": "input[type=text]", "value": "hello", "description": "type into text input"},
            {"type": "click", "selector": "input[type=checkbox]", "description": "toggle a checkbox"},
        ]
        return random.choice(actions)

    @staticmethod
    def _generate_synthetic_html(seed: str) -> tuple[str, dict]:
        """Create a simple interactive HTML page and a corresponding action."""
        rng = random.Random(seed)
        title = rng.choice(["Settings", "Dashboard", "Profile", "Search", "Checkout"])
        display = rng.choice(["Count: 0", "Status: Ready", "Items: 3"])

        num_elements = rng.randint(2, 5)
        elements = []
        for i in range(num_elements):
            template = rng.choice(_ELEMENT_TEMPLATES)
            label = rng.choice(["Submit", "Cancel", "Next", "Save", "Delete", "Search", "Apply"])
            elements.append(template.format(i=i, label=label))

        html = SYNTHETIC_HTML_TEMPLATE.format(
            title=title,
            display_text=display,
            elements="\n".join(elements),
        )

        target_idx = rng.randint(0, num_elements - 1)
        action = {
            "type": "click",
            "selector": f"#btn-{target_idx}, #input-{target_idx}, #check-{target_idx}, "
                        f"#select-{target_idx}, #link-{target_idx}",
            "target_index": target_idx,
            "description": f"interact with element {target_idx}",
        }
        return html, action

    @staticmethod
    def _apply_synthetic_action(html: str, action: dict) -> str:
        """Simulate an action by mutating the HTML string."""
        target_idx = action.get("target_index", 0)

        html = html.replace(
            f'id="display">',
            f'id="display" style="color: green;">',
        )

        old_btn = f'id="btn-{target_idx}"'
        if old_btn in html:
            html = html.replace(old_btn, f'id="btn-{target_idx}" disabled')
            label_start = html.find(old_btn.replace(old_btn, f'id="btn-{target_idx}" disabled'))
            html = html.replace(f">{action.get('description', 'Click')}</button>", ">Clicked!</button>", 1)

        old_check = f'id="check-{target_idx}"'
        if old_check in html:
            html = html.replace(old_check, f'id="check-{target_idx}" checked')

        return html
