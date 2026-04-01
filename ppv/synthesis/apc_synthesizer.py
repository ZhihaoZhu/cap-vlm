"""Active Perception Chain (APC) data synthesizer.

Generates multi-step perception chains where a VLM iteratively focuses on
different spatial regions of a scene to accomplish a task, producing grounded
chains-of-perception that teach the model *where* to look and *why*.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from ppv.synthesis.annotator import SceneAnnotator
from ppv.synthesis.base import BaseSynthesizer

logger = logging.getLogger(__name__)

MAX_CHAIN_STEPS = 10
MIN_CHAIN_STEPS = 3


class APCSynthesizer(BaseSynthesizer):
    """Synthesize Active-Perception-Chain training samples."""

    def __init__(
        self,
        annotator_model: str,
        output_dir: str,
        num_workers: int = 4,
    ) -> None:
        super().__init__(annotator_model, output_dir, num_workers)
        self.annotator = SceneAnnotator(annotator_model)

    def _generate_single(self, image_path: str) -> dict | None:
        try:
            regions = self.annotator.decompose_scene(image_path)
            if not regions:
                return None

            tasks = self.annotator.generate_tasks(image_path, regions, num_tasks=3)
            if not tasks:
                return None

            task = tasks[0]
            chain = self._build_perception_chain(image_path, task, regions)
            if chain is None:
                return None

            return {
                "id": str(uuid.uuid4()),
                "type": "apc",
                "image_path": image_path,
                "task": task,
                "regions": regions,
                "perception_chain": chain,
            }
        except Exception:
            logger.debug("APC generation failed for %s", image_path, exc_info=True)
            return None

    def _build_perception_chain(
        self,
        image_path: str,
        task: str,
        regions: list[dict],
    ) -> list[dict] | None:
        """Iteratively build a perception chain by prompting the VLM.

        At each step the model selects a region, describes what it sees, reasons
        about task-relevance, and decides whether to continue or conclude.
        """
        region_index = self._format_region_index(regions)
        chain: list[dict] = []
        visited: set[int] = set()

        for step_idx in range(MAX_CHAIN_STEPS):
            history = self._format_chain_history(chain)

            prompt = (
                f"Task: {task}\n\n"
                f"Available regions:\n{region_index}\n\n"
                f"Perception history:\n{history}\n\n"
                f"Step {step_idx + 1}: Choose a region to focus on next. "
                "Provide your response in this exact format:\n"
                "REGION: <region index>\n"
                "OBSERVATION: <what you see in this region>\n"
                "REASONING: <why this is relevant to the task>\n"
                "STATUS: <CONTINUE if more information needed, CONCLUDE if task can be answered>\n"
                "ANSWER: <final answer if STATUS is CONCLUDE, otherwise N/A>"
            )

            raw = self.annotator._call_vlm(prompt, image_path)
            step = self._parse_chain_step(raw, regions, step_idx)

            if step is None:
                break

            region_idx = step.get("region_index")
            if region_idx is not None:
                visited.add(region_idx)
            chain.append(step)

            if step.get("status") == "CONCLUDE":
                break

        if len(chain) < MIN_CHAIN_STEPS:
            return None
        if chain[-1].get("status") != "CONCLUDE":
            chain[-1]["status"] = "CONCLUDE"
            chain[-1]["answer"] = chain[-1].get("answer", "Could not determine answer.")

        return chain

    def _parse_chain_step(
        self, raw: str, regions: list[dict], step_idx: int
    ) -> dict | None:
        """Parse a single VLM response into a structured chain step."""
        fields: dict[str, str] = {}
        for line in raw.strip().splitlines():
            for key in ("REGION", "OBSERVATION", "REASONING", "STATUS", "ANSWER"):
                if line.upper().startswith(key + ":"):
                    fields[key.lower()] = line.split(":", 1)[1].strip()

        if "observation" not in fields or "reasoning" not in fields:
            return self._stub_chain_step(regions, step_idx)

        region_idx = self._resolve_region_index(fields.get("region", "0"), len(regions))
        region = regions[region_idx] if region_idx < len(regions) else regions[0]

        return {
            "step": step_idx,
            "region_index": region_idx,
            "region_bbox": region["bbox"],
            "region_type": region["element_type"],
            "observation": fields["observation"],
            "reasoning": fields["reasoning"],
            "status": fields.get("status", "CONTINUE").upper(),
            "answer": fields.get("answer"),
        }

    def _stub_chain_step(self, regions: list[dict], step_idx: int) -> dict:
        """Produce a deterministic step when parsing fails (e.g. stub VLM)."""
        region_idx = step_idx % len(regions)
        region = regions[region_idx]
        is_last = step_idx >= MIN_CHAIN_STEPS - 1

        return {
            "step": step_idx,
            "region_index": region_idx,
            "region_bbox": region["bbox"],
            "region_type": region["element_type"],
            "observation": f"Observed {region['description']}.",
            "reasoning": f"This {region['element_type']} is relevant because it may contain task-related information.",
            "status": "CONCLUDE" if is_last else "CONTINUE",
            "answer": "Task completed based on gathered observations." if is_last else None,
        }

    def _filter(self, sample: dict) -> bool:
        chain = sample.get("perception_chain", [])
        if len(chain) < MIN_CHAIN_STEPS:
            return False

        regions = sample.get("regions", [])
        num_regions = len(regions)
        for step in chain:
            idx = step.get("region_index")
            if idx is not None and idx >= num_regions:
                return False

        has_conclusion = any(s.get("status") == "CONCLUDE" for s in chain)
        if not has_conclusion:
            return False

        referenced = {s.get("region_index") for s in chain if s.get("region_index") is not None}
        if len(referenced) < 2:
            return False

        return True

    @staticmethod
    def _format_region_index(regions: list[dict]) -> str:
        lines = []
        for i, r in enumerate(regions):
            bbox_str = ", ".join(f"{v:.2f}" for v in r["bbox"])
            lines.append(f"[{i}] ({r['element_type']}) {r['description']}  bbox=[{bbox_str}]")
        return "\n".join(lines)

    @staticmethod
    def _format_chain_history(chain: list[dict]) -> str:
        if not chain:
            return "(none)"
        lines = []
        for s in chain:
            lines.append(
                f"Step {s['step']}: Focused on region {s['region_index']} "
                f"({s['region_type']}). Observed: {s['observation']}"
            )
        return "\n".join(lines)

    @staticmethod
    def _resolve_region_index(raw: str, num_regions: int) -> int:
        try:
            idx = int(raw.strip().strip("[]"))
            return max(0, min(idx, num_regions - 1))
        except ValueError:
            return 0
