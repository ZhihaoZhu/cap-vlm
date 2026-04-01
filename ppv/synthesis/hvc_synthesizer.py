"""Hypothesis-Verification Chain (HVC) data synthesizer.

Builds chains where the model predicts what an action will do (hypothesis),
specifies an expected outcome, and then compares with the actual next state.
Intentionally introduces wrong hypotheses with probability p=0.3 to train
self-correction capabilities.
"""

from __future__ import annotations

import json
import logging
import random
import uuid
from pathlib import Path

from ppv.synthesis.annotator import SceneAnnotator
from ppv.synthesis.base import BaseSynthesizer

logger = logging.getLogger(__name__)

WRONG_HYPOTHESIS_PROB = 0.3
MIN_TRAJECTORY_STEPS = 2


class HVCSynthesizer(BaseSynthesizer):
    """Synthesize Hypothesis-Verification-Chain training samples."""

    def __init__(
        self,
        annotator_model: str,
        output_dir: str,
        num_workers: int = 4,
    ) -> None:
        super().__init__(annotator_model, output_dir, num_workers)
        self.annotator = SceneAnnotator(annotator_model)

    def _generate_single(self, trajectory: dict) -> dict | None:
        """Build an HVC from a multi-step trajectory.

        Args:
            trajectory: dict with key ``steps`` -- a list of dicts, each having
                ``state_image``, ``action``, and ``next_state_image``.
        """
        if isinstance(trajectory, str):
            trajectory = self._load_trajectory(trajectory)
            if trajectory is None:
                return None

        steps = trajectory.get("steps", [])
        if len(steps) < MIN_TRAJECTORY_STEPS:
            return None

        try:
            chain = self._build_hvc_chain(steps)
        except Exception:
            logger.debug("HVC chain construction failed", exc_info=True)
            return None

        if chain is None or len(chain) < MIN_TRAJECTORY_STEPS:
            return None

        return {
            "id": str(uuid.uuid4()),
            "type": "hvc",
            "trajectory_id": trajectory.get("id", str(uuid.uuid4())),
            "chain": chain,
            "num_steps": len(chain),
            "num_wrong_hypotheses": sum(1 for s in chain if s.get("intentionally_wrong")),
        }

    def _build_hvc_chain(self, steps: list[dict]) -> list[dict] | None:
        chain: list[dict] = []

        for step_idx, step in enumerate(steps):
            state_image = step.get("state_image")
            action = step.get("action", {})
            next_state_image = step.get("next_state_image")

            if not state_image or not next_state_image:
                continue

            force_wrong = random.random() < WRONG_HYPOTHESIS_PROB

            state_desc = self.annotator.describe_state(state_image)

            if force_wrong:
                hypothesis = self._generate_wrong_hypothesis(state_image, action)
            else:
                hypothesis = self._generate_hypothesis(state_image, action)

            expected_outcome = self._generate_expected_outcome(
                state_desc, action, hypothesis
            )

            actual_desc = self.annotator.describe_state(next_state_image)

            assessment = self._assess_hypothesis(
                hypothesis, expected_outcome, actual_desc
            )

            corrective_plan = None
            if assessment["verdict"] == "incorrect":
                corrective_plan = self._generate_corrective_plan(
                    state_desc, action, hypothesis, actual_desc
                )

            chain.append({
                "step": step_idx,
                "state_image": state_image,
                "state_description": state_desc,
                "action": action if isinstance(action, dict) else {"description": str(action)},
                "hypothesis": hypothesis,
                "expected_outcome": expected_outcome,
                "next_state_image": next_state_image,
                "actual_outcome": actual_desc,
                "assessment": assessment,
                "intentionally_wrong": force_wrong,
                "corrective_plan": corrective_plan,
            })

        return chain if chain else None

    def _generate_hypothesis(self, state_image: str, action: dict | str) -> str:
        action_desc = action if isinstance(action, str) else action.get("description", str(action))
        prompt = (
            f"You are observing a UI screen. The next action to be performed is: "
            f"{action_desc}\n\n"
            "Based on the current state, predict what will happen after this action "
            "is performed. Be specific about which elements will change and how."
        )
        return self.annotator._call_vlm(prompt, state_image, temperature=0.3)

    def _generate_wrong_hypothesis(self, state_image: str, action: dict | str) -> str:
        action_desc = action if isinstance(action, str) else action.get("description", str(action))
        prompt = (
            f"You are observing a UI screen. The next action to be performed is: "
            f"{action_desc}\n\n"
            "Predict what will happen, but give a PLAUSIBLE BUT INCORRECT prediction. "
            "The prediction should sound reasonable but describe an outcome that would "
            "NOT actually happen. For example, predict a navigation to the wrong page, "
            "the wrong element changing, or the wrong kind of state transition."
        )
        return self.annotator._call_vlm(prompt, state_image, temperature=0.9)

    def _generate_expected_outcome(
        self, state_desc: str, action: dict | str, hypothesis: str
    ) -> str:
        action_desc = action if isinstance(action, str) else action.get("description", str(action))
        prompt = (
            f"Current state:\n{state_desc}\n\n"
            f"Action: {action_desc}\n\n"
            f"Hypothesis: {hypothesis}\n\n"
            "Based on this hypothesis, describe the specific expected outcome in "
            "concrete terms: which UI elements will be visible, what text will appear, "
            "what layout changes will occur."
        )
        return self.annotator._call_vlm(prompt, temperature=0.3)

    def _assess_hypothesis(
        self, hypothesis: str, expected_outcome: str, actual_outcome: str
    ) -> dict:
        prompt = (
            f"Hypothesis: {hypothesis}\n\n"
            f"Expected outcome: {expected_outcome}\n\n"
            f"Actual outcome: {actual_outcome}\n\n"
            "Assess whether the hypothesis was correct. Respond with:\n"
            "VERDICT: correct OR incorrect\n"
            "EXPLANATION: <brief explanation of why the assessment is correct or incorrect>"
        )
        raw = self.annotator._call_vlm(prompt, temperature=0.1)

        verdict = "incorrect"
        explanation = raw.strip()

        for line in raw.strip().splitlines():
            upper = line.upper()
            if upper.startswith("VERDICT:"):
                v = line.split(":", 1)[1].strip().lower()
                verdict = "correct" if "correct" in v and "incorrect" not in v else "incorrect"
            elif upper.startswith("EXPLANATION:"):
                explanation = line.split(":", 1)[1].strip()

        return {"verdict": verdict, "explanation": explanation}

    def _generate_corrective_plan(
        self,
        state_desc: str,
        action: dict | str,
        wrong_hypothesis: str,
        actual_outcome: str,
    ) -> str:
        action_desc = action if isinstance(action, str) else action.get("description", str(action))
        prompt = (
            f"The original hypothesis was wrong.\n\n"
            f"State before action: {state_desc}\n"
            f"Action taken: {action_desc}\n"
            f"Wrong hypothesis: {wrong_hypothesis}\n"
            f"Actual outcome: {actual_outcome}\n\n"
            "Explain what went wrong with the hypothesis and provide a corrective plan: "
            "what should the model have predicted instead, and what cues in the original "
            "state should have led to the correct prediction?"
        )
        return self.annotator._call_vlm(prompt, temperature=0.3)

    def _filter(self, sample: dict) -> bool:
        chain = sample.get("chain", [])
        if len(chain) < MIN_TRAJECTORY_STEPS:
            return False

        for step in chain:
            is_wrong = step.get("intentionally_wrong", False)
            verdict = step.get("assessment", {}).get("verdict", "")

            if is_wrong and verdict == "correct":
                return False

            if not is_wrong and verdict == "incorrect":
                return False

            if is_wrong and step.get("corrective_plan") is None:
                return False

        for step in chain:
            if not step.get("hypothesis") or not step.get("actual_outcome"):
                return False

        return True

    @staticmethod
    def _load_trajectory(path: str) -> dict | None:
        """Load a trajectory from a JSON file."""
        try:
            with open(path) as fh:
                data = json.load(fh)
            if "steps" not in data:
                if isinstance(data, list):
                    data = {"steps": data}
                else:
                    return None
            return data
        except (json.JSONDecodeError, OSError):
            logger.debug("Failed to load trajectory from %s", path)
            return None
