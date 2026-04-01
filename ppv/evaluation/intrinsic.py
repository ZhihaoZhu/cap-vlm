from __future__ import annotations

import logging
import re
from typing import Any

import torch
from tqdm import tqdm

from ppv.evaluation.metrics import compute_rouge

logger = logging.getLogger(__name__)


def _parse_regions(text: str) -> list[list[float]]:
    pattern = r"\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]"
    return [[float(m[0]), float(m[1]), float(m[2]), float(m[3])] for m in re.findall(pattern, text)]


def _box_overlap(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    if area_a <= 0:
        return 0.0
    return inter / area_a


class PPVIntrinsicEvaluator:
    """Evaluator for the three PPV intrinsic tasks: Perception, Prediction, Verification."""

    def __init__(self, model, tokenizer, image_processor):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"

    @torch.no_grad()
    def _generate_text(self, prompt: str, images: list | None = None) -> str:
        if hasattr(self.model, "processor"):
            processor = self.model.processor
            if images:
                inputs = processor(text=prompt, images=images, return_tensors="pt", padding=True)
            else:
                inputs = processor(text=prompt, return_tensors="pt", padding=True)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)

        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        if hasattr(self.model, "generate"):
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )
        else:
            output_ids = self.model.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def evaluate_perception_quality(self, eval_data: list[dict]) -> dict:
        """Evaluate Agentic Perception Chaining (APC) quality.

        Each item in eval_data should have:
            - "image": PIL Image or path
            - "task": str task description
            - "relevant_regions": list of [x1,y1,x2,y2] bounding boxes of task-relevant regions
            - "expected_conclusion": str ground-truth conclusion
        """
        region_relevances = []
        chain_completeness_scores = []
        spatial_grounding_scores = []

        for sample in tqdm(eval_data, desc="Evaluating perception"):
            prompt = (
                f"You are given a screenshot. Task: {sample['task']}\n"
                "Produce an agentic perception chain: identify relevant regions with bounding "
                "boxes [x1,y1,x2,y2], describe each region, and reach a conclusion about the task."
            )
            output = self._generate_text(prompt, images=sample.get("image"))

            predicted_regions = _parse_regions(output)
            gt_regions = sample.get("relevant_regions", [])

            if gt_regions and predicted_regions:
                hits = 0
                for pred_box in predicted_regions:
                    for gt_box in gt_regions:
                        if _box_overlap(pred_box, gt_box) > 0.3:
                            hits += 1
                            break
                region_relevances.append(hits / len(predicted_regions))
            elif not gt_regions and not predicted_regions:
                region_relevances.append(1.0)
            else:
                region_relevances.append(0.0)

            has_region = len(predicted_regions) > 0
            conclusion_keywords = ("therefore", "conclusion", "thus", "result", "answer")
            has_conclusion = any(kw in output.lower() for kw in conclusion_keywords)
            completeness = (0.5 * float(has_region)) + (0.5 * float(has_conclusion))
            chain_completeness_scores.append(completeness)

            expected = sample.get("expected_conclusion", "")
            if expected:
                rouge = compute_rouge(output, expected)
                spatial_grounding_scores.append(rouge["rougeL"]["f1"])
            else:
                spatial_grounding_scores.append(0.0)

        n = max(len(eval_data), 1)
        return {
            "region_relevance": sum(region_relevances) / n,
            "chain_completeness": sum(chain_completeness_scores) / n,
            "spatial_grounding_accuracy": sum(spatial_grounding_scores) / n,
            "num_samples": len(eval_data),
        }

    def evaluate_prediction_quality(self, eval_data: list[dict]) -> dict:
        """Evaluate Visual State Transition Prediction (VSTP) quality.

        Each item in eval_data should have:
            - "before_image": PIL Image or path
            - "action": str action description
            - "ground_truth_transition": str describing what actually changed
        """
        transition_rouge_scores = []
        action_relevance_scores = []

        for sample in tqdm(eval_data, desc="Evaluating prediction"):
            action = sample["action"]
            prompt = (
                f"You are given a screenshot showing the state before an action.\n"
                f"Action performed: {action}\n"
                "Predict what visual changes will occur after this action. "
                "Describe the expected state transition in detail."
            )
            output = self._generate_text(prompt, images=sample.get("before_image"))

            gt_transition = sample.get("ground_truth_transition", "")
            if gt_transition:
                rouge = compute_rouge(output, gt_transition)
                transition_rouge_scores.append(rouge["rougeL"]["f1"])
            else:
                transition_rouge_scores.append(0.0)

            action_terms = set(action.lower().split())
            output_terms = set(output.lower().split())
            stopwords = {"the", "a", "an", "is", "on", "in", "to", "of", "and", "for", "with"}
            action_terms -= stopwords
            if action_terms:
                overlap = len(action_terms & output_terms) / len(action_terms)
                action_relevance_scores.append(overlap)
            else:
                action_relevance_scores.append(0.0)

        n = max(len(eval_data), 1)
        rouge_avg = sum(transition_rouge_scores) / n
        return {
            "transition_accuracy_rougeL": rouge_avg,
            "transition_accuracy_avg_rouge": rouge_avg,
            "action_relevance": sum(action_relevance_scores) / n,
            "num_samples": len(eval_data),
        }

    def evaluate_verification_quality(self, eval_data: list[dict]) -> dict:
        """Evaluate Hypothesis Verification & Correction (HVC) quality.

        Each item in eval_data should have:
            - "image": PIL Image or path (the actual post-action screenshot)
            - "hypothesis": str prediction about what should have happened
            - "expected_state": str what was expected
            - "actual_state": str what actually happened
            - "is_correct": bool whether the hypothesis matches reality
            - "ground_truth_correction": str (only when is_correct=False)
        """
        assessment_correct = []
        correction_scores = []

        for sample in tqdm(eval_data, desc="Evaluating verification"):
            hypothesis = sample["hypothesis"]
            expected = sample["expected_state"]
            prompt = (
                f"You predicted: \"{hypothesis}\"\n"
                f"Expected state: \"{expected}\"\n"
                "Look at the actual screenshot. Is the prediction correct? "
                "Answer with CORRECT or INCORRECT, then explain. "
                "If incorrect, provide a corrective plan."
            )
            output = self._generate_text(prompt, images=sample.get("image"))

            output_lower = output.lower()
            predicted_correct = "correct" in output_lower and "incorrect" not in output_lower
            gt_correct = sample.get("is_correct", True)
            assessment_correct.append(float(predicted_correct == gt_correct))

            if not gt_correct:
                gt_correction = sample.get("ground_truth_correction", "")
                if gt_correction:
                    rouge = compute_rouge(output, gt_correction)
                    correction_scores.append(rouge["rougeL"]["f1"])
                else:
                    correction_scores.append(0.0)

        n = max(len(eval_data), 1)
        n_corrections = max(len(correction_scores), 1)
        return {
            "assessment_accuracy": sum(assessment_correct) / n,
            "correction_quality": sum(correction_scores) / n_corrections,
            "num_samples": len(eval_data),
            "num_incorrect_cases": len(correction_scores),
        }

    def evaluate_all(self, eval_data: dict) -> dict:
        """Run all three intrinsic evaluations.

        eval_data should be a dict with keys:
            - "perception": list[dict] for APC evaluation
            - "prediction": list[dict] for VSTP evaluation
            - "verification": list[dict] for HVC evaluation
        """
        results = {}

        if "perception" in eval_data and eval_data["perception"]:
            results["perception"] = self.evaluate_perception_quality(eval_data["perception"])
            logger.info("Perception: %s", results["perception"])

        if "prediction" in eval_data and eval_data["prediction"]:
            results["prediction"] = self.evaluate_prediction_quality(eval_data["prediction"])
            logger.info("Prediction: %s", results["prediction"])

        if "verification" in eval_data and eval_data["verification"]:
            results["verification"] = self.evaluate_verification_quality(eval_data["verification"])
            logger.info("Verification: %s", results["verification"])

        aggregate = {}
        for task, metrics in results.items():
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    aggregate[f"{task}/{k}"] = v

        results["aggregate"] = aggregate
        return results
