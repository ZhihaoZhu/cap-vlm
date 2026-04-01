from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from ppv.evaluation.metrics import (
    compute_element_accuracy,
    compute_pass_at_k,
    compute_step_success_rate,
)

logger = logging.getLogger(__name__)

SUPPORTED_BENCHMARKS = {
    "screenspot",
    "mind2web",
    "browsecomp",
    "vqav2",
    "gqa",
    "mme",
    "textvqa",
}


def _load_jsonl(path: str | Path) -> list[dict]:
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _load_json_or_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return _load_jsonl(path)
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if "data" in data:
        return data["data"]
    return [data]


def _parse_bbox_from_text(text: str) -> list[float] | None:
    import re

    patterns = [
        r"\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]",
        r"\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return [float(match.group(i)) for i in range(1, 5)]
    return None


def _try_load_image(image_path: str | Path):
    try:
        from PIL import Image

        return Image.open(image_path).convert("RGB")
    except Exception:
        return None


class AgenticBenchmarkEvaluator:
    """Evaluator for downstream agentic and VLM benchmarks."""

    def __init__(self, model, tokenizer, image_processor):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"

    @torch.no_grad()
    def _generate_text(
        self,
        prompt: str,
        images: list | None = None,
        max_new_tokens: int = 512,
    ) -> str:
        if hasattr(self.model, "processor"):
            processor = self.model.processor
            if images:
                inputs = processor(text=prompt, images=images, return_tensors="pt", padding=True)
            else:
                inputs = processor(text=prompt, return_tensors="pt", padding=True)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)

        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        if hasattr(self.model, "generate"):
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        else:
            output_ids = self.model.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def evaluate(self, benchmark: str, data_path: str) -> dict:
        benchmark = benchmark.lower().strip()
        if benchmark not in SUPPORTED_BENCHMARKS:
            raise ValueError(
                f"Unsupported benchmark '{benchmark}'. Supported: {SUPPORTED_BENCHMARKS}"
            )

        dispatch = {
            "screenspot": self._evaluate_screenspot,
            "mind2web": self._evaluate_mind2web,
            "browsecomp": self._evaluate_browsecomp,
            "vqav2": lambda p: self._evaluate_general_vlm(p, "vqav2"),
            "gqa": lambda p: self._evaluate_general_vlm(p, "gqa"),
            "mme": lambda p: self._evaluate_general_vlm(p, "mme"),
            "textvqa": lambda p: self._evaluate_general_vlm(p, "textvqa"),
        }
        logger.info("Running %s evaluation from %s", benchmark, data_path)
        results = dispatch[benchmark](data_path)
        logger.info("%s results: %s", benchmark, results)
        return results

    def _evaluate_screenspot(self, data_path: str) -> dict:
        """ScreenSpot GUI grounding: given a screenshot and element description, predict bbox.

        Data format (JSONL):
            {"image": "path/to/img.png", "instruction": "Click the search bar",
             "bbox": [x1, y1, x2, y2]}
        """
        data = _load_json_or_jsonl(data_path)
        data_dir = Path(data_path).parent

        correct = 0
        total = 0
        per_type_correct: dict[str, int] = {}
        per_type_total: dict[str, int] = {}

        for sample in tqdm(data, desc="ScreenSpot"):
            image_path = data_dir / sample["image"] if not Path(sample["image"]).is_absolute() else Path(sample["image"])
            image = _try_load_image(image_path)
            instruction = sample["instruction"]
            gt_bbox = sample["bbox"]

            prompt = (
                f"Given this screenshot, locate the UI element described below.\n"
                f"Element: {instruction}\n"
                "Output the bounding box as [x1, y1, x2, y2] in pixel coordinates."
            )
            images = [image] if image is not None else None
            output = self._generate_text(prompt, images=images)

            pred_bbox = _parse_bbox_from_text(output)
            elem_type = sample.get("type", "unknown")
            per_type_total[elem_type] = per_type_total.get(elem_type, 0) + 1

            if pred_bbox is not None:
                is_correct = compute_element_accuracy(pred_bbox, gt_bbox)
                if is_correct:
                    correct += 1
                    per_type_correct[elem_type] = per_type_correct.get(elem_type, 0) + 1
            total += 1

        per_type_accuracy = {}
        for t, cnt in per_type_total.items():
            per_type_accuracy[t] = per_type_correct.get(t, 0) / cnt

        return {
            "benchmark": "screenspot",
            "element_accuracy": correct / max(total, 1),
            "correct": correct,
            "total": total,
            "per_type_accuracy": per_type_accuracy,
        }

    def _evaluate_mind2web(self, data_path: str) -> dict:
        """Mind2Web web navigation: multi-step web tasks.

        Data format (JSONL):
            {"task": "Book a flight...", "steps": [
                {"image": "step_0.png", "instruction": "...",
                 "action": {"action_type": "click", "bbox": [...]}, ...},
                ...
            ]}
        """
        data = _load_json_or_jsonl(data_path)
        data_dir = Path(data_path).parent

        all_step_success_rates = []
        element_correct = 0
        element_total = 0
        task_success = 0
        task_total = 0

        for sample in tqdm(data, desc="Mind2Web"):
            task_desc = sample["task"]
            steps = sample["steps"]
            predicted_actions = []
            gt_actions = []

            all_correct = True
            for step in steps:
                image_path = (
                    data_dir / step["image"]
                    if not Path(step["image"]).is_absolute()
                    else Path(step["image"])
                )
                image = _try_load_image(image_path)
                step_instruction = step.get("instruction", task_desc)
                gt_action = step["action"]
                gt_actions.append(gt_action)

                prompt = (
                    f"Task: {task_desc}\n"
                    f"Current step instruction: {step_instruction}\n"
                    "Given the current screenshot, predict the next action.\n"
                    "Output format: ACTION_TYPE: <type>, BBOX: [x1, y1, x2, y2]"
                )
                images = [image] if image is not None else None
                output = self._generate_text(prompt, images=images)

                pred_bbox = _parse_bbox_from_text(output)
                output_lower = output.lower()
                pred_action_type = "click"
                for atype in ("click", "type", "scroll", "select", "hover", "press"):
                    if atype in output_lower:
                        pred_action_type = atype
                        break

                pred_action = {"action_type": pred_action_type}
                if pred_bbox is not None:
                    pred_action["bbox"] = pred_bbox
                predicted_actions.append(pred_action)

                if gt_action.get("bbox") and pred_bbox is not None:
                    if compute_element_accuracy(pred_bbox, gt_action["bbox"]):
                        element_correct += 1
                    else:
                        all_correct = False
                else:
                    all_correct = False
                element_total += 1

            ssr = compute_step_success_rate(predicted_actions, gt_actions)
            all_step_success_rates.append(ssr)
            if all_correct and len(steps) > 0:
                task_success += 1
            task_total += 1

        n_tasks = max(task_total, 1)
        return {
            "benchmark": "mind2web",
            "element_accuracy": element_correct / max(element_total, 1),
            "step_success_rate": sum(all_step_success_rates) / n_tasks,
            "task_success_rate": task_success / n_tasks,
            "total_tasks": task_total,
            "total_steps": element_total,
        }

    def _evaluate_browsecomp(self, data_path: str) -> dict:
        """BrowseComp deep research: answer complex questions requiring multi-step reasoning.

        Data format (JSONL):
            {"question": "...", "answer": "...", "images": ["img1.png", ...]}
        """
        data = _load_json_or_jsonl(data_path)
        data_dir = Path(data_path).parent

        results_correct = []

        for sample in tqdm(data, desc="BrowseComp"):
            question = sample["question"]
            gt_answer = sample["answer"].strip().lower()

            images = []
            for img_path in sample.get("images", []):
                full_path = (
                    data_dir / img_path
                    if not Path(img_path).is_absolute()
                    else Path(img_path)
                )
                img = _try_load_image(full_path)
                if img is not None:
                    images.append(img)

            prompt = (
                f"Answer the following question based on the provided information.\n"
                f"Question: {question}\n"
                "Provide a concise, direct answer."
            )
            output = self._generate_text(
                prompt,
                images=images if images else None,
                max_new_tokens=256,
            )

            output_normalized = output.strip().lower()
            is_correct = gt_answer in output_normalized or output_normalized in gt_answer
            results_correct.append(is_correct)

        pass1 = compute_pass_at_k(results_correct, k=1)
        accuracy = sum(results_correct) / max(len(results_correct), 1)

        return {
            "benchmark": "browsecomp",
            "pass_at_1": pass1,
            "accuracy": accuracy,
            "correct": sum(results_correct),
            "total": len(results_correct),
        }

    def _evaluate_general_vlm(self, data_path: str, benchmark_name: str) -> dict:
        """General VLM benchmarks (VQAv2, GQA, MME, TextVQA).

        Data format (JSONL):
            {"image": "path.png", "question": "...", "answer": "..."}
        For MME, answer is "yes" or "no".
        """
        data = _load_json_or_jsonl(data_path)
        data_dir = Path(data_path).parent

        correct = 0
        total = 0
        category_correct: dict[str, int] = {}
        category_total: dict[str, int] = {}

        for sample in tqdm(data, desc=benchmark_name):
            image_path = (
                data_dir / sample["image"]
                if not Path(sample["image"]).is_absolute()
                else Path(sample["image"])
            )
            image = _try_load_image(image_path)
            question = sample["question"]
            gt_answer = sample["answer"].strip().lower()

            if benchmark_name == "mme":
                prompt = f"Look at this image and answer the question with yes or no.\nQuestion: {question}\nAnswer:"
            else:
                prompt = f"Look at this image and answer the question concisely.\nQuestion: {question}\nAnswer:"

            images = [image] if image is not None else None
            output = self._generate_text(prompt, images=images, max_new_tokens=64)
            pred = output.strip().lower()

            if benchmark_name == "mme":
                is_correct = ("yes" in pred and gt_answer == "yes") or (
                    "no" in pred and "yes" not in pred and gt_answer == "no"
                )
            else:
                is_correct = gt_answer in pred or pred in gt_answer

            if is_correct:
                correct += 1
            total += 1

            cat = sample.get("category", "overall")
            category_total[cat] = category_total.get(cat, 0) + 1
            if is_correct:
                category_correct[cat] = category_correct.get(cat, 0) + 1

        per_category_accuracy = {}
        for cat, cnt in category_total.items():
            per_category_accuracy[cat] = category_correct.get(cat, 0) / cnt

        return {
            "benchmark": benchmark_name,
            "accuracy": correct / max(total, 1),
            "correct": correct,
            "total": total,
            "per_category_accuracy": per_category_accuracy,
        }
