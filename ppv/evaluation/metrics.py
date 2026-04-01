from __future__ import annotations

import math
from collections import Counter


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _rouge_n(prediction_tokens: list[str], reference_tokens: list[str], n: int) -> dict:
    pred_ngrams = Counter(_ngrams(prediction_tokens, n))
    ref_ngrams = Counter(_ngrams(reference_tokens, n))
    overlap = sum((pred_ngrams & ref_ngrams).values())
    precision = overlap / max(sum(pred_ngrams.values()), 1)
    recall = overlap / max(sum(ref_ngrams.values()), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(cur[j - 1], prev[j])
        prev = cur
    return prev[n]


def compute_rouge(prediction: str, reference: str) -> dict:
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return {
            "rouge1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "rouge2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "rougeL": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        }

    rouge1 = _rouge_n(pred_tokens, ref_tokens, 1)
    rouge2 = _rouge_n(pred_tokens, ref_tokens, 2)

    lcs = _lcs_length(pred_tokens, ref_tokens)
    lcs_precision = lcs / max(len(pred_tokens), 1)
    lcs_recall = lcs / max(len(ref_tokens), 1)
    lcs_f1 = 2 * lcs_precision * lcs_recall / max(lcs_precision + lcs_recall, 1e-12)
    rouge_l = {"precision": lcs_precision, "recall": lcs_recall, "f1": lcs_f1}

    return {"rouge1": rouge1, "rouge2": rouge2, "rougeL": rouge_l}


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def compute_element_accuracy(
    predicted_bbox: list[float],
    ground_truth_bbox: list[float],
    threshold: float = 0.5,
) -> bool:
    return compute_iou(predicted_bbox, ground_truth_bbox) >= threshold


def compute_step_success_rate(
    predicted_actions: list[dict],
    ground_truth_actions: list[dict],
) -> float:
    if not ground_truth_actions:
        return 1.0 if not predicted_actions else 0.0

    correct = 0
    for i, gt in enumerate(ground_truth_actions):
        if i >= len(predicted_actions):
            break
        pred = predicted_actions[i]
        action_match = pred.get("action_type") == gt.get("action_type")
        if action_match and "bbox" in gt and "bbox" in pred:
            action_match = compute_element_accuracy(pred["bbox"], gt["bbox"])
        if action_match:
            correct += 1

    return correct / len(ground_truth_actions)


def compute_pass_at_k(results: list[bool], k: int = 1) -> float:
    n = len(results)
    if n == 0:
        return 0.0
    if k >= n:
        return 1.0 if any(results) else 0.0

    c = sum(results)
    if c == 0:
        return 0.0

    log_val = 0.0
    for i in range(k):
        log_val += math.log(max(n - c - i, 0) + 1e-30) - math.log(n - i)
    return 1.0 - math.exp(log_val)
