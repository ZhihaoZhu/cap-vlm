#!/usr/bin/env python3
"""Evaluation entry point for PPV-CPT."""

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

ALL_BENCHMARKS = [
    "intrinsic",
    "gui_web",
    "deep_research",
    "visual_reasoning",
    "general_vlm",
]

BENCHMARK_EVALUATORS = {
    "intrinsic": ("ppv.evaluation.intrinsic", "PPVIntrinsicEvaluator"),
    "gui_web": ("ppv.evaluation.agentic", "AgenticBenchmarkEvaluator"),
    "deep_research": ("ppv.evaluation.agentic", "AgenticBenchmarkEvaluator"),
    "visual_reasoning": ("ppv.evaluation.agentic", "AgenticBenchmarkEvaluator"),
    "general_vlm": ("ppv.evaluation.agentic", "AgenticBenchmarkEvaluator"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="PPV-CPT Evaluation")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint or HuggingFace model name",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["intrinsic"],
        choices=ALL_BENCHMARKS,
        help="Benchmarks to run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per benchmark (for quick testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on",
    )
    return parser.parse_args()


def load_evaluator(benchmark_name: str):
    """Dynamically import and return the evaluator class for a benchmark."""
    import importlib

    module_path, class_name = BENCHMARK_EVALUATORS[benchmark_name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Model: %s", args.model_path)
    logger.info("Benchmarks: %s", args.benchmarks)
    logger.info("Output dir: %s", args.output_dir)

    from ppv.models import VLMWrapper

    logger.info("Loading model from %s ...", args.model_path)
    model = VLMWrapper(model_name_or_path=args.model_path)
    model.eval()

    all_results = {}

    for benchmark in args.benchmarks:
        logger.info("Running benchmark: %s", benchmark)

        EvaluatorClass = load_evaluator(benchmark)
        evaluator = EvaluatorClass(
            benchmark_name=benchmark,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            device=args.device,
        )

        results = evaluator.evaluate(model)
        all_results[benchmark] = results

        benchmark_path = output_dir / f"{benchmark}_results.json"
        with open(benchmark_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("  %s -> %s", benchmark, {k: f"{v:.4f}" for k, v in results.items()})

    summary_path = output_dir / "all_results.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("All results saved to %s", summary_path)

    # Print summary table
    logger.info("=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)
    for benchmark, results in all_results.items():
        primary_metric = next(iter(results))
        logger.info("  %-20s  %s = %.4f", benchmark, primary_metric, results[primary_metric])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
