#!/usr/bin/env python3
"""Data synthesis entry point for PPV-CPT."""

import argparse
import logging
import sys

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SYNTHESIZER_REGISTRY = {
    "apc": "ppv.synthesis.apc_synthesizer.APCSynthesizer",
    "vstp": "ppv.synthesis.vstp_synthesizer.VSTSynthesizer",
    "hvc": "ppv.synthesis.hvc_synthesizer.HVCSynthesizer",
}


def parse_args():
    parser = argparse.ArgumentParser(description="PPV-CPT Data Synthesis")
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["apc", "vstp", "hvc"],
        help="Type of data to synthesize",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Directory containing source data (e.g., raw screenshots, web pages)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write synthesized data",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--annotator-model",
        type=str,
        default="gpt-4o",
        help="Model to use for annotation / verification",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for synthesis",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def load_synthesizer(synth_type: str):
    """Dynamically import and return the synthesizer class."""
    qualified_name = SYNTHESIZER_REGISTRY[synth_type]
    module_path, class_name = qualified_name.rsplit(".", 1)

    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main():
    args = parse_args()

    logger.info("Synthesis type: %s", args.type)
    logger.info("Source dir: %s", args.source_dir)
    logger.info("Output dir: %s", args.output_dir)
    logger.info("Num samples: %d", args.num_samples)
    logger.info("Annotator model: %s", args.annotator_model)
    logger.info("Num workers: %d", args.num_workers)

    SynthesizerClass = load_synthesizer(args.type)
    synthesizer = SynthesizerClass(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        annotator_model=args.annotator_model,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    logger.info("Starting synthesis...")
    stats = synthesizer.run(num_samples=args.num_samples)
    logger.info("Synthesis complete. Stats: %s", stats)


if __name__ == "__main__":
    main()
