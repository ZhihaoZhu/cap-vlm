# PPV-CPT: Perceive, Predict, Verify

**Continual Pre-training for Multimodal Agentic Foundation Models**

PPV-CPT introduces a new training paradigm that transforms Vision-Language Models from passive viewers into active visual reasoners through continual pre-training. The PPV loop trains three mutually reinforcing capabilities:

1. **Perceive** (active) -- goal-directed visual attention that decides where to look and what to extract
2. **Predict** (ahead) -- anticipating how visual scenes change after actions (implicit world model)
3. **Verify** (check) -- comparing expected vs. actual outcomes and self-correcting

## Architecture

```
PERCEIVE (active) → PREDICT (ahead) → ACT → VERIFY (check) → UPDATE (correct) → loop
```

### Training Pipeline

```
VLM Base Model → CPT Stage 1 (32K, ~200B tokens) → CPT Stage 2 (128K, ~100B tokens) → SFT/RL
```

### Three Types of Synthetic CPT Data

| Data Type | Description | Stage 1 | Stage 2 |
|-----------|-------------|---------|---------|
| **APC** (Active Perception Chains) | Sequential, goal-directed visual examination | 40% | 20% |
| **VSTP** (Visual State Transition Predictions) | Predicting visual changes from actions | 40% | 20% |
| **HVC** (Hypothesis-Verification Chains) | Forming and verifying visual hypotheses | 10% | 40% |
| **General VL** | Anti-forgetting visual-language data | 10% | 20% |

## Installation

```bash
pip install -e .

# With synthesis dependencies
pip install -e ".[synthesis]"

# With evaluation dependencies
pip install -e ".[eval]"

# Full development install
pip install -e ".[synthesis,eval,dev]"
```

## Quick Start

### Data Synthesis

```bash
# Synthesize Active Perception Chain data
python scripts/synthesize.py \
    --type apc \
    --source-dir data/scenes/ \
    --output-dir data/apc/ \
    --annotator-model Qwen/Qwen2-VL-72B-Instruct \
    --num-samples 100000

# Synthesize Visual State Transition data
python scripts/synthesize.py \
    --type vstp \
    --source-dir data/web_interactions/ \
    --output-dir data/vstp/ \
    --num-samples 100000

# Synthesize Hypothesis-Verification Chain data
python scripts/synthesize.py \
    --type hvc \
    --source-dir data/trajectories/ \
    --output-dir data/hvc/ \
    --num-samples 50000
```

### Training

```bash
# Stage 1 CPT: Foundational perception + prediction (32K context)
accelerate launch scripts/train.py --config configs/stage1.yaml

# Stage 2 CPT: Full verify-and-correct loop (128K context)
accelerate launch scripts/train.py --config configs/stage2.yaml

# Post-CPT SFT
accelerate launch scripts/train.py --config configs/sft.yaml
```

### Evaluation

```bash
# Run intrinsic PPV evaluation
python scripts/evaluate.py \
    --model-path checkpoints/stage2/final \
    --benchmarks intrinsic \
    --output-dir results/

# Run downstream agentic benchmarks
python scripts/evaluate.py \
    --model-path checkpoints/sft/final \
    --benchmarks gui_web deep_research visual_reasoning general_vlm \
    --output-dir results/
```

## Project Structure

```
ppv-cpt/
├── configs/              # Training configurations
│   ├── base.yaml         # Shared base config
│   ├── stage1.yaml       # Stage 1 CPT config
│   ├── stage2.yaml       # Stage 2 CPT config
│   └── sft.yaml          # Post-CPT SFT config
├── ppv/
│   ├── data/             # Dataset classes
│   │   ├── base.py       # Base dataset class
│   │   ├── apc.py        # Active Perception Chain dataset
│   │   ├── vstp.py       # Visual State Transition dataset
│   │   ├── hvc.py        # Hypothesis-Verification Chain dataset
│   │   └── mixtures.py   # Data mixture sampler
│   ├── synthesis/        # Data synthesis pipelines
│   │   ├── annotator.py  # VLM scene annotator
│   │   ├── apc_synthesizer.py
│   │   ├── vstp_synthesizer.py
│   │   └── hvc_synthesizer.py
│   ├── models/           # Model wrappers
│   │   ├── vlm_wrapper.py
│   │   └── config.py
│   ├── training/         # Training infrastructure
│   │   ├── cpt_trainer.py
│   │   ├── config.py
│   │   └── lr_schedule.py
│   └── evaluation/       # Evaluation framework
│       ├── intrinsic.py  # PPV-specific metrics
│       ├── agentic.py    # Downstream benchmarks
│       └── metrics.py    # Shared metrics
├── scripts/              # Entry points
│   ├── train.py
│   ├── synthesize.py
│   └── evaluate.py
└── tests/
```

## Supported Base Models

- **Qwen2-VL** (7B / 72B) -- default
- **InternVL2** (8B / 40B / 76B)
- **LLaVA-OneVision** (7B / 72B)

## Key Ablations

| Experiment | Purpose |
|-----------|---------|
| PPV-CPT + SFT vs. Vanilla + SFT | Does CPT help? |
| APC-only vs. VSTP-only vs. HVC-only | Which component matters? |
| All-three vs. pairwise | Are they synergistic? |
| Stage 1 only vs. Stage 1+2 | Does progressive training help? |
| Data scaling (10B → 300B) | Scaling behavior |

## Citation

```bibtex
@article{ppv-cpt,
  title={Perceive, Predict, Verify: Continual Pre-training for Multimodal Agentic Foundation Models},
  year={2025}
}
```

## License

Apache 2.0
