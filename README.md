<div align="center">

# AgentCPT

### Agentic Continual Pre-training for Vision-Language Foundation Models

*Teaching VLMs to **Perceive**, **Predict**, and **Verify** for Multimodal Agency*

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)

---

**AgentCPT** introduces a new continual pre-training paradigm that transforms
Vision-Language Models from **passive image viewers** into **active visual agents**
capable of goal-directed perception, predictive reasoning, and self-correction.

[Overview](#overview) | [Method](#the-ppv-loop) | [Quick Start](#quick-start) | [Benchmarks](#evaluation) | [Citation](#citation)

</div>

---

## Overview

> **The Problem:** Current VLMs are trained on static image-text pairs, then fine-tuned for agentic tasks. This creates a fundamental gap -- the model never learns *how to actively use vision for agency*. It can describe what it sees, but cannot decide *where to look*, *what will happen next*, or *whether its understanding is correct*.

> **Our Solution:** Insert an **Agentic Continual Pre-training** stage between VLM pre-training and SFT/RL. During this stage, the model learns three foundational agentic capabilities through 100B+ tokens of scalable synthetic data -- before any task-specific training begins.

### Why Continual Pre-training?

Existing approaches to building VLM agents rely on SFT or RL, which forces the model to simultaneously learn agentic capabilities *and* align to specific tasks. This creates optimization tension (as shown by [AgentFounder](https://arxiv.org/abs/2509.13310) for text-only LLMs). **AgentCPT** resolves this by instilling agentic visual reasoning as a foundational capability during CPT, giving downstream SFT/RL a much stronger starting point.

```
                        ┌─────────────────────────────────────────────┐
                        │           AgentCPT (this work)              │
                        │                                             │
VLM Base Model ──────►  │  Stage 1 (32K ctx)  ──►  Stage 2 (128K ctx) │  ──────►  SFT / RL
(Qwen2-VL, etc.)        │  ~200B tokens            ~100B tokens       │           (task-specific)
                        │  perception +             full PPV loop +    │
                        │  prediction               self-correction    │
                        └─────────────────────────────────────────────┘
```

---

## The PPV Loop

AgentCPT trains three mutually reinforcing capabilities in a unified cognitive loop:

```
    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │   ┌───────────┐     ┌───────────┐     ┌───────┐     ┌────────────┐  │
    │   │ PERCEIVE  │────►│  PREDICT  │────►│  ACT  │────►│   VERIFY   │  │
    │   │ (active)  │     │ (ahead)   │     │       │     │  (check)   │  │
    │   └───────────┘     └───────────┘     └───────┘     └─────┬──────┘  │
    │         ▲                                                  │         │
    │         │              ┌────────────┐                      │         │
    │         └──────────────│   UPDATE   │◄─────────────────────┘         │
    │                        │ (correct)  │                                │
    │                        └────────────┘                                │
    └──────────────────────────────────────────────────────────────────────┘
```

| Capability | What the VLM Learns | Why It Matters for Agents |
|:--|:--|:--|
| **Perceive** (Active Perception) | Goal-directed visual attention -- deciding *where* to look and *what* to extract based on the current task | Agents can't process every pixel equally. They must selectively attend to task-relevant visual elements. |
| **Predict** (Predictive Grounding) | Anticipating how visual scenes change after actions -- an implicit world model | Before clicking a button, an agent should anticipate the result. This enables look-ahead planning. |
| **Verify** (Self-Verification) | Comparing expected vs. actual outcomes and self-correcting when wrong | Robust agents detect their own errors. PPV explicitly trains this with intentionally wrong hypotheses. |

### Why These Three Are Synergistic

- Better **perception** leads to more accurate **predictions** -- you predict better when looking at the right things
- Better **predictions** enable meaningful **verification** -- you need expectations to check against
- Better **verification** improves **perception** -- learning from errors teaches you where to look next time

---

## Three Types of Synthetic CPT Data

All data is synthesized at scale (~100B+ tokens) using VLM annotators, with no human annotation required.

### 1. Active Perception Chains (APC)

Sequential, goal-directed visual examination of complex scenes:

```
Task: "Find the checkout button on this e-commerce page"

Step 1: I focus on [top navigation bar] → I see a cart icon with badge "3"
        → Items are in cart. Now I look for a checkout flow.
Step 2: I focus on [main content area] → I see product listings, no checkout here
        → Checkout is likely below or in a sidebar.
Step 3: I focus on [right sidebar] → I see a green "Proceed to Checkout" button
        → Found it. The checkout button is in the right sidebar.
```

### 2. Visual State Transition Predictions (VSTP)

Predicting how visual scenes change after actions (no image generation -- all in text):

```
Current State: Product listing page with filters on left, 12 items displayed
Action: Click "Price: Low to High"
Predicted Changes: Products reorder with cheapest first. Sort dropdown
                   updates to show "Price: Low to High" as selected.
                   Filter sidebar remains unchanged.
```

**Data sources:** Web rendering (Playwright), GUI emulators, instructional video frame pairs, synthetic HTML

### 3. Hypothesis-Verification Chains (HVC)

Full observe-hypothesize-verify loops, *including intentionally wrong hypotheses* (~30%) to teach self-correction:

```
Hypothesis: Clicking "Account" will navigate to a profile settings page.
Expected:   A page with user info, order history, and settings tabs.
Actual:     A dropdown menu with "Sign In" and "Register" options.
Assessment: INCORRECT -- this is a logged-out state, so "Account" shows
            login options, not a profile page.
Correction: Click "Sign In" first, then navigate to profile settings.
```

### Data Mixture

| Data Type | Description | Stage 1 | Stage 2 |
|:--|:--|:-:|:-:|
| **APC** | Active Perception Chains | 40% | 20% |
| **VSTP** | Visual State Transition Predictions | 40% | 20% |
| **HVC** | Hypothesis-Verification Chains | 10% | 40% |
| **General VL** | Anti-forgetting VL data | 10% | 20% |

---

## Quick Start

### Installation

```bash
pip install -e .

# Full install with synthesis + evaluation + dev tools
pip install -e ".[synthesis,eval,dev]"
```

### Data Synthesis

```bash
# Active Perception Chains
python scripts/synthesize.py --type apc \
    --source-dir data/scenes/ --output-dir data/apc/ \
    --annotator-model Qwen/Qwen2-VL-72B-Instruct --num-samples 100000

# Visual State Transitions
python scripts/synthesize.py --type vstp \
    --source-dir data/web_interactions/ --output-dir data/vstp/ \
    --num-samples 100000

# Hypothesis-Verification Chains
python scripts/synthesize.py --type hvc \
    --source-dir data/trajectories/ --output-dir data/hvc/ \
    --num-samples 50000
```

### Continual Pre-training

```bash
# Stage 1: Foundational perception + prediction (32K context, ~200B tokens)
accelerate launch scripts/train.py --config configs/stage1.yaml

# Stage 2: Full PPV loop with self-correction (128K context, ~100B tokens)
accelerate launch scripts/train.py --config configs/stage2.yaml

# Post-CPT: Task-specific supervised fine-tuning
accelerate launch scripts/train.py --config configs/sft.yaml
```

### Evaluation

```bash
# Intrinsic PPV evaluation (perception / prediction / verification quality)
python scripts/evaluate.py --model-path checkpoints/stage2/final \
    --benchmarks intrinsic --output-dir results/

# Full downstream benchmark suite
python scripts/evaluate.py --model-path checkpoints/sft/final \
    --benchmarks gui_web deep_research visual_reasoning general_vlm \
    --output-dir results/
```

---

## Evaluation

### Intrinsic PPV Metrics

| Metric | What It Measures |
|:--|:--|
| Perception Quality | Region relevance, chain completeness, spatial grounding accuracy |
| Prediction Quality | State transition accuracy (ROUGE-L), action relevance |
| Verification Quality | Error detection rate, correction validity |

### Downstream Agentic Benchmarks

| Category | Benchmarks |
|:--|:--|
| **GUI / Web Agents** | Mind2Web, AITW, ScreenSpot, VisualWebBench |
| **Deep Research** | BrowseComp, GAIA, Xbench-DeepSearch |
| **Visual Reasoning** | VSR, BLINK, SpatialEval |
| **General VLM** | VQAv2, GQA, MME, TextVQA, ChartQA |

### Key Ablations

| Experiment | Question |
|:--|:--|
| AgentCPT + SFT vs. Vanilla VLM + SFT | Does agentic CPT improve downstream agentic performance? |
| APC-only vs. VSTP-only vs. HVC-only | Which PPV capability contributes most? |
| All-three vs. pairwise combinations | Are the three capabilities synergistic? |
| Stage 1 only vs. Stage 1 + 2 | Does progressive long-context training help? |
| Data scaling (10B to 300B tokens) | How does agentic capability scale with CPT data? |

---

## Supported VLM Backbones

| Model | Sizes | Notes |
|:--|:--|:--|
| **Qwen2-VL** | 7B / 72B | Default backbone |
| **InternVL2** | 8B / 40B / 76B | Strong multilingual VLM |
| **LLaVA-OneVision** | 7B / 72B | Community standard |

---

## Project Structure

```
agentcpt/
├── configs/                  # Training configurations
│   ├── base.yaml             #   Shared hyperparameters
│   ├── stage1.yaml           #   Stage 1 CPT (32K ctx, ~200B tokens)
│   ├── stage2.yaml           #   Stage 2 CPT (128K ctx, ~100B tokens)
│   └── sft.yaml              #   Post-CPT supervised fine-tuning
├── ppv/
│   ├── data/                 # Dataset classes (APC, VSTP, HVC, mixtures)
│   ├── synthesis/            # Scalable data synthesis pipelines
│   ├── models/               # VLM wrapper (Qwen2-VL, InternVL2, LLaVA)
│   ├── training/             # CPT trainer with Accelerate + DeepSpeed
│   └── evaluation/           # Intrinsic PPV + downstream benchmarks
├── scripts/                  # CLI entry points (train, synthesize, evaluate)
└── tests/                    # Test suite
```

---

## Citation

```bibtex
@article{agentcpt,
  title={AgentCPT: Agentic Continual Pre-training for Vision-Language Foundation Models via Perceive-Predict-Verify},
  year={2025}
}
```

## License

Apache 2.0
