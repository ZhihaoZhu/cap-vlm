<div align="center">

<h1>AgentForge-VLM</h1>

<h3>Forging Agentic Vision-Language Models<br/>through Perceive-Predict-Verify Continual Pre-Training</h3>

<p>
<em>The first framework that instills <strong>agentic visual reasoning</strong> into VLMs<br/>
at the <strong>continual pre-training</strong> stage — before any task-specific fine-tuning.</em>
</p>

<p>
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-green.svg" alt="Python 3.10+"></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1+-red.svg" alt="PyTorch"></a>
<a href="https://huggingface.co/docs/accelerate"><img src="https://img.shields.io/badge/Accelerate-DeepSpeed-orange.svg" alt="Accelerate"></a>
</p>

<p>
<a href="#-overview">Overview</a> &bull;
<a href="#-the-ppv-loop">Method</a> &bull;
<a href="#-three-types-of-agentic-cpt-data">Data</a> &bull;
<a href="#-quick-start">Quick Start</a> &bull;
<a href="#-evaluation">Benchmarks</a> &bull;
<a href="#-citation">Citation</a>
</p>

---

<table>
<tr>
<td width="33%" align="center"><h4>VLM</h4><em>Built for Vision-Language Models.<br/>Supports Qwen2-VL, InternVL2, LLaVA.</em></td>
<td width="33%" align="center"><h4>Agentic</h4><em>Teaches active perception,<br/>predictive grounding, and self-correction.</em></td>
<td width="33%" align="center"><h4>Continual Pre-Training</h4><em>300B tokens of synthetic agentic data<br/>between pre-training and SFT/RL.</em></td>
</tr>
</table>

</div>

---

## Overview

Current Vision-Language Models are trained on static image-text pairs, then fine-tuned for agentic tasks. This creates a fundamental gap: **the VLM never learns how to actively use vision for agency.** It can describe what it sees, but cannot decide *where to look*, *what will happen next*, or *whether its understanding is correct*.

**AgentForge-VLM** closes this gap by inserting an **Agentic Continual Pre-Training** stage into the VLM training pipeline. During this stage, the model learns three foundational agentic capabilities through 100B+ tokens of scalable synthetic data — **before any task-specific training begins.**

### Why Continual Pre-Training for VLM Agents?

Existing approaches build VLM agents through SFT or RL alone, which forces the model to simultaneously learn agentic capabilities *and* align to specific tasks — creating optimization tension ([AgentFounder, 2025](https://arxiv.org/abs/2509.13310) demonstrated this for text-only LLMs; [Magma, 2025](https://arxiv.org/abs/2502.13130) showed the perception-action gap for VLMs).

**AgentForge-VLM** resolves this by forging agentic visual reasoning as a **foundational VLM capability** during continual pre-training, giving downstream SFT/RL a dramatically stronger starting point.

### The AgentForge Training Pipeline

```
                         ┌────────────────────────────────────────────────────┐
                         │        AgentForge-VLM Continual Pre-Training       │
                         │                                                    │
  VLM Base Model  ────►  │   Stage 1 (32K ctx)   ───►   Stage 2 (128K ctx)   │  ────►  Agentic
  (Qwen2-VL, etc.)       │   ~200B tokens                ~100B tokens         │         SFT / RL
                         │   Active Perception +          Full PPV loop +     │
                         │   State Prediction             Self-Correction     │
                         └────────────────────────────────────────────────────┘

                         ▲ No agentic ability             Foundational agentic VLM ▲
                         │ (passive viewer)               (active visual agent)     │
```

---

## The PPV Loop

At the core of AgentForge-VLM is the **Perceive-Predict-Verify** loop — a unified cognitive architecture that trains three mutually reinforcing agentic capabilities:

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
| **Perceive** | Goal-directed visual attention — deciding *where* to look and *what* to extract based on the current task | Agents can't process every pixel equally. They must selectively attend to task-relevant visual elements. |
| **Predict** | Anticipating how visual scenes change after actions — an implicit world model in natural language | Before clicking a button, an agent should anticipate the result. This enables look-ahead planning. |
| **Verify** | Comparing expected vs. actual outcomes and self-correcting when predictions are wrong | Robust agents detect their own errors. AgentForge-VLM trains this with intentionally wrong hypotheses. |

### Why These Three Are Synergistic

```
  Perception ──improves──► Prediction      You predict better when looking at the right things.
  Prediction ──enables──► Verification     You need expectations to check against.
  Verification ──refines──► Perception     Learning from errors teaches you where to look next.
```

---

## Three Types of Agentic CPT Data

All data is synthesized at scale (**100B+ tokens**) using VLM annotators — **no human annotation required.**

<table>
<tr>
<td width="33%">

### Active Perception Chains

**Sequential, goal-directed visual examination.**

```
Task: "Find the checkout button"

Step 1: Focus on [top nav bar]
  → Cart icon with badge "3"
  → Items in cart, look for checkout

Step 2: Focus on [main content]
  → Product listings only
  → Checkout likely in sidebar

Step 3: Focus on [right sidebar]
  → Green "Checkout" button found
  → Task complete
```

</td>
<td width="33%">

### State Transition Predictions

**Predicting visual consequences of actions.**

```
State:   Product page, 12 items
Action:  Click "Price: Low to High"

Predicted Changes:
  ✓ Products reorder by price
  ✓ Sort dropdown updates
  ✓ Filter sidebar unchanged
  ✓ Item count stays at 12
```

*Sources: Playwright, GUI emulators,
instructional video frames,
synthetic HTML rendering*

</td>
<td width="34%">

### Hypothesis-Verification Chains

**Self-correction from wrong predictions.**

```
Hypothesis: "Account" → profile
Expected:   User info + settings
Actual:     Login dropdown appeared

Assessment: INCORRECT
  → Logged-out state shows
    login, not profile

Correction:
  → Click "Sign In" first,
    then navigate to profile
```

*~30% intentionally wrong hypotheses
to explicitly train self-correction*

</td>
</tr>
</table>

### Agentic CPT Data Mixture

| Data Type | Description | Stage 1 (32K) | Stage 2 (128K) |
|:--|:--|:-:|:-:|
| **APC** — Active Perception Chains | Goal-directed sequential visual examination | 40% | 20% |
| **VSTP** — Visual State Transition Predictions | Predicting visual consequences of actions | 40% | 20% |
| **HVC** — Hypothesis-Verification Chains | Self-correction from correct and incorrect predictions | 10% | 40% |
| **General VL** — Visual-language data | Prevents catastrophic forgetting of VLM capabilities | 10% | 20% |

---

## Quick Start

### Installation

```bash
pip install -e .

# Full install with synthesis + evaluation + dev tools
pip install -e ".[synthesis,eval,dev]"
```

### Synthesize Agentic CPT Data

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

### Agentic Continual Pre-Training

```bash
# Stage 1: Foundational perception + prediction (32K context, ~200B tokens)
accelerate launch scripts/train.py --config configs/stage1.yaml

# Stage 2: Full PPV loop with self-correction (128K context, ~100B tokens)
accelerate launch scripts/train.py --config configs/stage2.yaml
```

### Post-CPT Fine-Tuning

```bash
# Task-specific supervised fine-tuning on agentic trajectories
accelerate launch scripts/train.py --config configs/sft.yaml
```

### Evaluate

```bash
# Intrinsic PPV evaluation (perception / prediction / verification quality)
python scripts/evaluate.py --model-path checkpoints/stage2/final \
    --benchmarks intrinsic --output-dir results/

# Full downstream agentic benchmark suite
python scripts/evaluate.py --model-path checkpoints/sft/final \
    --benchmarks gui_web deep_research visual_reasoning general_vlm \
    --output-dir results/
```

---

## Evaluation

### Intrinsic Metrics — Does the VLM Learn the PPV Loop?

| Metric | What It Measures |
|:--|:--|
| **Perception Quality** | Region relevance, chain completeness, spatial grounding accuracy |
| **Prediction Quality** | State transition accuracy (ROUGE-L), action-change relevance |
| **Verification Quality** | Error detection rate, correction validity |

### Downstream Agentic Benchmarks — Does It Transfer?

| Category | Benchmarks |
|:--|:--|
| **GUI / Web Agents** | Mind2Web, AITW, ScreenSpot, VisualWebBench |
| **Deep Research** | BrowseComp, GAIA, Xbench-DeepSearch |
| **Visual Reasoning** | VSR, BLINK, SpatialEval |
| **General VLM** *(no degradation)* | VQAv2, GQA, MME, TextVQA, ChartQA |

### Ablation Studies

| Experiment | Research Question |
|:--|:--|
| AgentForge + SFT vs. Vanilla VLM + SFT | Does agentic continual pre-training improve agentic performance? |
| APC-only vs. VSTP-only vs. HVC-only | Which PPV capability contributes most? |
| All-three vs. pairwise combinations | Are the three capabilities synergistic? |
| Stage 1 only vs. Stage 1 + 2 | Does progressive long-context training help? |
| Data scaling (10B → 300B tokens) | How does VLM agentic capability scale with CPT data? |

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
AgentForge-VLM/
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
│   └── evaluation/           # Intrinsic PPV metrics + downstream benchmarks
├── scripts/                  # CLI entry points (train, synthesize, evaluate)
└── tests/                    # Test suite
```

---

## Citation

```bibtex
@article{agentforge-vlm,
  title={AgentForge-VLM: Forging Agentic Vision-Language Models through Perceive-Predict-Verify Continual Pre-Training},
  year={2025}
}
```

## License

Apache 2.0
