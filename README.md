# Training-Free Open-Vocabulary Visual Grounding for Remote Sensing Images and Videos

This repository is the official implementation:
> [Training-Free Open-Vocabulary Visual Grounding for Remote Sensing Images and Videos](https://arxiv.org/abs/2606.16124)  
> Ke Li, Di Wang, Yongshan Zhu, Ting Wang, Weiping Ni, Tao Lei, Quan Wang, Xinbo Gao

## Abstract
Remote sensing visual grounding (RSVG) aims to localize a referred target in a remote sensing image or video according to a natural language expression.
Existing RSVG methods usually rely on task-specific manual annotations, which are costly to collect and inevitably limited in covering the diversity of real-world geospatial scenarios. 
As a result, they often struggle to generalize to open-vocabulary queries involving novel objects, fine-grained attributes, complex spatial relationships, and functional semantics.
In this paper, we propose RSVG-ZeroOV, a training-free framework that leverages frozen generic foundation models for zero-shot open-vocabulary RSVG.
RSVG-ZeroOV follows an *Overview-Focus-Evolve* paradigm, which exploits the distinct yet complementary attention patterns of vision-language models (VLMs) and diffusion models (DMs) to progressively generate precise grounding results.
Specifically, 
*(i) Overview* utilizes a VLM to extract cross-attention maps that capture semantic correlations between the referring expression and visual regions; 
*(ii) Focus* leverages the fine-grained modeling priors of a DM to compensate for object structure and shape information often overlooked by VLM attention; 
and *(iii) Evolve* introduces a simple yet effective attention evolution module to suppress irrelevant activations, yielding purified object masks.
To handle video inputs, we further present Video RSVG-ZeroOV, which extends image-level grounding to spatio-temporal grounding through a query-relevant key-frame selector and a temporal propagator, enabling efficient and temporally coherent video grounding without video annotations or fine-tuning.
Extensive experiments on six image and video grounding benchmarks show that RSVG-ZeroOV consistently outperforms existing zero-shot baselines and achieves competitive or superior performance compared with weakly- and fully-supervised methods.

<div align="center">
  <img src="https://github.com/like413/RSVG-ZeorOV-v2/blob/main/framework_video.png" width="100%" height="100%"/>
  The framework of the proposed Video RSVG-ZeroOV.
</div><br/>

## Pipeline at a glance

| Stage | Role |
|-------|------|
| **Stage 1** | Temporal grounding (key-frame) + target extraction |
| **Stage 2** | Spatial grounding (key-frame mask/bbox) |
| **Stage 3** | Mask propagation (per-frame tube) |

Stage 1 reuses the **same Qwen model as Stage 2** for target-object extraction.

## Quick start

```bash
pip install -r requirements.txt

# Run the contract tests (no GPU / no model weights needed)
cd model && python -m pytest tests/ -q

# Minimal UAV-SAVG run: 10 clips
NUM=10 bash model/scripts/run_savg.sh
```

See [`model/README.md`](model/README.md) for the full guide, including the model
weights download/layout and the per-stage data schemas.

## Repository layout

```text
.
├── model/
│   ├── stvg/                 # refactored package (schemas, grounding, propagation, pipeline, cli)
│   ├── scripts/              # run_savg.sh
│   ├── tests/                # pytest contract tests
│   ├── stage*.py             # legacy research entrypoints (reused by stvg)
│   ├── config.yaml
│   ├── output/               # pipeline outputs (gitignored, created at runtime)
│   └── README.md             # full documentation
├── modules/                  # shared config/model utilities used by the legacy scripts
├── data/                     # UAV-SAVG dataset (gitignored, prepare separately)
├── weights/                  # model checkpoints (gitignored, see model/README.md)
├── requirements.txt
└── .gitignore
```

> `data/`, `weights/`, and `model/output/` are **not** committed (they are large
> and listed in `.gitignore`). See the **Dataset Preparation** and **Model Weights**
> sections in [`model/README.md`](model/README.md) for the exact expected layout.
