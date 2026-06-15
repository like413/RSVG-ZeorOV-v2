# RSVG-ZeroOV-v2 — Zero-Shot Spatio-Temporal Video Grounding

A **training-free**, **zero-shot** pipeline for Spatio-Temporal Video Grounding (STVG)
on **UAV-SAVG**. Given a video and a natural-language query
(e.g. *"the black all-terrain vehicle at the front of the curve"*), it localizes
the target **in time** (key frame) and **in space** (a per-frame segmentation
mask tube).

The refactored, high-cohesion / low-coupling implementation lives in
[`model/stvg/`](model/stvg). Full documentation — architecture diagram, weights
layout, and stage I/O reference — is in **[`model/README.md`](model/README.md)**.

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
