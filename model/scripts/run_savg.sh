#!/usr/bin/env bash
# Minimal end-to-end pipeline for the UAV-SAVG dataset.
#
# Stage 1 (temporal grounding, Qwen target extraction) -> Stage 2 (RSVG-ZeroOV
# spatial grounding) -> Stage 3 (SAM3 mask propagation).
#
# Hyperparameters are exposed as environment variables with sane defaults:
#   RSVG_DIR     path to the external RSVG-ZeroOV repo
#   NUM          limit number of clips
#   NUM_GPUS     parallel GPUs
#   GPU          physical GPU id for Stage 3 (sets CUDA_VISIBLE_DEVICES)
#   FORCE        1 = reprocess even if outputs exist
#   NO_VIS       1 = disable Stage 3 visualization videos
#   CONFIG, OUTPUT_DIR, GROUNDING_SUBDIR, STAGE3_SUBDIR
#
# Example:
#   NUM=10 NUM_GPUS=4 bash model/scripts/run_savg.sh
set -euo pipefail

# --------------------------- Hyperparameters --------------------------------- #
DATASET="savg"                                   # UAV-SAVG dataset key
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data/disk2/zyu/videoVG/model/output/savg}"

GROUNDING_SUBDIR="${GROUNDING_SUBDIR:-grounding}"
RSVG_DIR="${RSVG_DIR:-/mnt/data/disk2/zyu/videoVG/RSVG-ZeorOV}"
STAGE3_SUBDIR="${STAGE3_SUBDIR:-stage3_sam3}"

NUM="${NUM:-}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU="${GPU:-}"
FORCE="${FORCE:-0}"
NO_VIS="${NO_VIS:-0}"

# --------------------------- Path setup -------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$(dirname "${SCRIPT_DIR}")"        # model/
REPO_ROOT="$(dirname "${MODEL_DIR}")"         # repo root (parent of model/)
cd "${REPO_ROOT}"
export PYTHONPATH="${MODEL_DIR}:${PYTHONPATH:-}"   # makes `python -m stvg.cli...` importable
CONFIG="${CONFIG:-${MODEL_DIR}/config.yaml}"

NUM_ARG=()
[[ -n "${NUM}" ]] && NUM_ARG=(--num "${NUM}")
FORCE_ARG=()
[[ "${FORCE}" == "1" ]] && FORCE_ARG=(--force)
GPU_ARG=()
[[ -n "${GPU}" ]] && GPU_ARG=(--gpu "${GPU}")
NO_VIS_ARG=()
[[ "${NO_VIS}" == "1" ]] && NO_VIS_ARG=(--no-visualization)

echo "=== Stage 1: Temporal grounding (TFVTG, Qwen target extraction) ==="
python "${MODEL_DIR}/stage1_tfvtg.py" --dataset "${DATASET}" --output-dir "${OUTPUT_DIR}" "${NUM_ARG[@]}"

echo "=== Stage 2: Spatial grounding (RSVG-ZeroOV) ==="
python -m stvg.cli.stage2_grounding \
    --dataset "${DATASET}" \
    --output-dir "${OUTPUT_DIR}" \
    --config "${CONFIG}" \
    --grounding-subdir "${GROUNDING_SUBDIR}" \
    --rsvg-dir "${RSVG_DIR}" \
    "${NUM_ARG[@]}" "${FORCE_ARG[@]}"

echo "=== Stage 3: Mask propagation (SAM3) ==="
python -m stvg.cli.stage3_propagation \
    --dataset "${DATASET}" \
    --input-root "${OUTPUT_DIR}" \
    --output-root "${OUTPUT_DIR}" \
    --config "${CONFIG}" \
    --stage2-subdir "${GROUNDING_SUBDIR}" \
    --stage3-subdir "${STAGE3_SUBDIR}" \
    --num-gpus "${NUM_GPUS}" \
    "${GPU_ARG[@]}" "${NUM_ARG[@]}" "${NO_VIS_ARG[@]}"

echo "=== Done. Stage 3 masks under ${OUTPUT_DIR}/${STAGE3_SUBDIR}/ ==="
