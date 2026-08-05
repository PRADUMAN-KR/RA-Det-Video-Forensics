#!/bin/bash
################################################################################
# Re-validate saved dinov2_temporal_v1 checkpoints with the FIXED
# evaluate_decoder() scoring (two-pass decoder refinement, learned_weight
# fusion, std-based temporal_diff normalization, plateau-centered
# optimal_threshold selection — matching inference_api.py and train_epoch()
# exactly).
#
# Each run patches `optimal_threshold` directly into the checkpoint file
# in place. Training weights (decoder/classifier state dicts) are NOT
# modified — only the calibrated threshold key is added/updated.
#
# Usage:
#   bash scripts/revalidate_checkpoints.sh --epochs 12,13,14,18,20 [--checkpoint-dir PATH]
#
#   --epochs accepts a COMMA-separated list (preferred) or a space-separated
#   list. Comma-separated is strongly recommended in CI systems like Jenkins,
#   where an extra shell/eval layer can mangle quoted space-separated values
#   (observed: --epochs "18 20" arrived as the two tokens "18 and 20"",
#   silently dropping everything after the first word). Commas need no
#   quoting to survive that, so prefer: --epochs 18,20
#
#   Runner selection (pick ONE):
#     --use-uv                       Run via "uv run python" (default)
#     --python-bin /path/to/python   Run via a plain python executable
#                                     instead of uv — use this in Jenkins/CI
#                                     where uv isn't installed/on PATH.
#
# Notes:
#   - Run this FROM THE REPO ROOT on the training machine (has GPU + data).
#   - Written as single-line commands (no backslash continuations) to avoid
#     the "command not found" issue caused by trailing whitespace/CRLF after
#     a line-continuation backslash when pasting multi-line commands.
################################################################################
set -e

################################################################################
# Configuration (override via flags below)
################################################################################

EPOCHS="12,13,14,15,16,17,18,19,20"
CHECKPOINT_DIR="./checkpoints/dinov2_temporal_v1"
# NOTE: these must point at YOUR actual data directories on the machine this
# runs on. They differ from the original training command's paths
# (/home/praduman/...) if you're running as a different user (e.g. ubuntu).
# Override with --train-data-path / --val-data-path if these defaults are wrong.
VIDEO_TRAIN_DATA_PATH="${HOME}/RA-Det/data/video_dataset/train"
VIDEO_VAL_DATA_PATH="${HOME}/RA-Det/data/video_dataset/test"
CONFIG_NAME="dinov2_temporal_v1"
UV_BIN="/home/ubuntu/.local/bin/uv"
# If set (via --python-bin), plain python is used instead of uv — needed in
# environments like Jenkins where "uv" isn't installed/on PATH.
PYTHON_BIN=""

################################################################################
# Parse Arguments
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --train-data-path)
            VIDEO_TRAIN_DATA_PATH="$2"
            shift 2
            ;;
        --val-data-path)
            VIDEO_VAL_DATA_PATH="$2"
            shift 2
            ;;
        --python-bin)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --use-uv)
            PYTHON_BIN=""
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Defensively strip stray double-quote/single-quote characters that some CI
# wrappers (observed with Jenkins) leak into argument values instead of using
# them purely for shell tokenization. Also normalize commas to spaces so
# --epochs accepts either "18,20" or "18 20" once it actually arrives intact.
strip_quotes() {
    printf '%s' "$1" | tr -d '"'"'"
}
EPOCHS="$(strip_quotes "$EPOCHS")"
EPOCHS="${EPOCHS//,/ }"
CHECKPOINT_DIR="$(strip_quotes "$CHECKPOINT_DIR")"
VIDEO_TRAIN_DATA_PATH="$(strip_quotes "$VIDEO_TRAIN_DATA_PATH")"
VIDEO_VAL_DATA_PATH="$(strip_quotes "$VIDEO_VAL_DATA_PATH")"
PYTHON_BIN="$(strip_quotes "$PYTHON_BIN")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(dirname "$SCRIPT_DIR")"
cd "$WORK_DIR"

# Fail loudly instead of silently walking a non-existent/empty directory
# (os.walk() on a bad path returns 0 samples with no error — this is what
# caused "Loaded 0 video samples: 0 fake, 0 real." previously).
if [ ! -d "$VIDEO_TRAIN_DATA_PATH" ]; then
    echo "ERROR: --train-data-path does not exist: $VIDEO_TRAIN_DATA_PATH" >&2
    echo "Pass the correct path with --train-data-path /actual/path" >&2
    exit 1
fi
if [ ! -d "$VIDEO_VAL_DATA_PATH" ]; then
    echo "ERROR: --val-data-path does not exist: $VIDEO_VAL_DATA_PATH" >&2
    echo "Pass the correct path with --val-data-path /actual/path" >&2
    exit 1
fi
if [ -z "$(find "$VIDEO_VAL_DATA_PATH" -mindepth 1 -print -quit 2>/dev/null)" ]; then
    echo "ERROR: --val-data-path exists but appears empty: $VIDEO_VAL_DATA_PATH" >&2
    exit 1
fi

# Also validate that at least one requested checkpoint epoch is a plain
# integer — catches lingering quote-mangling early with a clear message
# instead of failing deep inside train.py's argument parser.
for EPOCH in $EPOCHS; do
    if ! [[ "$EPOCH" =~ ^[0-9]+$ ]]; then
        echo "ERROR: epoch value '$EPOCH' is not a plain integer (parsed EPOCHS='$EPOCHS')." >&2
        echo "This usually means quoting was mangled upstream (e.g. by a CI wrapper)." >&2
        echo "Try passing epochs comma-separated with no spaces, e.g.: --epochs 18,20" >&2
        exit 1
    fi
done

if [ -n "$PYTHON_BIN" ]; then
    if [ ! -x "$PYTHON_BIN" ]; then
        echo "ERROR: --python-bin is not an executable file: $PYTHON_BIN" >&2
        exit 1
    fi
else
    if [ ! -x "$UV_BIN" ]; then
        echo "ERROR: uv binary not found/executable at: $UV_BIN" >&2
        echo "Pass --python-bin /path/to/venv/bin/python to use plain python instead." >&2
        exit 1
    fi
fi

LOG_DIR="${CHECKPOINT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "========================================================================"
echo "RE-VALIDATING CHECKPOINTS (fixed evaluate_decoder scoring)"
echo "========================================================================"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Epochs:         $EPOCHS"
echo "Train data:     $VIDEO_TRAIN_DATA_PATH"
echo "Val data:       $VIDEO_VAL_DATA_PATH"
if [ -n "$PYTHON_BIN" ]; then
    echo "Runner:         $PYTHON_BIN (plain python)"
else
    echo "Runner:         $UV_BIN run python"
fi
echo "========================================================================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for EPOCH in $EPOCHS; do
    CKPT_PATH="${CHECKPOINT_DIR}/checkpoint_epoch_${EPOCH}.pt"
    LOG_PATH="${LOG_DIR}/revalidate_epoch_${EPOCH}.log"

    if [ ! -f "$CKPT_PATH" ]; then
        echo "=== Skipping epoch $EPOCH: checkpoint not found at $CKPT_PATH ==="
        continue
    fi

    echo "=== Validating epoch $EPOCH ==="
    if [ -n "$PYTHON_BIN" ]; then
        "$PYTHON_BIN" train.py --config "$CONFIG_NAME" --video-mode --video-train-data-path "$VIDEO_TRAIN_DATA_PATH" --video-val-data-path "$VIDEO_VAL_DATA_PATH" --num-frames 8 --batch-size 2 --num-workers 4 --normalize-loss --four-branch-ensemble --fusion-method learned_weight --checkpoint-dir "$CHECKPOINT_DIR" --validate-checkpoint "$CKPT_PATH" 2>&1 | tee "$LOG_PATH"
    else
        "$UV_BIN" run python train.py --config "$CONFIG_NAME" --video-mode --video-train-data-path "$VIDEO_TRAIN_DATA_PATH" --video-val-data-path "$VIDEO_VAL_DATA_PATH" --num-frames 8 --batch-size 2 --num-workers 4 --normalize-loss --four-branch-ensemble --fusion-method learned_weight --checkpoint-dir "$CHECKPOINT_DIR" --validate-checkpoint "$CKPT_PATH" 2>&1 | tee "$LOG_PATH"
    fi
done

echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"
for EPOCH in $EPOCHS; do
    LOG_PATH="${LOG_DIR}/revalidate_epoch_${EPOCH}.log"
    if [ -f "$LOG_PATH" ]; then
        echo "--- Epoch $EPOCH ---"
        grep -A8 "Validation metrics:" "$LOG_PATH" || echo "  (no metrics found in log)"
        grep "Patching checkpoint with calibrated threshold" "$LOG_PATH" || echo "  (no threshold patch found in log)"
    fi
done

echo "Done! Compare the printed val/auc, val/f1_score, val/fpr, val/fnr, and"
echo "the patched calibrated threshold across epochs above, then pick the best."
