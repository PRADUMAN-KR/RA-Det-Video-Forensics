#!/bin/bash
################################################################################
# Container Environment Setup — uv
#
# Run this ONCE inside your container after rsyncing the project.
# This installs uv and creates the virtual environment.
#
# Usage:
#   bash scripts/setup_env.sh
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(dirname "$SCRIPT_DIR")"

cd "$WORK_DIR"

echo "========================================================================"
echo "RA-Det Environment Setup (uv)"
echo "========================================================================"

# ── Step 1: Install uv if not present ────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    echo "[1/3] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "      uv installed: $(uv --version)"
else
    echo "[1/3] uv already installed: $(uv --version)"
fi

# ── Step 2: Create virtual environment ───────────────────────────────────────
echo "[2/3] Creating .venv with uv..."
uv venv .venv --python 3.10

# ── Step 3: Install dependencies from pyproject.toml ─────────────────────────
echo "[3/3] Installing dependencies from pyproject.toml..."
uv pip install -e ".[dev]" || uv pip install -r requirements.txt

echo ""
echo "========================================================================"
echo " ✅  Environment ready!"
echo ""
echo "  Activate with:  source .venv/bin/activate"
echo "  Or run with:    uv run python train.py ..."
echo "========================================================================"
