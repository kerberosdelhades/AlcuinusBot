#!/usr/bin/env bash
# bootstrap.sh — set up AlcuinusBot from a fresh clone.
#
# What this does:
#   1. Creates a Python venv and installs dependencies
#   2. Copies config/.env.example to config/.env (if not present)
#   3. Prints instructions for the next manual steps
#
# What this does NOT do:
#   - Fetch messages from Telegram (needs config/.env filled in first)
#   - Run the pipeline (needs data + MISTRAL_API_KEY)
#   - Create the SQLite DB (needs messages first, then `python -m alcuinus.db migrate`)
#
# Usage:
#   git clone <repo> && cd AlcuinusBot && bash bootstrap.sh

set -euo pipefail

echo "=== AlcuinusBot bootstrap ==="
echo ""

# 1. Python environment
echo "--- 1. Python environment ---"
if command -v uv &>/dev/null; then
    echo "  uv found — creating venv..."
    uv venv .venv
    uv pip install -e .
elif command -v python3 &>/dev/null; then
    echo "  uv not found — using python3..."
    python3 -m venv .venv
    .venv/bin/pip install -e .
else
    echo "  ERROR: neither uv nor python3 found."
    exit 1
fi
echo "  Dependencies installed."
echo ""

# 2. Config
echo "--- 2. Config ---"
if [ ! -f config/.env ]; then
    cp config/.env.example config/.env
    echo "  Created config/.env from template."
    echo "  Fill in your Telegram credentials (api_id, api_hash, source_channel)."
else
    echo "  config/.env already exists — skipping."
fi
echo ""

# 3. Next steps
echo "--- 3. Next steps ---"
echo "  1. Fill in config/.env with your Telegram API credentials."
echo "     Get them at: https://my.telegram.org/apps"
echo ""
echo "  2. Set MISTRAL_API_KEY in your environment:"
echo "     export MISTRAL_API_KEY=your-key-here"
echo ""
echo "  3. Fetch messages from Telegram:"
echo "     .venv/bin/python -m alcuinus.extraction_v2 --full"
echo ""
echo "  4. Run the full pipeline:"
echo "     .venv/bin/python -m alcuinus.anchor_detection"
echo "     .venv/bin/python -m alcuinus.association"
echo "     .venv/bin/python -m alcuinus.metadata"
echo "     .venv/bin/python -m alcuinus.chunking"
echo "     .venv/bin/python -m alcuinus.embedding"
echo "     .venv/bin/python -m alcuinus.clustering"
echo "     .venv/bin/python -m alcuinus.decay"
echo "     .venv/bin/python -m alcuinus.output"
echo "     .venv/bin/python -m alcuinus.syllabus"
echo "     .venv/bin/python -m alcuinus.review"
echo ""
echo "=== Bootstrap complete ==="
