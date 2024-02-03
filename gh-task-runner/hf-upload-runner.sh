#!/bin/bash

# -----
# Required ARGS check
# -----

# Check if HUGGING_FACE_HUB_TOKEN & WANDB_API_KEY is set
if [[ -z "${HUGGING_FACE_HUB_TOKEN}" ]]; then
    echo "[ERROR]: HUGGING_FACE_HUB_TOKEN is not set"
    exit 1
fi
# if [[ -z "${WANDB_API_KEY}" ]]; then
#     echo "[ERROR]: WANDB_API_KEY is not set"
#     exit 1
# fi

# The HF repo directory to use
if [[ -z "${HF_REPO_SYNC}" ]]; then
    HF_REPO_SYNC="rwkv-x-dev/lm-eval-output"
fi

# Get the current script directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

# Output directory
OUTPUT_DIR="$PROJ_DIR/output"

# -----

# Run the python uploader, passing all the args
cd "$PROJ_DIR"
python3 ./gh-task-runner/hf-upload.py "$1" "$2" "$OUTPUT_DIR"
