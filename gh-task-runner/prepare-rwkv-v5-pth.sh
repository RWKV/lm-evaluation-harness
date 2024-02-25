#!/bin/bash

# Properly bringing up errror
set -e

# Get the current script directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

# This script converts the RWKV pth file to the huggingface format
# We start by download the pth file from the given URL

# Get the path from the argument or env variable
if [[ -z "${RWKV_PTH_URL}" ]]; then
    export RWKV_PTH_URL="$1"
fi
if [[ -z "${RWKV_PTH_URL}" ]]; then
    echo "### [ERROR]: RWKV_PTH_URL is not set, run with the URL as the first argument : ./prepare-rwkv-v5-pth.sh <URL>"
    exit 1
else 
    echo "### RWKV_PTH_URL is set to $RWKV_PTH_URL"
fi

# Goto the project directory
cd "$PROJ_DIR"
mkdir -p ./model
cd ./model

# Clone the HF conversion repo, if not already cloned
echo "### Getting RWKV World HF repo"
if [ ! -d RWKV-World-HF-Tokenizer ]; then
    git clone https://github.com/BBuf/RWKV-World-HF-Tokenizer.git RWKV-World-HF-Tokenizer
else
    cd RWKV-World-HF-Tokenizer
    git pull
    cd ..
fi

# Download the model
echo "### Downloading the model from $RWKV_PTH_URL"
# rm -rf "$PROJ_DIR/model/rwkv-v5.pth" || true
wget -nv -c -O "$PROJ_DIR/model/rwkv-v5.pth" "$RWKV_PTH_URL"
echo "### Downloaded the model"

# Get the file size
FILE_SIZE=$(du -sk "$PROJ_DIR/model/rwkv-v5.pth" | awk '{print $1}')

# Compute the model size in GB
MODEL_SIZE_MB=$(echo "scale=2; $FILE_SIZE / 1024 " | bc)

# Figure out the model param size approximation
MODEL_SIZE="0B"
REF_REPO_URL=""
if (( $(echo "$MODEL_SIZE_MB > 12000" | bc -l) )); then
    MODEL_SIZE="7B"
elif (( $(echo "$MODEL_SIZE_MB > 5000" | bc -l) )); then
    MODEL_SIZE="3B"
elif (( $(echo "$MODEL_SIZE_MB > 2000" | bc -l) )); then
    MODEL_SIZE="1B5"
# elif (( $(echo "$MODEL_SIZE_MB > 500" | bc -l) )); then
#     MODEL_SIZE="430M"
# elif (( $(echo "$MODEL_SIZE_MB > 250" | bc -l) )); then
#     MODEL_SIZE="169M"
else
    # Throw an error
    echo "### [ERROR]: Model size unsupported: $MODEL_SIZE_MB MB"
    exit 1
fi
echo "### Model size: $MODEL_SIZE_MB MB / $MODEL_SIZE"

# Load the reference HF repo
echo "### Loading the reference HF repo"
if [[ "$MODEL_SIZE" == "7B" ]]; then
    REF_REPO_URL="https://huggingface.co/RWKV/v5-Eagle-7B-HF.git"
    REF_REPO_NAME="v5-Eagle-7B-HF"
elif [[ "$MODEL_SIZE" == "3B" ]]; then
    REF_REPO_URL="https://huggingface.co/RWKV/rwkv-5-world-3b.git"
    REF_REPO_NAME="rwkv-5-world-3b"
elif [[ "$MODEL_SIZE" == "1B5" ]]; then
    REF_REPO_URL="https://huggingface.co/RWKV/rwkv-5-world-1b5.git"
    REF_REPO_NAME="rwkv-5-world-1b5"
fi

# Clone the reference repo
if [ ! -d "$PROJ_DIR/model/$REF_REPO_NAME" ]; then
    cd "$PROJ_DIR/model"
    GIT_LFS_SKIP_SMUDGE=1 git clone "$REF_REPO_URL" "$REF_REPO_NAME"

    # Remove the bin file
    cd "$PROJ_DIR/model/$REF_REPO_NAME"
    rm -rf ./*.bin || true
# else
#     cd "$PROJ_DIR/model/$REF_REPO_NAME"
#     git pull
fi
cd "$PROJ_DIR/model"

# Prepare the output folder
rm -rf ./hf-format-output || true
mkdir -p ./hf-format-output/

# Convert the model
cd RWKV-World-HF-Tokenizer/scripts

# Perform the conversion
echo "### Converting the model"
python3 convert_rwkv5_checkpoint_to_hf.py \
    --local_model_file "$PROJ_DIR/model/rwkv-v5.pth" \
    --output_dir ../../hf-format-output/ \
    --tokenizer_file ../rwkv_world_tokenizer \
    --size "$MODEL_SIZE" \
    --is_world_tokenizer True

# Converted the model
echo "### Converted the model, at ./model/hf-format-output/"

# Copy the converted model
cd "$PROJ_DIR/model"
rm -rf "./TEST_MODEL" || true

# Copy the test ref
cp -r "$PROJ_DIR/model/$REF_REPO_NAME" "$PROJ_DIR/model/TEST_MODEL"
mv ./hf-format-output/pytorch_model* "$PROJ_DIR/model/TEST_MODEL/"

# The final model
echo "### Copied the model to $PROJ_DIR/model/TEST_MODEL/"
