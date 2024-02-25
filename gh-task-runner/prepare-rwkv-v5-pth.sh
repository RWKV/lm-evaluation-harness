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
    echo "### RWKV_PTH_URL : $RWKV_PTH_URL"
fi

# Goto the project directory
cd "$PROJ_DIR"
mkdir -p ./model
cd ./model

# Clone the HF conversion repo, if not already cloned
echo "### Getting RWKV World HF repo"
# rm -rf RWKV-World-HF-Tokenizer || true
if [ ! -d RWKV-World-HF-Tokenizer ]; then
    git clone https://github.com/PicoCreator/RWKV-World-HF-Tokenizer.git RWKV-World-HF-Tokenizer
else
    cd RWKV-World-HF-Tokenizer
    git pull
    cd ..
fi

# Download the model
echo "### Downloading the model from $RWKV_PTH_URL"

# Optimizing for huggingface, detect if it starts with "https://huggingface.co/"
# ends with a .pth / .pth?download=true and contains resolve/blob (in the middle)
#
# Eg: https://huggingface.co/BlinkDL/rwkv-5-world/resolve/main/RWKV-5-World-1B5-v2-20231025-ctx4096.pth?download=true
# Eg: https://huggingface.co/BlinkDL/rwkv-5-world/blob/commithash/RWKV-5-World-1B5-v2-20231025-ctx4096.pth
#
# We use this to reduce the amount of repeated download via the local HF cache
if [[ "$RWKV_PTH_URL" == "https://huggingface.co/"*".pth" ]] || [[ "$RWKV_PTH_URL" == "https://huggingface.co/"*".pth?download=true" ]]; then
   
    echo "### Detected huggingface URL"
    
    # Get the repo path, commit
    # and the file path
    if [[ "$RWKV_PTH_URL" == "https://huggingface.co/"*"/blob/"* ]]; then
        REPO_PATH=$(echo "$RWKV_PTH_URL"   | awk -F 'https://huggingface.co/' '{print $2}' | awk -F '/blob/' '{print $1}')
        COMMIT_HASH=$(echo "$RWKV_PTH_URL" | awk -F '/blob/' '{print $2}'                  | awk -F '/' '{print $1}')
        FILE_PATH=$(echo "$RWKV_PTH_URL"   | awk -F "/blob/$COMMIT_HASH/" '{print $2}'     | awk -F '.pth' '{print $1}').pth
    else
        REPO_PATH=$(echo "$RWKV_PTH_URL"   | awk -F 'https://huggingface.co/' '{print $2}' | awk -F '/resolve/' '{print $1}')
        COMMIT_HASH=$(echo "$RWKV_PTH_URL" | awk -F '/resolve/' '{print $2}'               | awk -F '/' '{print $1}')
        FILE_PATH=$(echo "$RWKV_PTH_URL"   | awk -F "/resolve/$COMMIT_HASH/" '{print $2}'  | awk -F '.pth' '{print $1}').pth
    fi

    # Filename from the file path, note it maybe nested in multiple directories
    FILE_NAME=$(basename $FILE_PATH)
    
    # Repo and file
    echo "### Repo: $REPO_PATH"
    echo "### Commit: $COMMIT_HASH"
    echo "### File path: $FILE_PATH"
    echo "### File name: $FILE_NAME"

    # Download via huggerface-cli
    # ---
    # rm -rf "$PROJ_DIR/model/rwkv-v5.pth" || true

    # Preload the file into the HF cache
    huggingface-cli download "$REPO_PATH" "$FILE_PATH"

    # Copy the file from the cache into the model folder
    rm "$FILE_NAME" || true
    huggingface-cli download --local-dir "./" --local-dir-use-symlinks False "$REPO_PATH" "$FILE_PATH"

    # Move it as rwkv-v5.pth
    mv "$FILE_NAME" "rwkv-v5.pth"

else 
    # Download the file
    # ---

    rm -rf "$PROJ_DIR/model/rwkv-v5.pth" || true
    wget -nv -c -O "$PROJ_DIR/model/rwkv-v5.pth" "$RWKV_PTH_URL"

fi

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
    echo "###"
    echo "### [ERROR]: Model size unsupported: $MODEL_SIZE_MB MB"
    echo "###"
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

## Reset to the project directory
cd "$PROJ_DIR"