#!/bin/bash
set -euo pipefail

# --- CONDA INITIALIZATION ---
# Find your main conda/anaconda installation path.
# Common locations are ~/anaconda3, ~/miniconda3, or specified during installation.
# Update this path if it's different for your system.
CONDA_BASE_PATH=$(conda info --base)

# Source the conda shell script to make the 'conda' command available.
source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set default variables
INPUT_FOLDER="./data/testcases/images"
JSON_FOLDER="./data/testcases/instructions.json"
OUTPUT_FOLDER="./outputs/testcases/api_llm"

API_KEY=""

API_ARGS=()
if [[ -n "$API_KEY" ]]; then
	API_ARGS=(--api_key "$API_KEY")
fi
    

# -----------------------------

echo "================================="
echo "=== STARTING GPT-API PIPELINE ==="
echo "================================="

# Step 1: Run object detection in env1
echo "STEP 1: Object Detection..."
conda activate reorm
python -m src.inference.api_llm.inference_api_llm_1 --image_folder "$INPUT_FOLDER" --json_path "$JSON_FOLDER" --output_folder "$OUTPUT_FOLDER" "${API_ARGS[@]}"
echo "STEP 1: Finished."

# Step 2: Run segmentation and initial inpainting in env2
echo "STEP 2: Inpainting..."
conda activate objectclear
python -m src.inference.api_llm.inference_api_llm_2 --image_folder "$INPUT_FOLDER" --json_path "$JSON_FOLDER" --output_folder "$OUTPUT_FOLDER" "${API_ARGS[@]}"
echo "STEP 2: Finished."

# Step 3: Run self-correction in env1
echo "STEP 3: Self-Correction..."
conda activate reorm
python -m src.inference.api_llm.inference_api_llm_3 --image_folder "$INPUT_FOLDER" --json_path "$JSON_FOLDER" --output_folder "$OUTPUT_FOLDER" "${API_ARGS[@]}"
echo "STEP 3: Finished."

echo "========================================="
echo "=== PIPELINE COMPLETED SUCCESSFULLY   ==="
echo "========================================="