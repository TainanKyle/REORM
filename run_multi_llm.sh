#!/bin/bash
set -euo pipefail

# --- CONDA INITIALIZATION ---
# Find your main conda/anaconda installation path.
# Common locations are ~/anaconda3, ~/miniconda3, or specified during installation.
# Update this path if it's different for your system.
CONDA_BASE_PATH=$(conda info --base)

# Set default variables
INPUT_FOLDER="./data/testcases/images"
JSON_FOLDER="./data/testcases/instructions.json"
OUTPUT_FOLDER="./outputs/testcases/multi_llm"

# Source the conda shell script to make the 'conda' command available.
source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
# -----------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==================================="
echo "=== STARTING MULTI-LLM PIPELINE ==="
echo "==================================="

# Step 1: Run object detection in env1
echo "STEP 1: Object Detection..."
conda activate llava_research
python -m src.inference.multi_llm.inference_multi_llm_1 --image_folder "$INPUT_FOLDER" --json_path "$JSON_FOLDER" --output_folder "$OUTPUT_FOLDER"
echo "STEP 1: Finished."

# Step 2: Run segmentation in env2
echo "STEP 2: Segmentation..."
conda activate myllm
python -m src.inference.multi_llm.inference_multi_llm_2 --image_folder "$INPUT_FOLDER" --json_path "$JSON_FOLDER" --output_folder "$OUTPUT_FOLDER"
echo "STEP 2: Finished."

# Step 2: Run inpainting in env3
echo "STEP 3: Inpainting..."
conda activate objectclear
python -m src.inference.multi_llm.inference_multi_llm_3 --image_folder "$INPUT_FOLDER" --json_path "$JSON_FOLDER" --output_folder "$OUTPUT_FOLDER"
echo "STEP 3: Finished."

echo "========================================="
echo "=== PIPELINE COMPLETED SUCCESSFULLY   ==="
echo "========================================="