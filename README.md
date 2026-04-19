# Interaction-Consistent Object Removal via MLLM-Based Reasoning

Official implementation of the paper **"Interaction-Consistent Object Removal via MLLM-Based Reasoning"**.

[![arXiv](https://img.shields.io/badge/arXiv-2602.01298-b31b1b.svg)](https://arxiv.org/abs/2602.01298)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

## 📌 Overview

Traditional object removal is often limited to erasures within the boundaries of a provided mask, focusing solely on the target object. However, this often leaves behind interaction evidence that are semantically tied to the target—resulting in edited outputs that are logically inconsistent. We formalize this challenge as **Interaction-Consistent Object Removal (ICOR)**. Unlike standard methods, ICOR demands a holistic understanding of the entire image and the interaction relationships among objects. When an object is removed, the system must infer the plausible appearance of the scene in its absence and identify all secondary elements that would become illogical or inconsistent.

---

## 🚀 Method: REORM

We propose **Reasoning-Enhanced Object Removal with MLLM (REORM)**, a modular framework that leverages the broad commonsense reasoning ability of Multimodal Large Language Models (MLLMs) to infer which additional elements in a scene would become inconsistent if the target were absent.

### Key Components:
- MLLM-driven Analysis: Employs a chain-of-thought prompting strategy to interpret instructions and identify both target objects and associated elements.
- Self-Correction Mechanism: An MLLM-controlled stage that simulates the expected result and resolves remaining inconsistencies or artifacts.
- Local Deployment Variant: A strategy utilizing MLLM-LLM collaboration and prompt chaining to enable robust reasoning on resource-constrained devices.

---

## 🛠️ Installation

### 1. Prerequisites

- Linux with NVIDIA GPU (24GB+) + CUDA
- Python 3.10

### 2. Clone dependencies

```bash
mkdir -p component

# Grounded-SAM
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git component/Grounded-Segment-Anything
cp ./src/tools/run_grounded_sam.py ./component/Grounded-Segment-Anything

# ObjectClear
git clone https://github.com/zjx0101/ObjectClear.git component/ObjectClear
```

### 3. Create `objectclear` environment

```bash
conda create -n objectclear python=3.10 -y
conda activate objectclear

# Install ObjectClear dependencies
pip install -r component/ObjectClear/requirements.txt
```

### 4. Create `reorm` environment

```bash
conda create -n reorm python=3.10 -y
conda activate reorm

# Grounded-SAM local GPU build settings
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/usr/local/cuda-12.2

# Install Grounded-SAM
python -m pip install -e component/Grounded-Segment-Anything/segment_anything
pip install --no-build-isolation -e component/Grounded-Segment-Anything/GroundingDINO
pip install --upgrade diffusers[torch]

# API-LLM dependencies
pip install openai
```

### 5. Create `reorm_llava` environment

```bash
conda create -n reorm_llava python=3.10 -y
conda activate reorm_llava

pip install torch torchvision torchaudio
pip install transformers==4.41.2
pip install "unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git"
pip install bitsandbytes accelerate
pip install pillow requests
pip install opencv-python omegaconf diffusers
```

### 6. Model checkpoints and access

#### Grounded-SAM checkpoints

`src/tools/run_grounded_sam.py` expects these files under `component/Grounded-Segment-Anything/`:

- `groundingdino_swint_ogc.pth`
- `sam_vit_h_4b8939.pth`
- `sam_hq_vit_h.pth`

Download them from [Grounded-Segment-Anything](https://github.com/idea-research/grounded-segment-anything).

#### Hugging Face model access (multi-LLM)

`run_multi_llm.sh` loads:

- `llava-hf/llava-v1.6-vicuna-13b-hf`
- `meta-llama/Llama-3.1-8B-Instruct`

Please ensure you have accepted model licenses on Hugging Face.

---

## 💻 Usage

### Run API-LLM pipeline

Edit `run_api_llm.sh` and set your OpenAI API key `API_KEY="..."`.

```bash
./run_api_llm.sh
```

### Run Multi-LLM pipeline

```bash
./run_multi_llm.sh
```

### Change input/output paths

Edit these variables in both shell scripts if needed:

- `INPUT_FOLDER`
- `JSON_FOLDER`
- `OUTPUT_FOLDER`

---

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{huang2026icor,
  title={Interaction-Consistent Object Removal via MLLM-Based Reasoning},
  author={Huang, Ching-Kai and Lin, Wen-Chieh and Lee, Yan-Cen},
  journal={arXiv preprint arXiv:2602.01298},
  year={2026}
}
```