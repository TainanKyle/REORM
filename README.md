# Interaction-Consistent Object Removal via MLLM-Based Reasoning

Official implementation of the paper **"Interaction-Consistent Object Removal via MLLM-Based Reasoning"**.

[![arXiv](https://img.shields.io/badge/arXiv-2602.01298-b31b1b.svg)](https://arxiv.org/abs/2602.01298)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

## 📌 Overview

Traditional object removal is often limited to erasures within the boundaries of a provided mask, focusing solely on the target object. However, this often leaves behind interaction evidence that are semantically tied to the target—resulting in edited outputs that are logically inconsistent.We formalize this challenge as **Interaction-Consistent Object Removal (ICOR)**. Unlike standard methods, ICOR demands a holistic understanding of the entire image and the interaction relationships among objects. When an object is removed, the system must infer the plausible appearance of the scene in its absence and identify all secondary elements that would become illogical or inconsistent.

---

## 🚀 Method: REORM

We propose **Reasoning-Enhanced Object Removal with MLLM (REORM)**, a modular framework that leverages the broad commonsense reasoning ability of Multimodal Large Language Models (MLLMs) to infer which additional elements in a scene would become inconsistent if the target were absent.

### Key Components:
- MLLM-driven Analysis: Employs a chain-of-thought prompting strategy to interpret instructions and identify both target objects and associated elements.
- Self-Correction Mechanism: An MLLM-controlled stage that simulates the expected result and resolves remaining inconsistencies or artifacts.
- Local Deployment Variant: A strategy utilizing MLLM-LLM collaboration and prompt chaining to enable robust reasoning on resource-constrained devices.

---

## 🛠️ Installation

Detailed environment setup and dependency installation steps will be added soon.

---

## 💻 Usage

Instructions for running inference, evaluation, and the REORM pipeline will be added soon.

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