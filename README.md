

# 🌟 PalmScore

Welcome to the official code repository for **[Enhanced LLM-As-A-Judge with Layer-Wise Logits Aggregation]**! 🚀 This repository contains all the code, resources, and instructions needed to explore our work and reproduce the experimental results from our paper. If you are interested in our work or wish to reproduce our experimental results, please read the following sections carefully.

<p align="center">
  <img src="./image/PalmScore.png" alt="PalmScore Overview" width="600"/>
</p>

---

## 📜 Abstract

**PalmScore** introduces a novel approach to LLM-as-a-judge, enabling fine-grained, flexible, and reliable evaluation for tasks like model response assessment and data synthesis. Unlike traditional prompt-based or finetuning-based evaluators, which often struggle with judge performance or generalization, our method leverages **layer-wise aggregated logits** from a general LLM. 

Inspired by the insight that middle-to-top layers capture critical judgment information, we aggregate logits across all layers using lightweight weight parameters, keeping the LLM backbone frozen. This results in a robust and fine-grained judgment score. 

🔍 **Key Highlights**:
- Outperforms baseline methods on **Flask**, **HelpSteer**, and **BIGGen** benchmarks.
- Achieves comparable or superior performance to reasoning-based baselines, even without reasoning.
- Excels in downstream tasks like **data selection** and **knowledge boundary detection**.

---

## 📑 Table of Contents

- [🌟 PalmScore](#-palmscore)
- [📜 Abstract](#-abstract)
- [🛠️ Environment Setup](##-Environment-Setup)
- [🔬 Experiments](##experiments)
- [📝 Citation](#-citation)

---

## 🛠️ Environment Setup

Follow the instructions below to set up the environment.

```bash
# Create a new conda environment
conda create --name PalmScore python=3.10 -y

# Activate the environment
conda activate PalmScore

# Clone the repository
git clone https://github.com/Laip11/PalmScore.git

# Navigate to the project directory
cd PalmScore

# Install dependencies
pip install -r requirements.txt
```

## 🔬 Experiments

## 📝 Citation

If you would like others to cite your paper, provide the BibTeX citation format here:
```bibtex
