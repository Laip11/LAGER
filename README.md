# PalmScore
The official implementation of the paper "Enhanced LLM-As-A-Judge with Layer-Wise Logits Aggregation"

## Abstract

LLM-as-a-judge offers fine-grained, flexible and reliable evaluation in various applications like model response evaluation and data synthesis. Previous prompt-based or finetuning-based LLM evaluators have unsatisfactory judge performance or face the problem of generalization. In this work, we propose to estimate the judge score from a general LLM with layer-wise aggregated logits. Motivated by preliminary observations that the middle to top layers collect information required for judgment, we propose aggregating the logits from all layers with lightweight weight parameters while keeping the LLM backbone frozen. Our proposed PalmScore method integrates logits across layers and probabilities across scores to produce a fine-grained judgment score.
Experiments demonstrate the superior performance of the PalmScore method over baseline methods across Flask, HelpSteer, and BIGGen bench. The PalmScore method, when making direct judgments without reasoning, achieves performance that is comparable to or even surpasses that of baseline methods that employ a reasoning process. Extensive experiments on downstream applications like data selection and knowledge boundary detection further prove the effectiveness of our method.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Models](#models)
- [Experiments](#experiments)
- [Citation](#citation)


# Environment Setup
To install the required packages, please follow the instructions below.

```bash
conda create --name PalmScore python=3.10 -y
conda activate PalmScore
git clone https://github.com/Laip11/PalmScore.git
cd PalmScore
pip install -r requirements.txt
```
