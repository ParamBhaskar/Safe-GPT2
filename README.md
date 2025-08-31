# Safe-GPT2

Safe-GPT2 is a project focused on fine-tuning the GPT2-medium language model to generate safe and ethically aligned responses, particularly aimed at handling harmful or stereotypic prompts. This repository implements two prominent techniques for preference-based fine-tuning: Reinforcement Learning from Human Feedback (RLHF) using Proximal Policy Optimization (PPO) and Direct Preference Optimization (DPO). The goal is to improve the quality, coherence and ethical alignment of model responses by leveraging human preferences.

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Installation](#installation)  
- [Dataset](#dataset)  
- [Methodology](#methodology)  
  - [Reward Model Training](#reward-model-training)  
  - [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)  
  - [Direct Preference Optimization (DPO)](#direct-preference-optimization-dpo)  
- [Evaluation](#evaluation)    
- [Results Summary](#results-summary)  
- [Trained Models](#trained-models)  
- [Future Work](#future-work)  
- [License](#license)  

---

## Overview

Large language models like GPT2 have demonstrated impressive generative capabilities but can produce unsafe, harmful or culturally insensitive outputs if not carefully fine-tuned. Safe-GPT2 addresses this by incorporating human preference data through RLHF and DPO methods to guide the language model towards safer and more aligned responses.

---

## Features

- Train a BERT-based reward model to score response quality based on human preference pairs.  
- Fine-tune GPT2-medium using RLHF (PPO) with a frozen reference model for stable policy updates.  
- Implement Direct Preference Optimization (DPO) as an alternative fine-tuning method without requiring a separate reward model.  
- Evaluation using automated metrics like BLEU and ROUGE to assess alignment and response quality.  
- Comparative analysis of RLHF vs. DPO in terms of sample efficiency, response coherence and computational cost.

---

## Installation

1. **Python Version:** Ensure Python 3.8 or higher is installed.

2. **Required Libraries:**  
```
pip install torch transformers pandas datasets trl==0.11.3 tqdm evaluate rouge_score nltk
```

3. **GPU:** A CUDA-compatible GPU is recommended for training and inference.

---

## Dataset

The project uses the **CulturalKaleidoscope_Preference** dataset, containing prompts with paired responses labeled as "more preferred" and "less preferred." This preference data is used to train the reward model and guide fine-tuning.

---

## Methodology

### Reward Model Training

- **Architecture:** BERT-base-uncased with a regression head predicts scalar reward scores.  
- **Objective:** Maximize the difference between reward scores of preferred and less preferred responses using a log-sigmoid pairwise ranking loss.  
- **Optimizer:** AdamW with a learning rate scheduler.  
- **Input Processing:** Tokenization with `BertTokenizer`.

### Proximal Policy Optimization (PPO)

- **Base Model:** GPT2-medium pretrained checkpoint.  
- **Training Loop:**  
  - Generate responses to prompts.  
  - Score responses using the reward model.  
  - Compute PPO loss with KL-divergence constraints to maintain policy closeness to the reference model.  
  - Update policy model while reference model remains frozen.  
- **Optimization:** Gradient clipping and iterative minibatch updates.

### Direct Preference Optimization (DPO)

- **Objective:** Directly optimize the preference probability ratio between preferred and less preferred responses.  
- **Advantage:** Eliminates the need for a separately trained reward model.  
- **Training:**  
  - Tokenize and pad preference dataset pairs.  
  - Compute DPO loss using log-sigmoid of likelihood ratios referenced against the frozen base model.  
  - Update policy parameters using AdamW with gradient clipping.

---

## Evaluation

- **Metrics:**  
  - **BLEU:** Measures n-gram precision.  
  - **ROUGE:** Measures recall and longest common subsequence for content coverage.  
- **Findings:**  
  - DPO achieves better response coherence and structural alignment than PPO, despite slightly lower BLEU scores.  
  - PPO occasionally outputs more token-completion-like responses.  
  - DPO demonstrates more stable training and faster convergence.  

---

## Results Summary

| Criterion          | RLHF (PPO)                   | DPO                       | Winner      |
|--------------------|------------------------------|---------------------------|-------------|
| Sample Efficiency  | Requires separate reward model; slower updates | Directly optimizes preferences rapidly | **DPO**     |
| Response Quality   | Some improvement, responses less direct | Clearer, more contextually relevant responses | **DPO**     |
| Computation Cost   | Over 11 hours, batch inefficiencies | ~5.3 hours, stable updates | **DPO**     |

---

## Trained Models

Pretrained models are available via the following links:

- Reward Model: [Download](https://www.kaggle.com/datasets/pbhaskar2003/assignment1-21cs30035-reward-model)  
- RLHF PPO Fine-tuned GPT2: [Download](https://www.kaggle.com/datasets/pbhaskar2003/assignment1-21cs30035-rlhf-trained)  
- DPO Fine-tuned GPT2: [Download](https://www.kaggle.com/datasets/pbhaskar2003/assignment1-21cs30035-dpo-trained)  

---

## Future Work

- Train models over multiple epochs to improve performance.  
- Hyperparameter tuning for better convergence and quality.  
- Explore alternative architectures and reinforcement learning algorithms (e.g. TRPO, A2C).  
- Incorporate pretrained reward models for more sample-efficiency.  
- Evaluate on larger, diverse datasets and assess generalization on out-of-distribution prompts.

---

## Acknowledgments

Special thanks to Prof. Animesh Mukherjee for guidance and mentorship throughout the project.

---