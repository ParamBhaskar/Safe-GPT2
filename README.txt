Roll Number: 21CS30035

Library Requirements

Python Version: 3.8+

Required Libraries:

- torch
- transformers
- pandas
- datasets
- trl
- tqdm
- evaluate
- rouge_score
- nltk

Ensure these dependencies are installed before running the code.

You can install all the required libraries using the following pip command:
pip install torch transformers pandas datasets trl==0.11.3 tqdm evaluate rouge_score nltk

If you're using Jupyter Notebook, run:
!pip install torch transformers pandas datasets trl==0.11.3 tqdm evaluate rouge_score nltk

Approach Details

Reward Model Training

Dataset Preparation:

The reward model is trained using a preference dataset where each data point consists of a question, a more preferred response, and a less preferred response.
Tokenization is performed using BertTokenizer.

Model Architecture:

The reward model is based on a BertForSequenceClassification architecture with a regression head to output a scalar reward score.

Training:

The model is trained to maximize the difference in reward scores between the preferred and less preferred responses.
The loss function used is a log-sigmoid of pairwise ranking loss, ensuring the model assigns higher scores to preferred responses.
AdamW optimizer with a learning rate scheduler is used.


Proximal Policy Optimization (PPO) Implementation

Policy Model:

Initialized using GPT2-medium, fine-tuned on supervised data.
PPO optimizes the model by interacting with the environment (reward model).
The reference model remains frozen during training.

Training Procedure:

The model generates responses to given prompts.
The reward model assigns scores to the responses.
The PPO loss is computed based on the reward feedback and policy updates.
KL-divergence constraints ensure the policy does not deviate too much from the original distribution.

Optimization:

Clip parameter ensures stable updates.
Gradient updates are applied iteratively with minibatches.


Direct Preference Optimization (DPO)

Dataset Handling:

Uses the preference dataset described above.
Inputs are tokenized and padded to maintain sequence consistency.

Loss Function:

The model learns a preference function by optimizing a log-sigmoid loss over response pairs.
Reference model remains frozen to provide baseline likelihoods.

Training:

The model generates log probabilities for preferred and non-preferred responses.
Computes DPO loss and updates policy network.
Trained using AdamW optimizer with gradient clipping.

Evaluation

The final PPO and DPO models are evaluated using:

BLEU Score: Measures n-gram overlap between generated and reference responses.
ROUGE Score: Evaluates recall-based text similarity.

Results indicate improvement in alignment and preference modeling.


Additional Notes

The trained models are saved as .pt and .safetensors files for reuse.
Ensure a GPU (cuda:0 and cuda:1) is available for sufficient as well as faster training and inference.
Logging and debugging can be monitored through print statements and tqdm progress bars.

For any issues, please check package versions and ensure proper GPU allocation.
