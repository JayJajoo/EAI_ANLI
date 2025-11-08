# BERT Fine-Tuning Report for ANLI

## Overview
This project involves fine-tuning BERT models for the Adversarial Natural Language Inference (ANLI) dataset, comparing multiple approaches including baseline DistilRoBERTa, BERT-base, BERT-large without prompt, and BERT-large with prompt. The goal is to study how model size and prompt-based input affect performance on a challenging NLI task.

EDA revealed that the mean similarity scores between entailment, neutral, and contradiction examples were very close, making it inherently difficult for models to distinguish between classes. This challenge persisted after training, which explains why ANLI remains a difficult dataset for achieving high accuracy, even with large models.

## Methodology
We followed a stepwise approach for fine-tuning:

1. **Baseline Evaluation**: Using `markusleonardo/DistilRoBERTa-For-Semantic-Similarity`, fine-tuned for NLI on SNLI/MNLI data to establish a reference point.
2. **BERT-base Fine-Tuning**: Fully fine-tuned on the ANLI dataset to observe improvements over the baseline.
3. **BERT-large Fine-Tuning**: Evaluated both without prompts and with manually crafted prompts to investigate prompt effectiveness for classification.

The use of prompts was intended to guide the model toward better semantic understanding and reduce ambiguity. However, results show that BERT-large did not significantly improve or degrade performance with prompts, suggesting the model may largely memorize the dataset patterns rather than relying on prompt guidance. ANLI is particularly challenging due to adversarial sentence pairs and high semantic ambiguity, making high accuracy difficult to achieve even for large models.

## Reproducibility
- `set_seed` and `torch.manual_seed`/`np.random.seed` ensure consistent results.
- Applies to PyTorch operations, NumPy computations, and Transformers initialization.

## Model Configuration
- Centralized `Config` class for hyperparameters and paths.
- Options include model selection (BERT-large), max sequence length, batch size, epochs, learning rate, dropout rates, early stopping thresholds, FP16 training, and device selection.

## Dataset Handling
- `NLIDataset` class handles tokenization, label conversion, and returns dicts with `input_ids`, `attention_mask`, and labels.
- Utilizes PyTorch DataLoader for batching and efficient GPU utilization.

## Optimizer and Scheduler
- AdamW with selective weight decay.
- Linear learning rate scheduler with warmup (first 10% of steps).

## Training Enhancements
- **Gradient Accumulation**: Allows larger effective batch sizes.
- **Mixed Precision (FP16)**: Reduces memory and speeds up training.
- **Gradient Clipping**: Stabilizes training and prevents exploding gradients.
- **Early Stopping**: Monitors validation F1 to prevent overfitting.

## Evaluation Metrics
- Tracks Loss, Accuracy, Macro F1, Weighted F1.
- Confusion matrix computed for detailed class-level performance.

## Model Checkpointing and Logging
- Saves best model and training history in JSON.
- Enables easy resumption and inference consistency.

## Inference Utility
- `predict_nli()` returns predicted labels and confidence probabilities using the trained tokenizer and model.

## Results Summary
| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| DistilRoBERTa Baseline | 0.337 | 0.242 | 0.242 |
| BERT-base | 0.431 | 0.427 | 0.427 |
| BERT-large without prompt | 0.447 | 0.443 | 0.443 |
| BERT-large with prompt | 0.454 | 0.450 | 0.450 |

**Observations:**
- Prompting did not significantly change performance.
- BERT-large outperforms smaller models due to higher capacity.
- ANLI remains challenging due to adversarial examples and semantic complexity.

## Key Features Summary
- Reproducibility through seeds
- Configurable hyperparameters and model selection
- Efficient dataset handling
- Optimizer with selective weight decay
- Gradient accumulation and FP16 for large batch training
- Gradient clipping and early stopping
- Detailed evaluation metrics and logging
- Model checkpointing for easy reuse
- Inference helper function for predictions

---
