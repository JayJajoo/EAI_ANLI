"""
Evaluation pipeline for trained NLI model
Loads best model and evaluates on test set
Checks if performance beats baseline metrics
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset

from config import Config

# ============================================================================
# SETUP
# ============================================================================

def log_message(message, log_file=Config.EVAL_LOG):
    """Log to console and file"""
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

# ============================================================================
# DATASET (reuse from training)
# ============================================================================

class NLIDataset(Dataset):
    """Custom dataset for NLI task"""
    def __init__(self, dataframe, tokenizer, max_length=256):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        premise = str(self.data.loc[idx, 'premise']).strip()
        hypothesis = str(self.data.loc[idx, 'hypothesis']).strip()
        label = int(self.data.loc[idx, 'label'])
        
        encoding = self.tokenizer(
            premise,
            hypothesis,
            max_length=self.max_length,
            padding='max_length',
            truncation='longest_first',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ============================================================================
# LOAD MODEL & DATA
# ============================================================================

def load_trained_model():
    """Load the best trained model"""
    log_message("Loading trained model...")
    
    # Check if model exists
    if not os.path.exists(Config.FINAL_MODEL_DIR):
        raise FileNotFoundError(f"Model not found at {Config.FINAL_MODEL_DIR}")
    
    tokenizer = AutoTokenizer.from_pretrained(Config.FINAL_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(Config.FINAL_MODEL_DIR)
    model.to(Config.DEVICE)
    model.eval()
    
    log_message(f"Model loaded from: {Config.FINAL_MODEL_DIR}")
    log_message(f"Device: {Config.DEVICE}")
    
    return model, tokenizer

def load_test_data():
    """Load and prepare test dataset"""
    log_message("\nLoading test dataset...")
    
    ds = load_dataset(Config.DATASET_NAME)
    test_data = ds[Config.TEST_SPLIT]
    
    # Convert to DataFrame with preprocessing
    test_df = pd.DataFrame({
        'premise': [p.lower().strip() for p in test_data['premise']],
        'hypothesis': [h.lower().strip() for h in test_data['hypothesis']],
        'label': test_data['label']
    })
    
    log_message(f"Test set size: {len(test_df)} examples")
    
    return test_df

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, test_loader, device):
    """Run full evaluation on test set"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    log_message("\nRunning evaluation on test set...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            if Config.USE_FP16:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            all_predictions.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
    
    return avg_loss, accuracy, f1_macro, f1_weighted, all_predictions, all_labels

def generate_classification_report(y_true, y_pred):
    """Generate detailed classification report"""
    log_message("\n" + "="*70)
    log_message("CLASSIFICATION REPORT")
    log_message("="*70)
    
    report = classification_report(
        y_true,
        y_pred,
        target_names=Config.LABEL_NAMES,
        digits=4
    )
    
    log_message(report)
    
    # Also get as dictionary for JSON saving
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=Config.LABEL_NAMES,
        output_dict=True
    )
    
    return report, report_dict

def generate_confusion_matrix(y_true, y_pred):
    """Generate and plot confusion matrix"""
    log_message("\n" + "="*70)
    log_message("CONFUSION MATRIX")
    log_message("="*70)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Log matrix
    log_message("\nAbsolute counts:")
    log_message(str(cm))
    
    # Per-class accuracy
    log_message("\nPer-class recall:")
    for i, label_name in enumerate(Config.LABEL_NAMES):
        class_recall = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        log_message(f"  {label_name}: {class_recall:.4f} ({cm[i, i]}/{cm[i].sum()})")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=Config.LABEL_NAMES,
        yticklabels=Config.LABEL_NAMES,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Test Set', fontsize=14, pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(Config.CONFUSION_MATRIX_PLOT, dpi=300, bbox_inches='tight')
    plt.close()
    
    log_message(f"\nConfusion matrix saved to: {Config.CONFUSION_MATRIX_PLOT}")
    
    return cm

def compare_with_baseline(accuracy, f1_macro):
    """Compare results with baseline metrics"""
    log_message("\n" + "="*70)
    log_message("BASELINE COMPARISON")
    log_message("="*70)
    
    acc_improvement = accuracy - Config.BASELINE_ACCURACY
    f1_improvement = f1_macro - Config.BASELINE_F1
    
    acc_relative = (acc_improvement / Config.BASELINE_ACCURACY) * 100
    f1_relative = (f1_improvement / Config.BASELINE_F1) * 100
    
    comparison = {
        'baseline_accuracy': Config.BASELINE_ACCURACY,
        'current_accuracy': float(accuracy),
        'accuracy_improvement_absolute': float(acc_improvement),
        'accuracy_improvement_relative_pct': float(acc_relative),
        'baseline_f1': Config.BASELINE_F1,
        'current_f1': float(f1_macro),
        'f1_improvement_absolute': float(f1_improvement),
        'f1_improvement_relative_pct': float(f1_relative),
        'beats_baseline': bool(accuracy > Config.BASELINE_ACCURACY and f1_macro > Config.BASELINE_F1)
    }
    
    log_message(f"\nBaseline Accuracy: {Config.BASELINE_ACCURACY:.4f}")
    log_message(f"Current Accuracy:  {accuracy:.4f}")
    log_message(f"Improvement:       {acc_improvement:+.4f} ({acc_relative:+.2f}%)")
    
    log_message(f"\nBaseline F1:       {Config.BASELINE_F1:.4f}")
    log_message(f"Current F1:        {f1_macro:.4f}")
    log_message(f"Improvement:       {f1_improvement:+.4f} ({f1_relative:+.2f}%)")
    
    if comparison['beats_baseline']:
        log_message("\nMODEL BEATS BASELINE METRICS")
    else:
        log_message("\nMODEL DOES NOT BEAT BASELINE")
    
    return comparison

# ============================================================================
# TEST EXAMPLES
# ============================================================================

def test_example_predictions(model, tokenizer):
    """Test on example inputs"""
    log_message("\n" + "="*70)
    log_message("EXAMPLE PREDICTIONS")
    log_message("="*70)
    
    examples = [
        ("A person is riding a bike.", "Someone is cycling."),
        ("The sky is blue.", "It is raining."),
        ("A dog is running in the park.", "An animal is outside.")
    ]
    
    model.eval()
    label_map = {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
    
    example_results = []
    
    for premise, hypothesis in examples:
        encoding = tokenizer(
            premise,
            hypothesis,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation='longest_first',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(Config.DEVICE)
        attention_mask = encoding['attention_mask'].to(Config.DEVICE)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            prediction = torch.argmax(logits, dim=1).item()
        
        pred_label = label_map[prediction]
        prob_dict = {label_map[i]: float(probs[i]) for i in range(3)}
        
        log_message(f"\nPremise: {premise}")
        log_message(f"Hypothesis: {hypothesis}")
        log_message(f"Prediction: {pred_label}")
        log_message(f"Confidence: {prob_dict}")
        
        example_results.append({
            'premise': premise,
            'hypothesis': hypothesis,
            'prediction': pred_label,
            'probabilities': prob_dict
        })
    
    return example_results

# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    """Main evaluation pipeline"""
    
    # Create directories
    Config.create_directories()
    
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message("="*70)
    log_message(f"EVALUATION PIPELINE STARTED: {start_time}")
    log_message("="*70)
    
    # Load model and data
    model, tokenizer = load_trained_model()
    test_df = load_test_data()
    
    # Create test dataset and loader
    test_dataset = NLIDataset(test_df, tokenizer, Config.MAX_LENGTH)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=0,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    
    # Evaluate
    test_loss, test_acc, test_f1_macro, test_f1_weighted, predictions, labels = evaluate_model(
        model, test_loader, Config.DEVICE
    )
    
    # ========================================================================
    # RESULTS
    # ========================================================================
    
    log_message("\n" + "="*70)
    log_message("TEST SET RESULTS")
    log_message("="*70)
    
    log_message(f"\nTest Loss:       {test_loss:.4f}")
    log_message(f"Test Accuracy:   {test_acc:.4f}")
    log_message(f"Test F1 (Macro): {test_f1_macro:.4f}")
    log_message(f"Test F1 (Weighted): {test_f1_weighted:.4f}")
    
    # Classification report
    report, report_dict = generate_classification_report(labels, predictions)
    
    # Confusion matrix
    cm = generate_confusion_matrix(labels, predictions)
    
    # Compare with baseline
    comparison = compare_with_baseline(test_acc, test_f1_macro)
    
    # Test examples
    example_results = test_example_predictions(model, tokenizer)
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    log_message("\n" + "="*70)
    log_message("SAVING EVALUATION RESULTS")
    log_message("="*70)
    
    results = {
        'timestamp': start_time,
        'model_path': Config.FINAL_MODEL_DIR,
        'test_metrics': {
            'loss': float(test_loss),
            'accuracy': float(test_acc),
            'f1_macro': float(test_f1_macro),
            'f1_weighted': float(test_f1_weighted)
        },
        'classification_report': report_dict,
        'confusion_matrix': cm.tolist(),
        'baseline_comparison': comparison,
        'example_predictions': example_results
    }
    
    # Save results JSON
    with open(Config.EVAL_RESULTS, 'w') as f:
        json.dump(results, f, indent=2)
    log_message(f"Results saved to: {Config.EVAL_RESULTS}")
    
    # ========================================================================
    # FINAL STATUS
    # ========================================================================
    
    log_message("\n" + "="*70)
    log_message("EVALUATION SUMMARY")
    log_message("="*70)
    
    log_message(f"\nTest Accuracy: {test_acc:.4f}")
    log_message(f"Test F1 Score: {test_f1_macro:.4f}")
    
    if comparison['beats_baseline']:
        log_message("\nMODEL SUCCESSFULLY BEATS BASELINE")
        log_message(f"  Accuracy improved by {comparison['accuracy_improvement_absolute']:.4f} ({comparison['accuracy_improvement_relative_pct']:.2f}%)")
        log_message(f"  F1 improved by {comparison['f1_improvement_absolute']:.4f} ({comparison['f1_improvement_relative_pct']:.2f}%)")
    else:
        log_message("\nMODEL DOES NOT BEAT BASELINE")
        log_message("  Review training process and hyperparameters")
    
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message(f"\nEvaluation completed: {end_time}")
    log_message(f"All results saved to: {Config.EVAL_DIR}")
    
    return results

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        results = main()
        
        print("\n" + "="*70)
        print("EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Results saved to: {Config.EVAL_DIR}")
        print(f"Accuracy: {results['test_metrics']['accuracy']:.4f}")
        print(f"F1 Score: {results['test_metrics']['f1_macro']:.4f}")
        
        if results['baseline_comparison']['beats_baseline']:
            print("\nMODEL BEATS BASELINE METRICS!")
        else:
            print("\nModel performance below baseline")
            
    except Exception as e:
        error_msg = f"\nEvaluation failed: {str(e)}"
        log_message(error_msg)
        raise