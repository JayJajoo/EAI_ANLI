"""
BERT model evaluation
"""

import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import Config
from utils.data_loader import load_anli_data
from models.bert.train import NLIDataset

def log_message(message, log_file):
    """Log to console and file"""
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def evaluate_bert_model(config_name):
    """
    Evaluate trained BERT model
    
    Args:
        config_name: Name of the configuration to evaluate
    
    Returns:
        dict: Evaluation results
    """
    artifact_dir = Config.get_model_artifact_dir('bert', config_name)
    log_file = os.path.join(artifact_dir, 'evaluation_log.txt')
    
    log_message("="*70, log_file)
    log_message(f"EVALUATING BERT MODEL: {config_name}", log_file)
    log_message("="*70, log_file)
    
    # Load model
    model_dir = os.path.join(artifact_dir, 'final_model')
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model not found at {model_dir}")
    
    log_message(f"\nLoading model from: {model_dir}", log_file)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(Config.DEVICE)
    model.eval()
    
    # Load test data
    _, _, test_df = load_anli_data()
    log_message(f"Test set size: {len(test_df)}", log_file)
    
    # Create test dataset
    test_dataset = NLIDataset(test_df, tokenizer, max_length=256)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        num_workers=0,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    
    # Evaluate
    log_message("\nRunning evaluation...", log_file)
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['labels'].to(Config.DEVICE)
            
            if Config.USE_FP16:
                with torch.amp.autocast('cuda'):
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
    test_loss = total_loss / len(test_loader)
    test_acc = accuracy_score(all_labels, all_predictions)
    test_f1_macro = f1_score(all_labels, all_predictions, average='macro')
    test_f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
    
    log_message("\n" + "="*70, log_file)
    log_message("TEST RESULTS", log_file)
    log_message("="*70, log_file)
    log_message(f"Test Loss: {test_loss:.4f}", log_file)
    log_message(f"Test Accuracy: {test_acc:.4f}", log_file)
    log_message(f"Test F1 (Macro): {test_f1_macro:.4f}", log_file)
    log_message(f"Test F1 (Weighted): {test_f1_weighted:.4f}", log_file)
    
    # Classification report
    log_message("\n" + "="*70, log_file)
    log_message("CLASSIFICATION REPORT", log_file)
    log_message("="*70, log_file)
    report = classification_report(all_labels, all_predictions, 
                                   target_names=Config.LABEL_NAMES, digits=4)
    log_message(report, log_file)
    
    report_dict = classification_report(all_labels, all_predictions,
                                       target_names=Config.LABEL_NAMES,
                                       output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    log_message("\n" + "="*70, log_file)
    log_message("CONFUSION MATRIX", log_file)
    log_message("="*70, log_file)
    log_message(str(cm), log_file)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.LABEL_NAMES,
                yticklabels=Config.LABEL_NAMES,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {config_name}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(artifact_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log_message(f"\nConfusion matrix saved: {cm_path}", log_file)
    
    # Baseline comparison
    acc_improvement = test_acc - Config.BASELINE_ACCURACY
    f1_improvement = test_f1_macro - Config.BASELINE_F1
    beats_baseline = test_acc > Config.BASELINE_ACCURACY and test_f1_macro > Config.BASELINE_F1
    
    log_message("\n" + "="*70, log_file)
    log_message("BASELINE COMPARISON", log_file)
    log_message("="*70, log_file)
    log_message(f"Baseline Accuracy: {Config.BASELINE_ACCURACY:.4f}", log_file)
    log_message(f"Current Accuracy: {test_acc:.4f}", log_file)
    log_message(f"Improvement: {acc_improvement:+.4f}", log_file)
    log_message(f"\nBaseline F1: {Config.BASELINE_F1:.4f}", log_file)
    log_message(f"Current F1: {test_f1_macro:.4f}", log_file)
    log_message(f"Improvement: {f1_improvement:+.4f}", log_file)
    
    if beats_baseline:
        log_message("\nMODEL BEATS BASELINE!", log_file)
    else:
        log_message("\nModel below baseline", log_file)
    
    # Save results
    results = {
        'config_name': config_name,
        'test_metrics': {
            'loss': float(test_loss),
            'accuracy': float(test_acc),
            'f1_macro': float(test_f1_macro),
            'f1_weighted': float(test_f1_weighted)
        },
        'classification_report': report_dict,
        'confusion_matrix': cm.tolist(),
        'baseline_comparison': {
            'baseline_accuracy': Config.BASELINE_ACCURACY,
            'baseline_f1': Config.BASELINE_F1,
            'accuracy_improvement': float(acc_improvement),
            'f1_improvement': float(f1_improvement),
            'beats_baseline': beats_baseline
        }
    }
    
    results_path = os.path.join(artifact_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    log_message(f"\nResults saved to: {results_path}", log_file)
    log_message("\nEvaluation completed!", log_file)
    
    return results