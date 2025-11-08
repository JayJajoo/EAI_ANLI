"""
Logistic Regression model evaluation
"""

import os
import json
import pickle
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from utils.data_loader import load_anli_data

def log_message(message, log_file):
    """Log to console and file"""
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def evaluate_logistic_regression(config_name):
    """
    Evaluate trained Logistic Regression model
    
    Args:
        config_name: Name of the configuration to evaluate
    
    Returns:
        dict: Evaluation results
    """
    artifact_dir = Config.get_model_artifact_dir('logistic_regression', config_name)
    log_file = os.path.join(artifact_dir, 'evaluation_log.txt')
    
    log_message("="*70, log_file)
    log_message(f"EVALUATING LOGISTIC REGRESSION: {config_name}", log_file)
    log_message("="*70, log_file)
    
    # Load model and vectorizer
    model_path = os.path.join(artifact_dir, 'model.pkl')
    vectorizer_path = os.path.join(artifact_dir, 'vectorizer.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    log_message(f"\nLoading model from: {model_path}", log_file)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Load test data
    _, _, test_df = load_anli_data()
    log_message(f"Test set size: {len(test_df)}", log_file)
    
    # Prepare features
    test_df['text'] = test_df['premise'] + ' ' + test_df['hypothesis']
    X_test = vectorizer.transform(test_df['text'])
    y_test = test_df['label'].values
    
    # Evaluate
    log_message("\nRunning evaluation...", log_file)
    test_preds = model.predict(X_test)
    
    test_acc = accuracy_score(y_test, test_preds)
    test_f1_macro = f1_score(y_test, test_preds, average='macro')
    test_f1_weighted = f1_score(y_test, test_preds, average='weighted')
    
    log_message("\n" + "="*70, log_file)
    log_message("TEST RESULTS", log_file)
    log_message("="*70, log_file)
    log_message(f"Test Accuracy: {test_acc:.4f}", log_file)
    log_message(f"Test F1 (Macro): {test_f1_macro:.4f}", log_file)
    log_message(f"Test F1 (Weighted): {test_f1_weighted:.4f}", log_file)
    
    # Classification report
    log_message("\n" + "="*70, log_file)
    log_message("CLASSIFICATION REPORT", log_file)
    log_message("="*70, log_file)
    report = classification_report(y_test, test_preds,
                                   target_names=Config.LABEL_NAMES, digits=4)
    log_message(report, log_file)
    
    report_dict = classification_report(y_test, test_preds,
                                       target_names=Config.LABEL_NAMES,
                                       output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, test_preds)
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