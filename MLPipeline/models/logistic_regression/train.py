"""
Logistic Regression model training
"""

import os
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime

from config import Config
from utils.data_loader import load_anli_data

def log_message(message, log_file):
    """Log to console and file"""
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def train_logistic_regression(lr_config):
    """
    Train Logistic Regression model with given configuration
    
    Args:
        lr_config: Dictionary containing model configuration
    
    Returns:
        dict: Training results
    """
    config_name = lr_config['name']
    artifact_dir = Config.get_model_artifact_dir('logistic_regression', config_name)
    log_file = os.path.join(artifact_dir, 'training_log.txt')
    
    log_message("="*70, log_file)
    log_message(f"TRAINING LOGISTIC REGRESSION: {config_name}", log_file)
    log_message("="*70, log_file)
    
    # Load data
    log_message("\nLoading data...", log_file)
    train_df, dev_df, test_df = load_anli_data()
    
    # Combine premise and hypothesis
    train_df['text'] = train_df['premise'] + ' ' + train_df['hypothesis']
    dev_df['text'] = dev_df['premise'] + ' ' + dev_df['hypothesis']
    test_df['text'] = test_df['premise'] + ' ' + test_df['hypothesis']
    
    # Extract TF-IDF features
    log_message("\nExtracting TF-IDF features...", log_file)
    log_message(f"Max features: {lr_config['tfidf_max_features']}", log_file)
    log_message(f"N-gram range: {lr_config['tfidf_ngram_range']}", log_file)
    
    vectorizer = TfidfVectorizer(
        max_features=lr_config['tfidf_max_features'],
        ngram_range=lr_config['tfidf_ngram_range'],
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train = vectorizer.fit_transform(train_df['text'])
    X_dev = vectorizer.transform(dev_df['text'])
    X_test = vectorizer.transform(test_df['text'])
    
    y_train = train_df['label'].values
    y_dev = dev_df['label'].values
    y_test = test_df['label'].values
    
    log_message(f"\nFeature matrix shapes:", log_file)
    log_message(f"  Train: {X_train.shape}", log_file)
    log_message(f"  Dev: {X_dev.shape}", log_file)
    log_message(f"  Test: {X_test.shape}", log_file)
    
    # Train model
    log_message("\n" + "="*70, log_file)
    log_message("TRAINING MODEL", log_file)
    log_message("="*70, log_file)
    log_message(f"Max iterations: {lr_config['max_iter']}", log_file)
    log_message(f"C (regularization): {lr_config['C']}", log_file)
    log_message(f"Solver: {lr_config['solver']}", log_file)
    log_message(f"Penalty: {lr_config['penalty']}", log_file)
    
    model = LogisticRegression(
        max_iter=lr_config['max_iter'],
        C=lr_config['C'],
        solver=lr_config['solver'],
        penalty=lr_config['penalty'],
        class_weight=lr_config['class_weight'],
        random_state=Config.SEED,
        n_jobs=-1,
        verbose=0
    )
    
    log_message("\nFitting model...", log_file)
    model.fit(X_train, y_train)
    log_message("Training complete!", log_file)
    
    # Evaluate on all splits
    train_preds = model.predict(X_train)
    dev_preds = model.predict(X_dev)
    test_preds = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_preds)
    train_f1 = f1_score(y_train, train_preds, average='macro')
    
    dev_acc = accuracy_score(y_dev, dev_preds)
    dev_f1 = f1_score(y_dev, dev_preds, average='macro')
    
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average='macro')
    
    log_message("\n" + "="*70, log_file)
    log_message("RESULTS", log_file)
    log_message("="*70, log_file)
    log_message(f"Train - Acc: {train_acc:.4f}, F1: {train_f1:.4f}", log_file)
    log_message(f"Dev   - Acc: {dev_acc:.4f}, F1: {dev_f1:.4f}", log_file)
    log_message(f"Test  - Acc: {test_acc:.4f}, F1: {test_f1:.4f}", log_file)
    
    # Save model and vectorizer
    model_path = os.path.join(artifact_dir, 'model.pkl')
    vectorizer_path = os.path.join(artifact_dir, 'vectorizer.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    log_message(f"\nModel saved to: {model_path}", log_file)
    log_message(f"Vectorizer saved to: {vectorizer_path}", log_file)
    
    # Save training results
    results = {
        'config_name': config_name,
        'config': lr_config,
        'train_metrics': {
            'accuracy': float(train_acc),
            'f1_macro': float(train_f1)
        },
        'dev_metrics': {
            'accuracy': float(dev_acc),
            'f1_macro': float(dev_f1)
        },
        'test_metrics': {
            'accuracy': float(test_acc),
            'f1_macro': float(test_f1)
        },
        'artifact_dir': artifact_dir
    }
    
    results_path = os.path.join(artifact_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    log_message("\nTraining completed!", log_file)
    
    return results