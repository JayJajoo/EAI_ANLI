"""
BERT model training
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    set_seed
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from datetime import datetime
import numpy as np

from config import Config
from utils.data_loader import load_anli_data

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

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

def log_message(message, log_file):
    """Log to console and file"""
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def train_epoch(model, dataloader, optimizer, scheduler, device, scaler, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        if scaler:
            with torch.amp.autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / config['gradient_accumulation_steps']
            
            scaler.scale(loss).backward()
            
            if (step + 1) % config['gradient_accumulation_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / config['gradient_accumulation_steps']
            loss.backward()
            
            if (step + 1) % config['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * config['gradient_accumulation_steps']
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({
            'loss': loss.item() * config['gradient_accumulation_steps'],
            'lr': scheduler.get_last_lr()[0]
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro')
    
    return avg_loss, accuracy, f1_macro

def validate(model, dataloader, device, config):
    """Validate model"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
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
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro')
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    
    return avg_loss, accuracy, f1_macro, f1_weighted

def train_bert_model(bert_config):
    """
    Train BERT model with given configuration
    
    Args:
        bert_config: Dictionary containing model configuration
    
    Returns:
        dict: Training results
    """
    config_name = bert_config['name']
    artifact_dir = Config.get_model_artifact_dir('bert', config_name)
    log_file = os.path.join(artifact_dir, 'training_log.txt')
    
    log_message("="*70, log_file)
    log_message(f"TRAINING BERT MODEL: {config_name}", log_file)
    log_message(f"Model: {bert_config['model_name']}", log_file)
    log_message("="*70, log_file)
    
    # Set seed
    set_seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    # Load data
    log_message("\nLoading data...", log_file)
    train_df, val_df, test_df = load_anli_data()
    
    # Initialize model and tokenizer
    log_message(f"\nInitializing model: {bert_config['model_name']}", log_file)
    tokenizer = AutoTokenizer.from_pretrained(bert_config['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        bert_config['model_name'],
        num_labels=Config.NUM_LABELS,
        hidden_dropout_prob=bert_config['hidden_dropout_prob'],
        attention_probs_dropout_prob=bert_config['attention_dropout_prob']
    )
    model.to(Config.DEVICE)
    
    log_message(f"Device: {Config.DEVICE}", log_file)
    log_message(f"FP16: {Config.USE_FP16}", log_file)
    
    # Create datasets and loaders
    train_dataset = NLIDataset(train_df, tokenizer, bert_config['max_length'])
    val_dataset = NLIDataset(val_df, tokenizer, bert_config['max_length'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=bert_config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bert_config['batch_size'],
        num_workers=0,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    
    # Setup optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': bert_config['weight_decay']
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=bert_config['learning_rate'],
        eps=bert_config['adam_epsilon']
    )
    
    total_steps = len(train_loader) * bert_config['epochs'] // bert_config['gradient_accumulation_steps']
    warmup_steps = int(total_steps * bert_config['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    scaler = torch.cuda.amp.GradScaler() if Config.USE_FP16 else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=bert_config['patience'],
        min_delta=bert_config['min_delta']
    )
    
    # Training history
    best_val_f1 = 0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    log_message("\n" + "="*70, log_file)
    log_message("STARTING TRAINING", log_file)
    log_message("="*70, log_file)
    
    # Training loop
    for epoch in range(bert_config['epochs']):
        log_message(f"\nEpoch {epoch + 1}/{bert_config['epochs']}", log_file)
        log_message("-" * 70, log_file)
        
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, Config.DEVICE, scaler, bert_config
        )
        log_message(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}", log_file)
        
        val_loss, val_acc, val_f1_macro, val_f1_weighted = validate(
            model, val_loader, Config.DEVICE, bert_config
        )
        log_message(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1_macro:.4f}", log_file)
        
        # Save metrics
        training_history['train_loss'].append(float(train_loss))
        training_history['train_acc'].append(float(train_acc))
        training_history['train_f1'].append(float(train_f1))
        training_history['val_loss'].append(float(val_loss))
        training_history['val_acc'].append(float(val_acc))
        training_history['val_f1'].append(float(val_f1_macro))
        
        # Save best model
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            checkpoint_path = os.path.join(artifact_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1': val_f1_macro,
                'val_acc': val_acc,
                'config': bert_config
            }, checkpoint_path)
            log_message(f"Best model saved (F1: {val_f1_macro:.4f})", log_file)
        
        # Early stopping
        early_stopping(val_f1_macro)
        if early_stopping.early_stop:
            log_message(f"\nEarly stopping triggered at epoch {epoch + 1}", log_file)
            break
    
    # Load best checkpoint and save final model
    checkpoint_path = os.path.join(artifact_dir, 'best_model.pt')
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_model_dir = os.path.join(artifact_dir, 'final_model')
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    log_message(f"\nModel saved to: {final_model_dir}", log_file)
    
    # Save training history
    history_path = os.path.join(artifact_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    log_message("\nTraining completed!", log_file)
    
    return {
        'config_name': config_name,
        'best_val_f1': float(best_val_f1),
        'best_val_acc': float(checkpoint['val_acc']),
        'artifact_dir': artifact_dir
    }