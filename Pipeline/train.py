"""
Training pipeline for BERT NLI fine-tuning
Trains model and saves artifacts to ./artifacts/model/
"""

import os
import json
import pandas as pd
import numpy as np
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
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from datetime import datetime

from config import Config

# ============================================================================
# SETUP
# ============================================================================

def setup_environment():
    """Initialize directories and set random seeds"""
    Config.create_directories()
    set_seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_message(message, log_file=Config.TRAINING_LOG):
    """Log to console and file"""
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

# ============================================================================
# DATASET
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

def load_and_prepare_data():
    """Load ANLI dataset and convert to DataFrames"""
    log_message("\nLoading ANLI dataset...")
    ds = load_dataset(Config.DATASET_NAME)
    
    train_data = ds[Config.TRAIN_SPLIT]
    dev_data = ds[Config.DEV_SPLIT]
    test_data = ds[Config.TEST_SPLIT]
    
    log_message(f"Train: {len(train_data)} examples")
    log_message(f"Dev: {len(dev_data)} examples")
    log_message(f"Test: {len(test_data)} examples")
    
    # Convert to DataFrames with preprocessing
    def to_df(dataset):
        return pd.DataFrame({
            'premise': [p.lower().strip() for p in dataset['premise']],
            'hypothesis': [h.lower().strip() for h in dataset['hypothesis']],
            'label': dataset['label']
        })
    
    train_df = to_df(train_data)
    val_df = to_df(dev_data)
    test_df = to_df(test_data)
    
    # Log label distribution
    label_dist = train_df['label'].value_counts().sort_index()
    log_message("\nTraining label distribution:")
    for label, count in label_dist.items():
        log_message(f"  {Config.LABEL_NAMES[label]}: {count} ({count/len(train_df)*100:.2f}%)")
    
    return train_df, val_df, test_df

# ============================================================================
# MODEL & TRAINING SETUP
# ============================================================================

def setup_model_and_tokenizer():
    """Initialize model and tokenizer"""
    log_message(f"\nInitializing model: {Config.MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=Config.NUM_LABELS,
        hidden_dropout_prob=Config.HIDDEN_DROPOUT_PROB,
        attention_probs_dropout_prob=Config.ATTENTION_DROPOUT_PROB
    )
    model.to(Config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    log_message(f"Total parameters: {total_params:,}")
    log_message(f"Trainable parameters: {trainable_params:,}")
    log_message(f"Device: {Config.DEVICE}")
    log_message(f"Mixed precision (FP16): {Config.USE_FP16}")
    
    return model, tokenizer

def create_data_loaders(train_df, val_df, test_df, tokenizer):
    """Create PyTorch DataLoaders"""
    train_dataset = NLIDataset(train_df, tokenizer, Config.MAX_LENGTH)
    val_dataset = NLIDataset(val_df, tokenizer, Config.MAX_LENGTH)
    test_dataset = NLIDataset(test_df, tokenizer, Config.MAX_LENGTH)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=0,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=0,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    
    return train_loader, val_loader, test_loader

def setup_optimizer_and_scheduler(model, train_loader):
    """Setup optimizer and learning rate scheduler"""
    # Separate parameters for weight decay
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': Config.WEIGHT_DECAY
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=Config.LEARNING_RATE,
        eps=Config.ADAM_EPSILON
    )
    
    # Scheduler with warmup
    total_steps = len(train_loader) * Config.EPOCHS // Config.GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    log_message(f"Total training steps: {total_steps}")
    log_message(f"Warmup steps: {warmup_steps}")
    
    return optimizer, scheduler

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, scaler=None):
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
        
        # Forward pass with mixed precision if available
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / Config.GRADIENT_ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
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
            loss = outputs.loss / Config.GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            if (step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS
        
        # Collect predictions
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({
            'loss': loss.item() * Config.GRADIENT_ACCUMULATION_STEPS,
            'lr': scheduler.get_last_lr()[0]
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro')
    
    return avg_loss, accuracy, f1_macro

def validate(model, dataloader, device):
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
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro')
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    
    return avg_loss, accuracy, f1_macro, f1_weighted, predictions, true_labels

# ============================================================================
# EARLY STOPPING
# ============================================================================

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

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    """Main training pipeline"""
    
    # Setup
    start_time = setup_environment()
    log_message("="*70)
    log_message(f"TRAINING PIPELINE STARTED: {start_time}")
    log_message("="*70)
    
    # Load data
    train_df, val_df, test_df = load_and_prepare_data()
    
    # Initialize model
    model, tokenizer = setup_model_and_tokenizer()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df, tokenizer
    )
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, train_loader)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if Config.USE_FP16 else None
    if scaler:
        log_message("Using mixed precision training (FP16)")
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE,
        min_delta=Config.MIN_DELTA
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
    
    log_message("\n" + "="*70)
    log_message("STARTING TRAINING")
    log_message("="*70)
    
    # Training loop
    for epoch in range(Config.EPOCHS):
        log_message(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        log_message("-" * 70)
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, Config.DEVICE, scaler
        )
        log_message(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        
        # Validate
        val_loss, val_acc, val_f1_macro, val_f1_weighted, _, _ = validate(
            model, val_loader, Config.DEVICE
        )
        log_message(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1_macro:.4f}")
        
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1': val_f1_macro,
                'val_acc': val_acc,
                'config': Config.get_config_dict()
            }, Config.CHECKPOINT_PATH)
            log_message(f"Best model saved (F1: {val_f1_macro:.4f})")
        
        # Early stopping check
        early_stopping(val_f1_macro)
        if early_stopping.early_stop:
            log_message(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
    
    # ========================================================================
    # SAVE FINAL MODEL
    # ========================================================================
    
    log_message("\n" + "="*70)
    log_message("SAVING FINAL MODEL")
    log_message("="*70)
    
    # Load best checkpoint
    checkpoint = torch.load(Config.CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    log_message(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
    
    # Save model and tokenizer
    model.save_pretrained(Config.FINAL_MODEL_DIR)
    tokenizer.save_pretrained(Config.FINAL_MODEL_DIR)
    log_message(f"Model saved to: {Config.FINAL_MODEL_DIR}")
    
    # Save training history
    with open(Config.TRAINING_HISTORY, 'w') as f:
        json.dump(training_history, f, indent=2)
    log_message(f"Training history saved to: {Config.TRAINING_HISTORY}")
    
    # Save configuration
    with open(Config.CONFIG_FILE, 'w') as f:
        json.dump(Config.get_config_dict(), f, indent=2)
    log_message(f"Configuration saved to: {Config.CONFIG_FILE}")
    
    # ========================================================================
    # TRAINING SUMMARY
    # ========================================================================
    
    log_message("\n" + "="*70)
    log_message("TRAINING SUMMARY")
    log_message("="*70)
    
    summary = {
        'best_epoch': int(checkpoint['epoch'] + 1),
        'best_val_f1': float(best_val_f1),
        'best_val_acc': float(checkpoint['val_acc']),
        'final_train_loss': float(training_history['train_loss'][-1]),
        'final_train_acc': float(training_history['train_acc'][-1]),
        'total_epochs_trained': len(training_history['train_loss']),
        'early_stopped': early_stopping.early_stop
    }
    
    for key, value in summary.items():
        log_message(f"{key}: {value}")
    
    # Save summary
    summary_path = os.path.join(Config.TRAINING_DIR, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log_message(f"\nSummary saved to: {summary_path}")
    
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message(f"\nTraining completed: {end_time}")
    log_message(f"All artifacts saved to: {Config.ARTIFACTS_DIR}")
    
    return model, tokenizer

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        model, tokenizer = main()
        print("\n" + "="*70)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Model saved to: {Config.FINAL_MODEL_DIR}")
        print(f"Logs and metrics in: {Config.TRAINING_DIR}")
    except Exception as e:
        error_msg = f"\nTraining failed: {str(e)}"
        log_message(error_msg)
        raise