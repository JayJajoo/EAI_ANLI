"""
Configuration file for NLI fine-tuning pipeline
All hyperparameters and paths centralized here
"""

import torch
import os

class Config:
    """Training and model configuration"""
    
    # Model settings
    MODEL_NAME = "prajjwal1/bert-tiny"
    NUM_LABELS = 3
    MAX_LENGTH = 256
    
    # Training hyperparameters
    BATCH_SIZE = 64
    GRADIENT_ACCUMULATION_STEPS = 2
    EPOCHS = 2
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0
    
    # Early stopping
    PATIENCE = 2
    MIN_DELTA = 0.001
    
    # Optimizer
    ADAM_EPSILON = 1e-8
    
    # Regularization
    HIDDEN_DROPOUT_PROB = 0.1
    ATTENTION_DROPOUT_PROB = 0.1
    
    # Paths
    ARTIFACTS_DIR = './artifacts'
    MODEL_DIR = os.path.join(ARTIFACTS_DIR, 'model')
    TRAINING_DIR = os.path.join(ARTIFACTS_DIR, 'training')
    EVAL_DIR = os.path.join(ARTIFACTS_DIR, 'evaluation')
    PIPELINE_DIR = os.path.join(ARTIFACTS_DIR, 'pipeline')
    
    # Output files
    CHECKPOINT_PATH = os.path.join(TRAINING_DIR, 'best_model.pt')
    FINAL_MODEL_DIR = os.path.join(MODEL_DIR, 'final_model')
    TRAINING_LOG = os.path.join(TRAINING_DIR, 'training_log.txt')
    TRAINING_HISTORY = os.path.join(TRAINING_DIR, 'training_history.json')
    CONFIG_FILE = os.path.join(MODEL_DIR, 'config.json')
    
    # Evaluation files
    EVAL_LOG = os.path.join(EVAL_DIR, 'evaluation_log.txt')
    EVAL_RESULTS = os.path.join(EVAL_DIR, 'evaluation_results.json')
    CONFUSION_MATRIX_PLOT = os.path.join(EVAL_DIR, 'confusion_matrix.png')
    
    # Pipeline files
    PIPELINE_LOG = os.path.join(PIPELINE_DIR, 'pipeline_log.txt')
    PIPELINE_SUMMARY = os.path.join(PIPELINE_DIR, 'pipeline_summary.json')
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_FP16 = torch.cuda.is_available()
    
    # Random seed
    SEED = 42
    
    # Dataset
    DATASET_NAME = "facebook/anli"
    TRAIN_SPLIT = "train_r2"
    DEV_SPLIT = "dev_r2"
    TEST_SPLIT = "test_r2"
    
    # Labels
    LABEL_NAMES = ['ENTAILMENT', 'NEUTRAL', 'CONTRADICTION']
    
    # Baseline metrics (to compare against)
    BASELINE_ACCURACY = 0.337
    BASELINE_F1 = 0.242
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        for dir_path in [cls.ARTIFACTS_DIR, cls.MODEL_DIR, cls.TRAINING_DIR, 
                         cls.EVAL_DIR, cls.PIPELINE_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def get_config_dict(cls):
        """Return config as dictionary for saving"""
        return {
            'model_name': cls.MODEL_NAME,
            'num_labels': cls.NUM_LABELS,
            'max_length': cls.MAX_LENGTH,
            'batch_size': cls.BATCH_SIZE,
            'gradient_accumulation_steps': cls.GRADIENT_ACCUMULATION_STEPS,
            'epochs': cls.EPOCHS,
            'learning_rate': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY,
            'warmup_ratio': cls.WARMUP_RATIO,
            'max_grad_norm': cls.MAX_GRAD_NORM,
            'patience': cls.PATIENCE,
            'min_delta': cls.MIN_DELTA,
            'adam_epsilon': cls.ADAM_EPSILON,
            'hidden_dropout_prob': cls.HIDDEN_DROPOUT_PROB,
            'attention_dropout_prob': cls.ATTENTION_DROPOUT_PROB,
            'seed': cls.SEED,
            'device': str(cls.DEVICE),
            'use_fp16': cls.USE_FP16
        }