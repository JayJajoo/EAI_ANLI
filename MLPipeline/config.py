"""
Configuration file for NLI model training pipeline
Supports multiple models: BERT, Logistic Regression, Random Forest, XGBoost
"""

import torch
import os

class Config:
    """Global configuration for the pipeline"""
    
    # ========================================================================
    # DATASET CONFIGURATION
    # ========================================================================
    DATASET_NAME = "facebook/anli"
    TRAIN_SPLIT = "train_r2"
    DEV_SPLIT = "dev_r2"
    TEST_SPLIT = "test_r2"
    LABEL_NAMES = ['ENTAILMENT', 'NEUTRAL', 'CONTRADICTION']
    NUM_LABELS = 3
    
    # ========================================================================
    # BASELINE METRICS
    # ========================================================================
    BASELINE_ACCURACY = 0.337
    BASELINE_F1 = 0.242
    
    # ========================================================================
    # PATHS
    # ========================================================================
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    ARTIFACTS_DIR = os.path.join(ROOT_DIR, 'artifacts')
    
    # EDA paths
    EDA_DIR = os.path.join(ARTIFACTS_DIR, 'eda')
    
    # Model-specific artifact directories
    BERT_ARTIFACTS = os.path.join(ARTIFACTS_DIR, 'bert')
    LR_ARTIFACTS = os.path.join(ARTIFACTS_DIR, 'logistic_regression')
    RF_ARTIFACTS = os.path.join(ARTIFACTS_DIR, 'random_forest')
    XGB_ARTIFACTS = os.path.join(ARTIFACTS_DIR, 'xgboost')
    
    # Pipeline logs
    PIPELINE_DIR = os.path.join(ARTIFACTS_DIR, 'pipeline')
    PIPELINE_LOG = os.path.join(PIPELINE_DIR, 'pipeline_log.txt')
    PIPELINE_SUMMARY = os.path.join(PIPELINE_DIR, 'pipeline_summary.json')
    
    # ========================================================================
    # DEVICE CONFIGURATION
    # ========================================================================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_FP16 = torch.cuda.is_available()
    SEED = 42
    
    # ========================================================================
    # BERT MODEL CONFIGURATIONS
    # ========================================================================
    BERT_CONFIGS = [
        {
            'name': 'bert-tiny',
            'model_name': 'prajjwal1/bert-tiny',
            'max_length': 256,
            'batch_size': 64,
            'gradient_accumulation_steps': 2,
            'epochs': 2,
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'warmup_ratio': 0.1,
            'max_grad_norm': 1.0,
            'hidden_dropout_prob': 0.1,
            'attention_dropout_prob': 0.1,
            'patience': 2,
            'min_delta': 0.001,
            'adam_epsilon': 1e-8
        },
        # {
        #     'name': 'bert-base-uncased',
        #     'model_name': 'bert-base-uncased',
        #     'max_length': 128,
        #     'batch_size': 32,
        #     'gradient_accumulation_steps': 2,
        #     'epochs': 3,
        #     'learning_rate': 2e-5,
        #     'weight_decay': 0.01,
        #     'warmup_ratio': 0.1,
        #     'max_grad_norm': 1.0,
        #     'hidden_dropout_prob': 0.1,
        #     'attention_dropout_prob': 0.1,
        #     'patience': 2,
        #     'min_delta': 0.001,
        #     'adam_epsilon': 1e-8
        # },
        # {
        #     'name': 'bert-base-uncased',
        #     'model_name': 'roberta-base',
        #     'max_length': 128,
        #     'batch_size': 32,
        #     'gradient_accumulation_steps': 2,
        #     'epochs': 3,
        #     'learning_rate': 1e-5,
        #     'weight_decay': 0.01,
        #     'warmup_ratio': 0.1,
        #     'max_grad_norm': 1.0,
        #     'hidden_dropout_prob': 0.1,
        #     'attention_dropout_prob': 0.1,
        #     'patience': 2,
        #     'min_delta': 0.001,
        #     'adam_epsilon': 1e-8
        # }
    ]
    
    # ========================================================================
    # LOGISTIC REGRESSION CONFIGURATIONS
    # ========================================================================
    LR_CONFIGS = [
        {
            'name': 'lr-default',
            'max_iter': 1000,
            'C': 1.0,
            'solver': 'lbfgs',
            'penalty': 'l2',
            'class_weight': None,
            'tfidf_max_features': 10000,
            'tfidf_ngram_range': (1, 2)
        },
        {
            'name': 'lr-l1-balanced',
            'max_iter': 1000,
            'C': 0.5,
            'solver': 'saga',
            'penalty': 'l1',
            'class_weight': 'balanced',
            'tfidf_max_features': 15000,
            'tfidf_ngram_range': (1, 2)
        },
        {
            'name': 'lr-strong-reg',
            'max_iter': 1000,
            'C': 0.1,
            'solver': 'lbfgs',
            'penalty': 'l2',
            'class_weight': None,
            'tfidf_max_features': 20000,
            'tfidf_ngram_range': (1, 3)
        }
    ]
    
    # ========================================================================
    # RANDOM FOREST CONFIGURATIONS
    # ========================================================================
    RF_CONFIGS = [
        {
            'name': 'rf-default',
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'class_weight': None,
            'tfidf_max_features': 10000,
            'tfidf_ngram_range': (1, 2)
        },
        {
            'name': 'rf-deep',
            'n_estimators': 200,
            'max_depth': 30,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'tfidf_max_features': 15000,
            'tfidf_ngram_range': (1, 2)
        },
        {
            'name': 'rf-shallow',
            'n_estimators': 150,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'log2',
            'class_weight': None,
            'tfidf_max_features': 8000,
            'tfidf_ngram_range': (1, 2)
        }
    ]
    
    # ========================================================================
    # XGBOOST CONFIGURATIONS
    # ========================================================================
    XGB_CONFIGS = [
        {
            'name': 'xgb-default',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'min_child_weight': 1,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'tfidf_max_features': 10000,
            'tfidf_ngram_range': (1, 2)
        },
        # {
        #     'name': 'xgb-deep',
        #     'n_estimators': 200,
        #     'max_depth': 8,
        #     'learning_rate': 0.05,
        #     'subsample': 0.9,
        #     'colsample_bytree': 0.9,
        #     'gamma': 0.1,
        #     'min_child_weight': 1,
        #     'reg_alpha': 0.1,
        #     'reg_lambda': 1,
        #     'tfidf_max_features': 15000,
        #     'tfidf_ngram_range': (1, 2)
        # },
        {
            'name': 'xgb-regularized',
            'n_estimators': 150,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'gamma': 0.2,
            'min_child_weight': 3,
            'reg_alpha': 0.5,
            'reg_lambda': 2,
            'tfidf_max_features': 12000,
            'tfidf_ngram_range': (1, 3)
        }
    ]
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        dirs = [
            cls.ARTIFACTS_DIR,
            cls.EDA_DIR,
            cls.BERT_ARTIFACTS,
            cls.LR_ARTIFACTS,
            cls.RF_ARTIFACTS,
            cls.XGB_ARTIFACTS,
            cls.PIPELINE_DIR
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def get_model_artifact_dir(cls, model_type, config_name):
        """Get artifact directory for specific model and config"""
        base_dirs = {
            'bert': cls.BERT_ARTIFACTS,
            'logistic_regression': cls.LR_ARTIFACTS,
            'random_forest': cls.RF_ARTIFACTS,
            'xgboost': cls.XGB_ARTIFACTS
        }
        model_dir = os.path.join(base_dirs[model_type], config_name)
        os.makedirs(model_dir, exist_ok=True)
        return model_dir