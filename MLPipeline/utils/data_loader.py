"""
Shared data loading utilities for all models
"""

import pandas as pd
from datasets import load_dataset
from config import Config

def load_anli_data():
    """
    Load ANLI dataset and return as DataFrames
    
    Returns:
        tuple: (train_df, dev_df, test_df)
    """
    print(f"Loading dataset: {Config.DATASET_NAME}")
    ds = load_dataset(Config.DATASET_NAME)
    
    train_data = ds[Config.TRAIN_SPLIT]
    dev_data = ds[Config.DEV_SPLIT]
    test_data = ds[Config.TEST_SPLIT]
    
    print(f"Train: {len(train_data)} examples")
    print(f"Dev: {len(dev_data)} examples")
    print(f"Test: {len(test_data)} examples")
    
    # Convert to DataFrames with preprocessing
    def to_df(dataset):
        return pd.DataFrame({
            'premise': [p.lower().strip() for p in dataset['premise']],
            'hypothesis': [h.lower().strip() for h in dataset['hypothesis']],
            'label': dataset['label']
        })
    
    train_df = to_df(train_data)
    dev_df = to_df(dev_data)
    test_df = to_df(test_data)
    
    return train_df, dev_df, test_df

def get_label_distribution(df, split_name=""):
    """
    Get label distribution for a dataset
    
    Args:
        df: DataFrame with 'label' column
        split_name: Name of the split (for logging)
    
    Returns:
        dict: Label distribution
    """
    dist = df['label'].value_counts().sort_index()
    
    if split_name:
        print(f"\n{split_name} label distribution:")
    
    label_dist = {}
    for label, count in dist.items():
        pct = count / len(df) * 100
        label_name = Config.LABEL_NAMES[label]
        label_dist[label_name] = {
            'count': int(count),
            'percentage': float(pct)
        }
        if split_name:
            print(f"  {label_name}: {count} ({pct:.2f}%)")
    
    return label_dist