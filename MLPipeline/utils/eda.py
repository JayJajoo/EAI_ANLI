"""
Comprehensive Exploratory Data Analysis for ANLI Dataset
"""

import os
import json
import re
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from utils.data_loader import load_anli_data
from config import Config

def log_message(message, log_file):
    """Log to console and file"""
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def run_eda():
    """
    Run complete EDA and save results
    """
    os.makedirs(Config.EDA_DIR, exist_ok=True)
    log_file = os.path.join(Config.EDA_DIR, 'eda_log.txt')
    
    log_message("="*70, log_file)
    log_message("COMPREHENSIVE EDA PIPELINE", log_file)
    log_message("="*70, log_file)
    log_message(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    
    # Load data
    log_message("\nLoading ANLI dataset...", log_file)
    train_df, dev_df, test_df = load_anli_data()
    
    log_message(f"Train size: {len(train_df)}", log_file)
    log_message(f"Dev size: {len(dev_df)}", log_file)
    log_message(f"Test size: {len(test_df)}", log_file)
    
    # ========================================================================
    # 1. LABEL DISTRIBUTION
    # ========================================================================
    log_message("\n" + "="*70, log_file)
    log_message("LABEL DISTRIBUTION ANALYSIS", log_file)
    log_message("="*70, log_file)
    
    label_dist = analyze_label_distribution(train_df, dev_df, test_df, log_file)
    
    # ========================================================================
    # 2. TEXT LENGTH ANALYSIS
    # ========================================================================
    log_message("\n" + "="*70, log_file)
    log_message("TEXT LENGTH ANALYSIS", log_file)
    log_message("="*70, log_file)
    
    length_stats = analyze_text_lengths(train_df, dev_df, test_df, log_file)
    
    # ========================================================================
    # 3. VOCABULARY ANALYSIS
    # ========================================================================
    log_message("\n" + "="*70, log_file)
    log_message("VOCABULARY ANALYSIS", log_file)
    log_message("="*70, log_file)
    
    vocab_stats = analyze_vocabulary(train_df, log_file)
    
    # ========================================================================
    # 4. PREMISE-HYPOTHESIS OVERLAP
    # ========================================================================
    log_message("\n" + "="*70, log_file)
    log_message("PREMISE-HYPOTHESIS OVERLAP ANALYSIS", log_file)
    log_message("="*70, log_file)
    
    overlap_stats = analyze_text_overlap(train_df, log_file)
    
    # ========================================================================
    # 5. TF-IDF COSINE SIMILARITY
    # ========================================================================
    log_message("\n" + "="*70, log_file)
    log_message("TF-IDF COSINE SIMILARITY ANALYSIS", log_file)
    log_message("="*70, log_file)
    
    tfidf_stats = analyze_tfidf_similarity(train_df, log_file)
    
    # ========================================================================
    # 6. DATA QUALITY CHECKS
    # ========================================================================
    log_message("\n" + "="*70, log_file)
    log_message("DATA QUALITY CHECKS", log_file)
    log_message("="*70, log_file)
    
    quality_stats = analyze_data_quality(train_df, log_file)
    
    # ========================================================================
    # 7. GENERATE SUMMARY
    # ========================================================================
    summary = generate_eda_summary(
        train_df, dev_df, test_df,
        label_dist, length_stats, vocab_stats,
        overlap_stats, tfidf_stats, quality_stats,
        log_file
    )
    
    log_message("\n" + "="*70, log_file)
    log_message("EDA PIPELINE COMPLETED SUCCESSFULLY", log_file)
    log_message("="*70, log_file)
    log_message(f"All artifacts saved to: {Config.EDA_DIR}", log_file)
    
    return summary

def analyze_label_distribution(train_df, dev_df, test_df, log_file):
    """Analyze and visualize label distribution"""
    
    # Training set distribution
    train_dist = train_df['label'].value_counts(normalize=True).sort_index()
    
    log_message("\nTraining set label distribution:", log_file)
    for label, prop in train_dist.items():
        label_name = Config.LABEL_NAMES[label]
        count = (train_df['label'] == label).sum()
        log_message(f"  {label_name}: {count} ({prop*100:.2f}%)", log_file)
    
    # Plot distribution across splits
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (df, split_name) in enumerate([(train_df, 'Train'), 
                                              (dev_df, 'Dev'), 
                                              (test_df, 'Test')]):
        label_counts = df['label'].value_counts().sort_index()
        label_names = [Config.LABEL_NAMES[i] for i in label_counts.index]
        
        axes[idx].bar(label_names, label_counts.values, color='steelblue', edgecolor='black')
        axes[idx].set_title(f'{split_name} Set', fontsize=14)
        axes[idx].set_ylabel('Count' if idx == 0 else '', fontsize=12)
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].set_xticklabels(label_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.EDA_DIR, 'label_distributions.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    log_message("\nSaved: label_distributions.png", log_file)
    
    # Save distribution as JSON
    label_dist_dict = {
        'train': {Config.LABEL_NAMES[k]: float(v) for k, v in train_dist.items()},
        'dev': {Config.LABEL_NAMES[k]: float(v) for k, v in dev_df['label'].value_counts(normalize=True).sort_index().items()},
        'test': {Config.LABEL_NAMES[k]: float(v) for k, v in test_df['label'].value_counts(normalize=True).sort_index().items()}
    }
    
    with open(os.path.join(Config.EDA_DIR, 'label_distribution.json'), 'w') as f:
        json.dump(label_dist_dict, f, indent=2)
    
    return label_dist_dict

def analyze_text_lengths(train_df, dev_df, test_df, log_file):
    """Analyze text length statistics"""
    
    # Calculate lengths
    for df in [train_df, dev_df, test_df]:
        df['premise_len'] = df['premise'].str.split().apply(len)
        df['hypothesis_len'] = df['hypothesis'].str.split().apply(len)
    
    log_message(f"\nAverage premise length: {train_df['premise_len'].mean():.2f} words", log_file)
    log_message(f"Average hypothesis length: {train_df['hypothesis_len'].mean():.2f} words", log_file)
    
    # Statistics
    length_stats = train_df[['premise_len', 'hypothesis_len']].describe()
    log_message("\nLength statistics:", log_file)
    log_message(str(length_stats), log_file)
    
    # Save statistics
    length_stats.to_csv(os.path.join(Config.EDA_DIR, 'length_statistics.csv'))
    log_message("\nSaved: length_statistics.csv", log_file)
    
    # Plot distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    splits = [(train_df, 'Train'), (dev_df, 'Dev'), (test_df, 'Test')]
    
    for idx, (df, split_name) in enumerate(splits):
        # Premise lengths
        axes[0, idx].hist(df['premise_len'], bins=30, color='coral', edgecolor='black', alpha=0.7)
        axes[0, idx].set_title(f'{split_name} - Premise Length', fontsize=12)
        axes[0, idx].set_xlabel('Word Count')
        axes[0, idx].set_ylabel('Frequency' if idx == 0 else '')
        axes[0, idx].axvline(df['premise_len'].mean(), color='red', linestyle='--', 
                            label=f'Mean: {df["premise_len"].mean():.1f}')
        axes[0, idx].legend()
        axes[0, idx].grid(alpha=0.3)
        
        # Hypothesis lengths
        axes[1, idx].hist(df['hypothesis_len'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1, idx].set_title(f'{split_name} - Hypothesis Length', fontsize=12)
        axes[1, idx].set_xlabel('Word Count')
        axes[1, idx].set_ylabel('Frequency' if idx == 0 else '')
        axes[1, idx].axvline(df['hypothesis_len'].mean(), color='red', linestyle='--',
                            label=f'Mean: {df["hypothesis_len"].mean():.1f}')
        axes[1, idx].legend()
        axes[1, idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.EDA_DIR, 'text_length_distributions.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    log_message("Saved: text_length_distributions.png", log_file)
    
    return length_stats.to_dict()

def analyze_vocabulary(train_df, log_file):
    """Analyze vocabulary and top words"""
    
    def unique_word_count(texts):
        return len(set(" ".join(texts).split()))
    
    unique_premise = unique_word_count(train_df['premise'])
    unique_hypothesis = unique_word_count(train_df['hypothesis'])
    
    log_message(f"\nUnique words in premises: {unique_premise:,}", log_file)
    log_message(f"Unique words in hypotheses: {unique_hypothesis:,}", log_file)
    
    # Top words per label
    def top_words_for_label(label):
        subset = train_df[train_df['label'] == label]
        words = " ".join(subset['premise'] + " " + subset['hypothesis'])
        words = re.findall(r'\b\w+\b', words.lower())
        return Counter(words).most_common(15)
    
    top_words_dict = {}
    
    for label in sorted(train_df['label'].unique()):
        label_name = Config.LABEL_NAMES[label]
        top_words = top_words_for_label(label)
        top_words_dict[label_name] = [{'word': w, 'count': c} for w, c in top_words]
        
        log_message(f"\nTop 15 words for {label_name.upper()}:", log_file)
        for word, count in top_words[:15]:
            log_message(f"  {word}: {count}", log_file)
    
    # Save top words
    with open(os.path.join(Config.EDA_DIR, 'top_words_by_label.json'), 'w') as f:
        json.dump(top_words_dict, f, indent=2)
    
    log_message("\nSaved: top_words_by_label.json", log_file)
    
    return {
        'unique_premise_words': unique_premise,
        'unique_hypothesis_words': unique_hypothesis,
        'top_words_by_label': top_words_dict
    }

def analyze_text_overlap(train_df, log_file):
    """Analyze premise-hypothesis overlap using Jaccard similarity"""
    
    def jaccard_similarity(a, b):
        a_set, b_set = set(a.split()), set(b.split())
        return len(a_set & b_set) / len(a_set | b_set) if a_set | b_set else 0
    
    train_df['overlap'] = [
        jaccard_similarity(p, h) 
        for p, h in zip(train_df['premise'], train_df['hypothesis'])
    ]
    
    # Overall statistics
    log_message(f"\nOverall Jaccard similarity:", log_file)
    log_message(f"  Mean: {train_df['overlap'].mean():.4f}", log_file)
    log_message(f"  Median: {train_df['overlap'].median():.4f}", log_file)
    log_message(f"  Std: {train_df['overlap'].std():.4f}", log_file)
    
    # Statistics by label
    label_map = {i: Config.LABEL_NAMES[i] for i in range(Config.NUM_LABELS)}
    
    similarity_stats = {}
    log_message("\nJaccard Similarity by Label:", log_file)
    
    for label in sorted(train_df['label'].unique()):
        label_name = label_map[label]
        label_data = train_df[train_df['label'] == label]['overlap']
        
        stats = {
            'mean': float(label_data.mean()),
            'median': float(label_data.median()),
            'std': float(label_data.std()),
            '25th_percentile': float(label_data.quantile(0.25)),
            '75th_percentile': float(label_data.quantile(0.75)),
            '95th_percentile': float(label_data.quantile(0.95))
        }
        
        similarity_stats[label_name] = stats
        
        log_message(f"\n{label_name.upper()}:", log_file)
        log_message(f"  Mean: {stats['mean']:.4f}", log_file)
        log_message(f"  Median: {stats['median']:.4f}", log_file)
        log_message(f"  75th percentile: {stats['75th_percentile']:.4f}", log_file)
        log_message(f"  95th percentile: {stats['95th_percentile']:.4f}", log_file)
    
    # Plot overall distribution
    plt.figure(figsize=(10, 6))
    plt.hist(train_df['overlap'], bins=60, color='purple', edgecolor='black')
    plt.title("Premise-Hypothesis Word Overlap (Jaccard Similarity)", fontsize=14)
    plt.xlabel("Jaccard Similarity")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(Config.EDA_DIR, 'overall_overlap_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot by label
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, label in enumerate(sorted(train_df['label'].unique())):
        label_name = label_map[label]
        label_data = train_df[train_df['label'] == label]['overlap']
        
        axes[idx].hist(label_data, bins=40, color='steelblue', edgecolor='black')
        axes[idx].axvline(label_data.mean(), color='red', linestyle='--', linewidth=2, 
                         label=f'Mean: {label_data.mean():.3f}')
        axes[idx].axvline(label_data.median(), color='green', linestyle='--', linewidth=2, 
                         label=f'Median: {label_data.median():.3f}')
        axes[idx].set_title(f"{label_name.capitalize()} - Overlap Distribution")
        axes[idx].set_xlabel("Jaccard Similarity")
        axes[idx].set_ylabel("Frequency")
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.EDA_DIR, 'overlap_by_label.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Boxplot comparison
    train_df['label_name'] = train_df['label'].map(label_map)
    plt.figure(figsize=(10, 6))
    plt.boxplot([train_df[train_df['label'] == i]['overlap'] for i in range(Config.NUM_LABELS)],
                labels=[Config.LABEL_NAMES[i] for i in range(Config.NUM_LABELS)])
    plt.title("Premise-Hypothesis Overlap by Label", fontsize=14)
    plt.xlabel("Label")
    plt.ylabel("Jaccard Similarity")
    plt.savefig(os.path.join(Config.EDA_DIR, 'overlap_boxplot.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    log_message("\nSaved: overall_overlap_distribution.png", log_file)
    log_message("Saved: overlap_by_label.png", log_file)
    log_message("Saved: overlap_boxplot.png", log_file)
    
    # Save statistics
    with open(os.path.join(Config.EDA_DIR, 'similarity_stats_by_label.json'), 'w') as f:
        json.dump(similarity_stats, f, indent=2)
    
    return similarity_stats

def analyze_tfidf_similarity(train_df, log_file):
    """Analyze TF-IDF cosine similarity between premise and hypothesis"""
    
    max_features_list = [5000, 10000, 15000, 20000]
    fig, axes = plt.subplots(1, len(max_features_list), figsize=(20, 4))
    
    tfidf_results = {}
    
    for idx, max_f in enumerate(max_features_list):
        log_message(f"\nComputing TF-IDF with max_features={max_f}...", log_file)
        
        vectorizer = TfidfVectorizer(max_features=max_f)
        X = vectorizer.fit_transform(train_df['premise'])
        Y = vectorizer.transform(train_df['hypothesis'])
        
        similarities = [cosine_similarity(X[i], Y[i])[0][0] for i in range(len(train_df))]
        
        tfidf_results[f'max_features_{max_f}'] = {
            'mean': float(np.mean(similarities)),
            'median': float(np.median(similarities)),
            'std': float(np.std(similarities))
        }
        
        log_message(f"  Mean similarity: {np.mean(similarities):.4f}", log_file)
        log_message(f"  Median similarity: {np.median(similarities):.4f}", log_file)
        
        # Plot
        axes[idx].hist(similarities, bins=40, color='steelblue', edgecolor='black')
        axes[idx].set_title(f"max_features = {max_f:,}")
        axes[idx].set_xlabel("Cosine Similarity")
        axes[idx].set_ylabel("Frequency")
    
    plt.suptitle("TF-IDF Cosine Similarity: Premise vs Hypothesis", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.EDA_DIR, 'tfidf_similarity_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    log_message("\nSaved: tfidf_similarity_comparison.png", log_file)
    
    # Save results
    with open(os.path.join(Config.EDA_DIR, 'tfidf_similarity_stats.json'), 'w') as f:
        json.dump(tfidf_results, f, indent=2)
    
    log_message("Saved: tfidf_similarity_stats.json", log_file)
    
    return tfidf_results

def analyze_data_quality(train_df, log_file):
    """Check for duplicates and missing values"""
    
    # Duplicates
    duplicates = train_df.duplicated(subset=['premise', 'hypothesis', 'label'])
    duplicate_count = duplicates.sum()
    
    log_message(f"\nDuplicate samples: {duplicate_count}", log_file)
    log_message(f"Percentage: {duplicate_count/len(train_df)*100:.4f}%", log_file)
    
    # Missing values
    missing = train_df.isna().sum()
    log_message("\nMissing values per column:", log_file)
    log_message(str(missing), log_file)
    
    return {
        'duplicate_count': int(duplicate_count),
        'duplicate_percentage': float(duplicate_count/len(train_df)*100),
        'missing_values': missing.to_dict()
    }

def generate_eda_summary(train_df, dev_df, test_df, label_dist, length_stats,
                        vocab_stats, overlap_stats, tfidf_stats, quality_stats, log_file):
    """Generate and save comprehensive EDA summary"""
    
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': {
            'name': Config.DATASET_NAME,
            'train_size': len(train_df),
            'dev_size': len(dev_df),
            'test_size': len(test_df),
            'num_classes': Config.NUM_LABELS,
            'class_names': Config.LABEL_NAMES
        },
        'label_distribution': label_dist,
        'text_lengths': {
            'avg_premise_length': float(train_df['premise_len'].mean()),
            'avg_hypothesis_length': float(train_df['hypothesis_len'].mean()),
            'avg_combined_length': float(train_df['premise_len'].mean() + train_df['hypothesis_len'].mean())
        },
        'vocabulary': vocab_stats,
        'text_overlap': overlap_stats,
        'tfidf_similarity': tfidf_stats,
        'data_quality': quality_stats
    }
    
    # Save summary
    summary_path = os.path.join(Config.EDA_DIR, 'eda_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    log_message("\n" + "="*70, log_file)
    log_message("EDA SUMMARY", log_file)
    log_message("="*70, log_file)
    log_message(json.dumps(summary, indent=2), log_file)
    log_message(f"\nSaved: eda_summary.json", log_file)
    
    # List all generated files
    log_message("\nGenerated files:", log_file)
    for file in sorted(os.listdir(Config.EDA_DIR)):
        log_message(f"  - {file}", log_file)
    
    return summary


# """
# Exploratory Data Analysis utilities
# """

# import os
# import json
# import pandas as pd
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# from utils.data_loader import load_anli_data, get_label_distribution
# from config import Config

# def run_eda():
#     """
#     Run complete EDA and save results
#     """
#     print("="*70)
#     print("RUNNING EXPLORATORY DATA ANALYSIS")
#     print("="*70)
    
#     # Create EDA directory
#     os.makedirs(Config.EDA_DIR, exist_ok=True)
    
#     # Load data
#     train_df, dev_df, test_df = load_anli_data()
    
#     # Get distributions
#     train_dist = get_label_distribution(train_df, "Train")
#     dev_dist = get_label_distribution(dev_df, "Dev")
#     test_dist = get_label_distribution(test_df, "Test")
    
#     # Calculate text statistics
#     stats = calculate_text_statistics(train_df, dev_df, test_df)
    
#     # Generate visualizations
#     plot_label_distributions(train_df, dev_df, test_df)
#     plot_text_length_distributions(train_df, dev_df, test_df)
    
#     # Save EDA results
#     eda_results = {
#         'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#         'dataset': Config.DATASET_NAME,
#         'splits': {
#             'train': {'size': len(train_df), 'distribution': train_dist},
#             'dev': {'size': len(dev_df), 'distribution': dev_dist},
#             'test': {'size': len(test_df), 'distribution': test_dist}
#         },
#         'statistics': stats
#     }
    
#     results_path = os.path.join(Config.EDA_DIR, 'eda_results.json')
#     with open(results_path, 'w') as f:
#         json.dump(eda_results, f, indent=2)
    
#     print(f"\nEDA results saved to: {Config.EDA_DIR}")
#     print("="*70)
    
#     return eda_results

# def calculate_text_statistics(train_df, dev_df, test_df):
#     """Calculate text length statistics"""
    
#     def get_stats(df, split_name):
#         premise_lens = df['premise'].str.split().str.len()
#         hyp_lens = df['hypothesis'].str.split().str.len()
        
#         return {
#             'premise': {
#                 'mean': float(premise_lens.mean()),
#                 'median': float(premise_lens.median()),
#                 'min': int(premise_lens.min()),
#                 'max': int(premise_lens.max()),
#                 'std': float(premise_lens.std())
#             },
#             'hypothesis': {
#                 'mean': float(hyp_lens.mean()),
#                 'median': float(hyp_lens.median()),
#                 'min': int(hyp_lens.min()),
#                 'max': int(hyp_lens.max()),
#                 'std': float(hyp_lens.std())
#             }
#         }
    
#     stats = {
#         'train': get_stats(train_df, 'Train'),
#         'dev': get_stats(dev_df, 'Dev'),
#         'test': get_stats(test_df, 'Test')
#     }
    
#     print("\nText Length Statistics (in words):")
#     for split, split_stats in stats.items():
#         print(f"\n{split.capitalize()}:")
#         print(f"  Premise - Mean: {split_stats['premise']['mean']:.1f}, "
#               f"Median: {split_stats['premise']['median']:.1f}, "
#               f"Range: [{split_stats['premise']['min']}, {split_stats['premise']['max']}]")
#         print(f"  Hypothesis - Mean: {split_stats['hypothesis']['mean']:.1f}, "
#               f"Median: {split_stats['hypothesis']['median']:.1f}, "
#               f"Range: [{split_stats['hypothesis']['min']}, {split_stats['hypothesis']['max']}]")
    
#     return stats

# def plot_label_distributions(train_df, dev_df, test_df):
#     """Plot label distributions across splits"""
    
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
#     for idx, (df, split_name) in enumerate([(train_df, 'Train'), 
#                                               (dev_df, 'Dev'), 
#                                               (test_df, 'Test')]):
#         label_counts = df['label'].value_counts().sort_index()
#         label_names = [Config.LABEL_NAMES[i] for i in label_counts.index]
        
#         axes[idx].bar(label_names, label_counts.values, color='steelblue', edgecolor='black')
#         axes[idx].set_title(f'{split_name} Set', fontsize=14)
#         axes[idx].set_ylabel('Count' if idx == 0 else '', fontsize=12)
#         axes[idx].grid(axis='y', alpha=0.3)
#         axes[idx].set_xticklabels(label_names, rotation=45, ha='right')
    
#     plt.tight_layout()
#     plot_path = os.path.join(Config.EDA_DIR, 'label_distributions.png')
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"Label distribution plot saved: {plot_path}")

# def plot_text_length_distributions(train_df, dev_df, test_df):
#     """Plot text length distributions"""
    
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
#     splits = [(train_df, 'Train'), (dev_df, 'Dev'), (test_df, 'Test')]
    
#     for idx, (df, split_name) in enumerate(splits):
#         premise_lens = df['premise'].str.split().str.len()
#         hyp_lens = df['hypothesis'].str.split().str.len()
        
#         # Premise lengths
#         axes[0, idx].hist(premise_lens, bins=30, color='coral', edgecolor='black', alpha=0.7)
#         axes[0, idx].set_title(f'{split_name} - Premise Length', fontsize=12)
#         axes[0, idx].set_xlabel('Word Count')
#         axes[0, idx].set_ylabel('Frequency' if idx == 0 else '')
#         axes[0, idx].axvline(premise_lens.mean(), color='red', linestyle='--', 
#                             label=f'Mean: {premise_lens.mean():.1f}')
#         axes[0, idx].legend()
#         axes[0, idx].grid(alpha=0.3)
        
#         # Hypothesis lengths
#         axes[1, idx].hist(hyp_lens, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
#         axes[1, idx].set_title(f'{split_name} - Hypothesis Length', fontsize=12)
#         axes[1, idx].set_xlabel('Word Count')
#         axes[1, idx].set_ylabel('Frequency' if idx == 0 else '')
#         axes[1, idx].axvline(hyp_lens.mean(), color='red', linestyle='--',
#                             label=f'Mean: {hyp_lens.mean():.1f}')
#         axes[1, idx].legend()
#         axes[1, idx].grid(alpha=0.3)
    
#     plt.tight_layout()
#     plot_path = os.path.join(Config.EDA_DIR, 'text_length_distributions.png')
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"Text length distribution plot saved: {plot_path}")