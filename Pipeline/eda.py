"""
EDA Pipeline for ANLI Dataset
Performs exploratory data analysis and saves all artifacts
"""

import os
import json
from collections import Counter
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Docker
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset

# Configuration
ARTIFACTS_DIR = './artifacts/eda'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Logging function
def log_and_save(message, filename=None):
    """Log message and optionally save to file"""
    print(message)
    if filename:
        filepath = os.path.join(ARTIFACTS_DIR, filename)
        with open(filepath, 'a') as f:
            f.write(message + '\n')

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
log_and_save("\n" + "="*70, "eda_log.txt")
log_and_save("STARTING EDA PIPELINE", "eda_log.txt")
log_and_save("="*70 + "\n", "eda_log.txt")

log_and_save("Loading ANLI dataset...", "eda_log.txt")
ds = load_dataset("facebook/anli")

train = ds['train_r2']
dev = ds['dev_r2']
test = ds['test_r2']

# Preprocess: lowercase
train = train.map(lambda x: {
    'premise': x['premise'].lower(),
    'hypothesis': x['hypothesis'].lower()
})

log_and_save(f"Dataset loaded successfully", "eda_log.txt")
log_and_save(f"Train size: {len(train)}", "eda_log.txt")
log_and_save(f"Dev size: {len(dev)}", "eda_log.txt")
log_and_save(f"Test size: {len(test)}\n", "eda_log.txt")

# ============================================================================
# 2. DATASET OVERVIEW
# ============================================================================
log_and_save("="*70, "eda_log.txt")
log_and_save("DATASET OVERVIEW", "eda_log.txt")
log_and_save("="*70 + "\n", "eda_log.txt")

# Features
log_and_save("Features:", "eda_log.txt")
log_and_save(str(train.features) + "\n", "eda_log.txt")

# Shape
log_and_save("Dataset Shapes:", "eda_log.txt")
for split_name, split_data in {'Train': train, 'Validation': dev, 'Test': test}.items():
    log_and_save(f"{split_name}: {split_data.shape}", "eda_log.txt")

# Label names
label_names = train.features['label'].names
log_and_save(f"\nLabel names: {label_names}\n", "eda_log.txt")

# ============================================================================
# 3. LABEL DISTRIBUTION
# ============================================================================
log_and_save("="*70, "eda_log.txt")
log_and_save("LABEL DISTRIBUTION", "eda_log.txt")
log_and_save("="*70 + "\n", "eda_log.txt")

df_train = train.to_pandas()
df_dev = dev.to_pandas()
df_test = test.to_pandas()

label_dist = df_train['label'].value_counts(normalize=True).sort_index()
log_and_save("Training set label distribution (normalized):", "eda_log.txt")
for label, prop in label_dist.items():
    log_and_save(f"  Label {label} ({label_names[label]}): {prop:.4f}", "eda_log.txt")

# Save distribution as JSON
label_dist_dict = {label_names[k]: float(v) for k, v in label_dist.items()}
with open(os.path.join(ARTIFACTS_DIR, 'label_distribution.json'), 'w') as f:
    json.dump(label_dist_dict, f, indent=2)
log_and_save("\nSaved: label_distribution.json\n", "eda_log.txt")

# ============================================================================
# 4. TEXT LENGTH ANALYSIS
# ============================================================================
log_and_save("="*70, "eda_log.txt")
log_and_save("TEXT LENGTH ANALYSIS", "eda_log.txt")
log_and_save("="*70 + "\n", "eda_log.txt")

# Add length columns
df_train['premise_len'] = df_train['premise'].str.split().apply(len)
df_train['hypothesis_len'] = df_train['hypothesis'].str.split().apply(len)

log_and_save(f"Average premise length: {df_train['premise_len'].mean():.2f} words", "eda_log.txt")
log_and_save(f"Average hypothesis length: {df_train['hypothesis_len'].mean():.2f} words", "eda_log.txt")

# Statistics
length_stats = df_train[['premise_len', 'hypothesis_len']].describe()
log_and_save("\nLength statistics:", "eda_log.txt")
log_and_save(str(length_stats), "eda_log.txt")

# Save statistics
length_stats.to_csv(os.path.join(ARTIFACTS_DIR, 'length_statistics.csv'))
log_and_save("\nSaved: length_statistics.csv\n", "eda_log.txt")

# Plot premise length distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_train['premise_len'], bins=50, color='steelblue', edgecolor='black')
plt.title("Premise Length Distribution", fontsize=14)
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.savefig(os.path.join(ARTIFACTS_DIR, 'premise_length_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
log_and_save("Saved: premise_length_distribution.png", "eda_log.txt")

# Plot hypothesis length distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_train['hypothesis_len'], bins=50, color='coral', edgecolor='black')
plt.title("Hypothesis Length Distribution", fontsize=14)
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.savefig(os.path.join(ARTIFACTS_DIR, 'hypothesis_length_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
log_and_save("Saved: hypothesis_length_distribution.png\n", "eda_log.txt")

# ============================================================================
# 5. VOCABULARY ANALYSIS
# ============================================================================
log_and_save("="*70, "eda_log.txt")
log_and_save("VOCABULARY ANALYSIS", "eda_log.txt")
log_and_save("="*70 + "\n", "eda_log.txt")

def unique_word_count(texts):
    return len(set(" ".join(texts).split()))

unique_premise = unique_word_count(df_train['premise'])
unique_hypothesis = unique_word_count(df_train['hypothesis'])

log_and_save(f"Unique words in premises: {unique_premise:,}", "eda_log.txt")
log_and_save(f"Unique words in hypotheses: {unique_hypothesis:,}\n", "eda_log.txt")

# Top words per label
def top_words_for_label(label):
    subset = df_train[df_train['label'] == label]
    words = " ".join(subset['premise'] + " " + subset['hypothesis'])
    words = re.findall(r'\b\w+\b', words.lower())
    return Counter(words).most_common(15)

top_words_dict = {}
label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

for lbl in sorted(df_train['label'].unique()):
    label_name = label_map[lbl]
    top_words = top_words_for_label(lbl)
    top_words_dict[label_name] = [{'word': w, 'count': c} for w, c in top_words]
    
    log_and_save(f"\nTop 15 words for {label_name.upper()}:", "eda_log.txt")
    for word, count in top_words[:15]:
        log_and_save(f"  {word}: {count}", "eda_log.txt")

# Save top words
with open(os.path.join(ARTIFACTS_DIR, 'top_words_by_label.json'), 'w') as f:
    json.dump(top_words_dict, f, indent=2)
log_and_save("\nSaved: top_words_by_label.json\n", "eda_log.txt")

# ============================================================================
# 6. PREMISE-HYPOTHESIS OVERLAP
# ============================================================================
log_and_save("="*70, "eda_log.txt")
log_and_save("PREMISE-HYPOTHESIS OVERLAP ANALYSIS", "eda_log.txt")
log_and_save("="*70 + "\n", "eda_log.txt")

# Jaccard similarity
def jaccard_similarity(a, b):
    a_set, b_set = set(a.split()), set(b.split())
    return len(a_set & b_set) / len(a_set | b_set) if a_set | b_set else 0

df_train['overlap'] = [
    jaccard_similarity(p, h) 
    for p, h in zip(df_train['premise'], df_train['hypothesis'])
]

# Overall overlap distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(df_train['overlap'], bins=60, color='purple', edgecolor='black')
plt.title("Premise-Hypothesis Word Overlap (Jaccard Similarity)", fontsize=14)
plt.xlabel("Jaccard Similarity")
plt.ylabel("Frequency")
plt.savefig(os.path.join(ARTIFACTS_DIR, 'overall_overlap_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
log_and_save("Saved: overall_overlap_distribution.png", "eda_log.txt")

# Similarity statistics by label
similarity_stats = df_train.groupby('label')['overlap'].agg([
    ('mean', 'mean'),
    ('median', 'median'),
    ('25th_percentile', lambda x: x.quantile(0.25)),
    ('50th_percentile', lambda x: x.quantile(0.50)),
    ('75th_percentile', lambda x: x.quantile(0.75)),
    ('90th_percentile', lambda x: x.quantile(0.90)),
    ('95th_percentile', lambda x: x.quantile(0.95))
])

log_and_save("\nJaccard Similarity Statistics by Label:", "eda_log.txt")
log_and_save("="*70, "eda_log.txt")
log_and_save(str(similarity_stats), "eda_log.txt")

# Save statistics
similarity_stats.to_csv(os.path.join(ARTIFACTS_DIR, 'similarity_stats_by_label.csv'))
log_and_save("\nSaved: similarity_stats_by_label.csv", "eda_log.txt")

# Detailed stats per label
for label in sorted(df_train['label'].unique()):
    label_name = label_map[label]
    stats = similarity_stats.loc[label]
    log_and_save(f"\n{label_name.upper()} (Label {label}):", "eda_log.txt")
    log_and_save(f"  Mean:            {stats['mean']:.4f}", "eda_log.txt")
    log_and_save(f"  Median:          {stats['median']:.4f}", "eda_log.txt")
    log_and_save(f"  75th percentile: {stats['75th_percentile']:.4f}", "eda_log.txt")
    log_and_save(f"  95th percentile: {stats['95th_percentile']:.4f}", "eda_log.txt")

# Plot overlap by label (3 subplots)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, label in enumerate(sorted(df_train['label'].unique())):
    label_name = label_map[label]
    label_data = df_train[df_train['label'] == label]['overlap']
    
    sns.histplot(label_data, bins=40, ax=axes[idx], color='steelblue', edgecolor='black')
    axes[idx].axvline(label_data.mean(), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {label_data.mean():.3f}')
    axes[idx].axvline(label_data.median(), color='green', linestyle='--', linewidth=2, 
                      label=f'Median: {label_data.median():.3f}')
    axes[idx].set_title(f"{label_name.capitalize()} - Overlap Distribution")
    axes[idx].set_xlabel("Jaccard Similarity")
    axes[idx].set_ylabel("Frequency")
    axes[idx].legend()

plt.tight_layout()
plt.savefig(os.path.join(ARTIFACTS_DIR, 'overlap_by_label.png'), dpi=300, bbox_inches='tight')
plt.close()
log_and_save("Saved: overlap_by_label.png\n", "eda_log.txt")

# Boxplot comparison
plt.figure(figsize=(10, 6))
df_train['label_name'] = df_train['label'].map(label_map)
sns.boxplot(data=df_train, x='label_name', y='overlap')
plt.title("Premise-Hypothesis Overlap by Label", fontsize=14)
plt.xlabel("Label")
plt.ylabel("Jaccard Similarity")
plt.savefig(os.path.join(ARTIFACTS_DIR, 'overlap_boxplot.png'), dpi=300, bbox_inches='tight')
plt.close()
log_and_save("Saved: overlap_boxplot.png\n", "eda_log.txt")

# ============================================================================
# 7. TF-IDF COSINE SIMILARITY
# ============================================================================
log_and_save("="*70, "eda_log.txt")
log_and_save("TF-IDF COSINE SIMILARITY ANALYSIS", "eda_log.txt")
log_and_save("="*70 + "\n", "eda_log.txt")

max_features_list = [5000, 15000, 20000, 25000]
fig, axes = plt.subplots(1, len(max_features_list), figsize=(20, 4))

tfidf_results = {}
for idx, max_f in enumerate(max_features_list):
    log_and_save(f"Computing TF-IDF with max_features={max_f}...", "eda_log.txt")
    
    vectorizer = TfidfVectorizer(max_features=max_f)
    X = vectorizer.fit_transform(df_train['premise'])
    Y = vectorizer.transform(df_train['hypothesis'])
    
    similarities = [cosine_similarity(X[i], Y[i])[0][0] for i in range(len(df_train))]
    
    # Store results
    tfidf_results[f'max_features_{max_f}'] = {
        'mean': float(np.mean(similarities)),
        'median': float(np.median(similarities)),
        'std': float(np.std(similarities))
    }
    
    # Plot
    sns.histplot(similarities, bins=40, ax=axes[idx], color='steelblue', edgecolor='black')
    axes[idx].set_title(f"max_features = {max_f:,}")
    axes[idx].set_xlabel("Cosine Similarity")
    axes[idx].set_ylabel("Frequency")

plt.suptitle("TF-IDF Cosine Similarity: Premise vs Hypothesis", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACTS_DIR, 'tfidf_similarity_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
log_and_save("Saved: tfidf_similarity_comparison.png", "eda_log.txt")

# Save TF-IDF results
with open(os.path.join(ARTIFACTS_DIR, 'tfidf_similarity_stats.json'), 'w') as f:
    json.dump(tfidf_results, f, indent=2)
log_and_save("Saved: tfidf_similarity_stats.json\n", "eda_log.txt")

# ============================================================================
# 8. REASON COLUMN ANALYSIS
# ============================================================================
log_and_save("="*70, "eda_log.txt")
log_and_save("REASON COLUMN ANALYSIS", "eda_log.txt")
log_and_save("="*70 + "\n", "eda_log.txt")

df_train['reason_len'] = df_train['reason'].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
df_train['reason'].fillna('None', inplace=True)

missing_reasons = (df_train['reason'] == 'None').sum()
log_and_save(f"Missing reasons: {missing_reasons} ({missing_reasons/len(df_train)*100:.2f}%)", "eda_log.txt")
log_and_save(f"Average reason length: {df_train['reason_len'].mean():.2f} words\n", "eda_log.txt")

# Most common reasons
log_and_save("Top 10 most common reasons:", "eda_log.txt")
top_reasons = df_train['reason'].value_counts().head(10)
for reason, count in top_reasons.items():
    log_and_save(f"  [{count}x] {reason[:80]}...", "eda_log.txt")
log_and_save("", "eda_log.txt")

# ============================================================================
# 9. DUPLICATE DETECTION
# ============================================================================
log_and_save("="*70, "eda_log.txt")
log_and_save("DUPLICATE DETECTION", "eda_log.txt")
log_and_save("="*70 + "\n", "eda_log.txt")

duplicates = df_train.duplicated(subset=['premise', 'hypothesis', 'label'])
duplicate_count = duplicates.sum()
log_and_save(f"Duplicate samples: {duplicate_count}", "eda_log.txt")
log_and_save(f"Percentage: {duplicate_count/len(df_train)*100:.4f}%\n", "eda_log.txt")

# ============================================================================
# 10. MISSING VALUES CHECK
# ============================================================================
log_and_save("="*70, "eda_log.txt")
log_and_save("MISSING VALUES", "eda_log.txt")
log_and_save("="*70 + "\n", "eda_log.txt")

missing = df_train.isna().sum()
log_and_save("Missing values per column:", "eda_log.txt")
log_and_save(str(missing), "eda_log.txt")
log_and_save("", "eda_log.txt")

# ============================================================================
# 11. SUMMARY STATISTICS
# ============================================================================
log_and_save("="*70, "eda_log.txt")
log_and_save("SUMMARY", "eda_log.txt")
log_and_save("="*70 + "\n", "eda_log.txt")

summary = {
    'dataset_name': 'ANLI Round 2',
    'total_train_samples': len(df_train),
    'total_dev_samples': len(df_dev),
    'total_test_samples': len(df_test),
    'num_classes': len(label_names),
    'class_names': label_names,
    'avg_premise_length': float(df_train['premise_len'].mean()),
    'avg_hypothesis_length': float(df_train['hypothesis_len'].mean()),
    'avg_combined_length': float(df_train['premise_len'].mean() + df_train['hypothesis_len'].mean()),
    'unique_premise_words': unique_premise,
    'unique_hypothesis_words': unique_hypothesis,
    'duplicate_samples': int(duplicate_count),
    'missing_values': int(missing.sum()),
    'class_distribution': label_dist_dict
}

# Save summary
with open(os.path.join(ARTIFACTS_DIR, 'eda_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

log_and_save("EDA Summary:", "eda_log.txt")
log_and_save(json.dumps(summary, indent=2), "eda_log.txt")
log_and_save("\nSaved: eda_summary.json", "eda_log.txt")

# ============================================================================
# COMPLETION
# ============================================================================
log_and_save("\n" + "="*70, "eda_log.txt")
log_and_save("EDA PIPELINE COMPLETED SUCCESSFULLY", "eda_log.txt")
log_and_save("="*70, "eda_log.txt")
log_and_save(f"\nAll artifacts saved to: {ARTIFACTS_DIR}", "eda_log.txt")

# List all generated files
log_and_save("\nGenerated files:", "eda_log.txt")
for file in sorted(os.listdir(ARTIFACTS_DIR)):
    log_and_save(f"  - {file}", "eda_log.txt")

print(f"\nEDA pipeline complete. Check {ARTIFACTS_DIR}/ for results.")