# ANLI Dataset Exploratory Data Analysis (EDA) - README

## Overview
This project performs exploratory data analysis (EDA) on the ANLI (Adversarial Natural Language Inference) dataset, round 2 (R2). The goal is to understand the structure, distribution, and relationships between premises and hypotheses before building NLP models.

## Dataset Description
Each sample in the dataset contains:
- **Premise**: Context or statement.
- **Hypothesis**: Statement to evaluate against the premise.
- **Label**: Relationship between premise and hypothesis:
  - `0`: entailment
  - `1`: neutral
  - `2`: contradiction
- **Reason**: Explanation for the label (optional).

Dataset splits:
- **Train**: Training data
- **Dev/Validation**: Hyperparameter tuning
- **Test**: Evaluation

## Methodology

### 1. Data Loading and Preprocessing
- Dataset loaded using the `datasets` library.
- Premises and hypotheses converted to lowercase for consistent tokenization.
- Ensures uniformity and reduces noise from capitalization.

### 2. Feature Exploration
- **Features and Types**: Checked dataset columns and types.
- **Dataset Shape**: Verified number of rows in train, validation, and test splits.
- **Label Names**: Mapped numeric labels to descriptive names (`entailment`, `neutral`, `contradiction`).

### 3. Label Distribution
- Calculated label proportions in the training set.
- **Purpose**: Identify class imbalance that may affect model performance.

### 4. Text Length Analysis
- Added `premise_len` and `hypothesis_len` representing word counts.
- Plotted distributions using histograms:
  - **Premise Length Distribution**
  - **Hypothesis Length Distribution**
- **Reason**:
  - Understand typical sequence lengths for tokenization/padding.
  - Detect outliers or unusually long sentences.

### 5. Vocabulary Analysis
- Calculated **unique words** in premises and hypotheses.
- Computed **top words per label** using word frequencies.
- **Reason**:
  - Identify label-specific terms and patterns.
  - Inform feature engineering and tokenization choices like if using TFIDF then what vocab_size to set.

### 6. Premise-Hypothesis Similarity
- **Jaccard Similarity**: Word overlap between premise and hypothesis.
- **TF-IDF Cosine Similarity**: Semantic similarity using vector representations.
- Plots:
  - Histogram of Jaccard overlap
  - TF-IDF cosine similarity histograms for multiple `max_features`
- **Reason**:
  - Detect similarity trends by label.
  - High overlap may indicate entailment; low overlap may indicate neutral or contradiction.

### 7. Reason Column Analysis
- Measured length of reason column and filled missing values.
- Printed most common reasons.
- **Reason**:
  - Understand human rationale behind labels.
  - Evaluate dataset complexity and reasoning patterns.

### 8. Duplicate Detection
- Checked duplicates in premise, hypothesis, and label.
- **Reason**:
  - Duplicates can bias models and inflate metrics.
  - Ensures dataset integrity.

### 9. Similarity Analysis by Label
- Grouped data by label and calculated overlap statistics:
  - Mean, Median, 25th, 50th, 75th, 90th, 95th percentiles
- Plots:
  - Histograms of overlap for each label
  - Boxplot comparing overlap across labels
- **Reason**:
  - Histograms visualize distribution for each class.
  - Boxplots provide easy comparison of medians, quartiles, and outliers.
  - Highlight differences in premise-hypothesis similarity by label.

## Summary of Insights
- Premises tend to be longer than hypotheses.
- Certain words are strongly associated with specific labels.
- Cosine similarity and Jaccard overlap are useful features for distinguishing labels.
- Most samples are unique, though some duplicates exist.
- Visualizations guide preprocessing, feature engineering, and model design.

## Conclusion
This EDA provides a detailed understanding of the ANLI dataset. The analysis of text lengths, vocabulary, similarity, and labels can inform preprocessing steps, feature selection, and modeling strategies for natural language inference tasks.

**Note**: Each plot was chosen to reveal a specific characteristic of the data, helping interpret relationships between premises, hypotheses, and labels.

