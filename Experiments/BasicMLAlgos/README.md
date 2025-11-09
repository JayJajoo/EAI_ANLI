# Traditional ML Baselines for NLI

Traditional machine learning approaches using TF-IDF features for Natural Language Inference on ANLI R2 dataset.

---

## 📋 Overview

This folder contains baseline experiments using classical ML algorithms before moving to deep learning approaches.

**Approach**: TF-IDF feature extraction + Traditional ML classifiers

---

## 📊 Results Summary

| Model | Test Accuracy | Test F1 (Macro) | Status |
|-------|--------------|-----------------|--------|
| **Logistic Regression** | **35.6%** | **0.339** | Beats Baseline ✓ |
| **Random Forest** | 36.5% | 0.245 | Beats Baseline ✓ |
| **XGBoost** | 38.7% | 0.329 | Beats Baseline ✓ |
| *Baseline (DistilRoBERTa)* | *33.7%* | *0.242* | *Reference* |

**Best Traditional ML Model**: XGBoost with 38.7% accuracy

---

## 🔧 Technical Details

### Feature Extraction
- **Method**: TF-IDF Vectorization
- **Max Features**: 15,000
- **N-gram Range**: (1, 2) - unigrams and bigrams
- **Text Preprocessing**: Lowercase, combined premise + hypothesis

### Models Trained
1. **Logistic Regression** - Linear classifier with L2 regularization
2. **Random Forest** - 100 trees, max depth 20
3. **XGBoost** - 100 estimators, max depth 6

---

## 📁 Contents

```
BasicMLAlgos/
├── train.ipynb          # Main training notebook
└── README.md           # This file
```

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install scikit-learn xgboost pandas numpy matplotlib seaborn datasets

# Run the notebook
jupyter notebook train.ipynb
```

The notebook will:
1. Load ANLI R2 dataset
2. Extract TF-IDF features
3. Train 3 ML models
4. Generate comparison plots and confusion matrices
5. Save results to `./artifacts/ml_baseline/results.json`

---

## 📈 Key Findings

✅ **All models beat the baseline** (33.7%)  
✅ **XGBoost performed best** among traditional ML (38.7%)  
✅ **Logistic Regression had best F1** score (0.339)  
⚠️ **Significant overfitting observed** (72% train vs 36% test for LR)  
⚠️ **Deep learning outperforms** by 6-10% absolute (see main README)

---

## 📊 Artifacts Generated

After running the notebook, check `./artifacts/ml_baseline/`:
- `results.json` - Complete results and predictions
- `model_comparison.png` - Accuracy & F1 comparison plot
- `confusion_matrices.png` - Per-model confusion matrices
- `ml_baseline_log.txt` - Detailed training log
- `models/` - Saved model files (.pkl)

---

## 💡 Why This Matters

These traditional ML baselines establish:
- **Lower bound performance**: What's achievable without deep learning
- **Feature engineering insights**: TF-IDF captures ~36% accuracy
- **Computational baseline**: <5 minutes training vs hours for BERT
- **Comparison reference**: Shows value of transformers (+6-12% improvement)

---

## 🔗 Next Steps

After reviewing these baselines:
1. See `../EDA/` for data analysis
2. Check `../Finetuning/Before/` for transformer baseline
3. Explore `../Finetuning/After/` for fine-tuned BERT models achieving 43-45% accuracy

---

**Note**: Traditional ML serves as an important baseline but is outperformed by fine-tuned transformers. However, these models are much faster to train and require no GPU!