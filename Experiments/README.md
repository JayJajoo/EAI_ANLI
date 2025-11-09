# BERT Fine-tuning for Natural Language Inference (NLI)

Fine-tuning BERT models on the Adversarial NLI (ANLI) Round 2 dataset with comprehensive analysis of different approaches including traditional ML baselines.

---

## 📊 Project Overview

**Task**: Natural Language Inference (NLI)  
**Dataset**: ANLI Round 2 (45K train, 1K val, 1K test)  
**Goal**: Classify premise-hypothesis relationships as Entailment, Neutral, or Contradiction

---

## 🎯 Key Results

### Deep Learning Models

| Approach | Accuracy | F1 Score (Macro) | Improvement |
|----------|----------|------------------|-------------|
| Baseline (DistilRoBERTa) | 33.7% | 0.242 | - |
| BERT-base | 43.1% | 0.427 | +9.4% |
| BERT-large | 44.7% | 0.443 | +11.0% |
| BERT-large + CoT | **45.4%** | **0.450** | **+11.7%** |

### Traditional ML Baselines (TF-IDF + max_features=15K)

| Model | Train Acc | Dev Acc | Test Acc | Test F1 (Macro) | Beats Baseline |
|-------|-----------|---------|----------|-----------------|----------------|
| Logistic Regression | 72.4% | 32.6% | **35.6%** | **0.339** | ✓ |
| Random Forest | 52.6% | 36.0% | 36.5% | 0.245 | ✓ |
| XGBoost | 64.3% | 37.8% | 38.7% | 0.329 | ✓ |

**Key Observations:**
- All ML models beat the 33.7% baseline
- XGBoost achieved highest test accuracy (38.7%) among traditional ML
- Logistic Regression had best F1 score (0.339) among ML models
- Significant overfitting observed (train vs test gap)
- Deep learning models substantially outperform traditional ML (+6-10% absolute)

---

## 📁 Project Structure

```
Experiments/
│
├── 📂 BasicMLAlgos/
│   └── 📓 train.ipynb                    # TF-IDF + LR/RF/XGBoost baselines
│
├── 📂 EDA/
│   ├── 📄 README.md                      # EDA findings & methodology
│   └── 📓 eai-eda-ipynb.ipynb           # Interactive EDA notebook
│
└── 📂 Finetuning/
    │
    ├── 📂 After/
    │   ├── 📂 BERTBase/
    │   │   └── 📓 bert-base-full-fine-tuning.ipynb      # 43.1% acc
    │   │
    │   ├── 📂 BERTLarge/
    │   │   ├── 📓 bert-large-full-fine-tuning.ipynb     # 44.7% acc
    │   │   └── 📓 promt_based_bert-large-full-fine-tuning.ipynb  # 45.4% acc
    │   │
    │   └── 📄 README.md                  # Fine-tuning experiments documentation
    │
    ├── 📂 Before/
    │   ├── 📓 eai_before_finetuning_py.ipynb           # Baseline: 33.7% acc
    │   └── 📄 README.md                  # Baseline evaluation documentation
    │
    └── 📄 README.md                      # Finetuning overview
```

---

## 📖 How to Navigate

### 0️⃣ Traditional ML Baselines
**→ [BasicMLAlgos/train.ipynb](BasicMLAlgos/train.ipynb)**
- TF-IDF feature extraction (15K features, 1-2 grams)
- Logistic Regression, Random Forest, XGBoost
- All models beat the 33.7% baseline
- Best ML result: XGBoost with 38.7% accuracy
- Serves as strong traditional ML benchmark before deep learning

### 1️⃣ Understanding the Data
**→ [EDA/README.md](EDA/README.md)**
- Dataset statistics (label distribution, text lengths)
- Similarity analysis (informed context length: 256 vs 512 tokens)
- Key finding: 22% word overlap in contradictions

**→ [EDA/eai-eda-ipynb.ipynb](EDA/eai-eda-ipynb.ipynb)**
- Interactive notebook with visualizations

### 2️⃣ Baseline Performance
**→ [Finetuning/Before/README.md](Finetuning/Before/README.md)**
- Pre-trained DistilRoBERTa evaluation
- 33.7% accuracy baseline
- Extreme Neutral bias identified (85% predictions)

**→ [Finetuning/Before/eai_before_finetuning_py.ipynb](Finetuning/Before/eai_before_finetuning_py.ipynb)**
- Baseline evaluation notebook

### 3️⃣ Fine-tuning Experiments
**→ [Finetuning/After/README.md](Finetuning/After/README.md)**
- Overview of all fine-tuning approaches
- Results comparison
- Analysis of overfitting issues

**→ Individual Experiments:**
- **[BERTBase/bert-base-full-fine-tuning.ipynb](Finetuning/After/BERTBase/bert-base-full-fine-tuning.ipynb)** - 109M params, 43.1% acc
- **[BERTLarge/bert-large-full-fine-tuning.ipynb](Finetuning/After/BERTLarge/bert-large-full-fine-tuning.ipynb)** - 335M params, 44.7% acc
- **[BERTLarge/promt_based_bert-large-full-fine-tuning.ipynb](Finetuning/After/BERTLarge/promt_based_bert-large-full-fine-tuning.ipynb)** - With CoT, 45.4% acc

### 4️⃣ Complete Analysis
**→ [Finetuning/README.md](Finetuning/README.md)**
- Full results comparison
- Why performance plateaued at ~45%
- Overfitting deep-dive

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install transformers datasets torch scikit-learn pandas numpy matplotlib seaborn tqdm xgboost

# Run traditional ML baselines
jupyter notebook BasicMLAlgos/train.ipynb

# Run EDA
jupyter notebook EDA/eai-eda-ipynb.ipynb

# Run baseline
jupyter notebook Finetuning/Before/eai_before_finetuning_py.ipynb

# Run fine-tuning (choose one)
jupyter notebook Finetuning/After/BERTBase/bert-base-full-fine-tuning.ipynb
```

---

## 🛠️ Technical Specs

**Hardware**: NVIDIA Tesla T4  
**Framework**: 
- Deep Learning: PyTorch + Transformers  
- Traditional ML: Scikit-learn + XGBoost
**Training Time**: 
- ML Models: <5 minutes
- BERT-base: ~3 hours
- BERT-large: ~5.5 hours (with CoT)

---

## 📈 Performance Comparison

```
Model Performance on ANLI R2 Test Set (Accuracy):

Traditional ML:
├── Logistic Regression  35.6% ████████████░░░░░░░░░░░░░░░░░░░░
├── Random Forest        36.5% █████████████░░░░░░░░░░░░░░░░░░
└── XGBoost              38.7% ██████████████░░░░░░░░░░░░░░░░

Deep Learning:
├── BERT-base            43.1% ████████████████░░░░░░░░░░░░░░
├── BERT-large           44.7% █████████████████░░░░░░░░░░░░
└── BERT-large + CoT     45.4% ██████████████████░░░░░░░░░░░

Baseline                 33.7% ███████████░░░░░░░░░░░░░░░░░░░░░
```

**Key Insights:**
- Traditional ML provides solid baselines (+2-5% over DistilRoBERTa)
- Deep learning models achieve +6-12% improvement over ML baselines
- BERT-large with Chain-of-Thought prompting yields best results
- Task remains challenging: even best model achieves only 45.4% (vs 33.3% random)

---

**For detailed methodology, results, and analysis, see the documentation files linked above.**