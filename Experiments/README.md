# BERT Fine-tuning for Natural Language Inference (NLI)

Fine-tuning BERT models on the Adversarial NLI (ANLI) Round 2 dataset with comprehensive analysis of different approaches.

---

## 📊 Project Overview

**Task**: Natural Language Inference (NLI)  
**Dataset**: ANLI Round 2 (45K train, 1K val, 1K test)  
**Goal**: Classify premise-hypothesis relationships as Entailment, Neutral, or Contradiction

---

## 🎯 Key Results

| Approach | Accuracy | F1 Score | Improvement |
|----------|----------|----------|-------------|
| Baseline (DistilRoBERTa) | 33.7% | 0.242 | - |
| BERT-base | 43.1% | 0.427 | +9.4% |
| BERT-large | 44.7% | 0.443 | +11.0% |
| BERT-large + CoT | **45.4%** | **0.450** | **+11.7%** |

---

## 📁 Project Structure

```
Experiments/
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
pip install transformers datasets torch scikit-learn pandas numpy matplotlib seaborn tqdm

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
**Framework**: PyTorch + Transformers  
**Training Time**: 3hrs (base) to 5.5h (large+CoT)
---

**For detailed methodology, results, and analysis, see the documentation files linked above.**