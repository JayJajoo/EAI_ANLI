# Natural Language Inference (NLI) - Complete Project

Comprehensive exploration and production implementation of Natural Language Inference models on the ANLI Round 2 dataset, ranging from exploratory notebooks to production-ready pipelines.

---

## 🎯 Project Overview

**Task**: Natural Language Inference (NLI)  
**Dataset**: Adversarial NLI (ANLI) Round 2  
**Dataset Size**: 45,548 train / 1,000 dev / 1,000 test  
**Goal**: Classify premise-hypothesis relationships as:
- Entailment
- Neutral  
- Contradiction

---

## 📊 Best Results Achieved

| Model | Accuracy | F1 (Macro) | Approach |
|-------|----------|------------|----------|
| **BERT-large + CoT** | **45.4%** | **0.450** | Fine-tuning |
| BERT-large | 44.7% | 0.443 | Fine-tuning |
| BERT-base | 43.1% | 0.427 | Fine-tuning |
| XGBoost | 38.7% | 0.329 | TF-IDF + ML |
| Logistic Regression | 35.6% | 0.339 | TF-IDF + ML |
| DistilRoBERTa (baseline) | 33.7% | 0.242 | Pre-trained |

Model Comparision
<img width="4769" height="1764" alt="image" src="https://github.com/user-attachments/assets/7cc7e89d-6426-4828-b649-75fb1bdba588" />

BERT-Large Performance Per Class
<img width="296" height="76" alt="image" src="https://github.com/user-attachments/assets/87e6fe9b-7c81-4359-bd6d-5e81a58075c7" />

---

## 📁 Repository Structure

```
.
├── 📂 Experiments/           # Jupyter notebook-based experiments
│   ├── BasicMLAlgos/         # Traditional ML baselines
│   ├── EDA/                  # Exploratory data analysis
│   └── Finetuning/           # BERT fine-tuning experiments
│
└── 📂 MLPipeline/            # Production-ready modular pipeline
    ├── models/               # Organized model implementations
    ├── utils/                # Shared utilities
    └── artifacts/            # Training outputs & results
```

---

## 🚀 Two Ways to Use This Project

### 1️⃣ **Experiments** (Exploratory & Interactive)
**Best for**: Understanding the problem, trying different approaches, rapid prototyping

Navigate to `Experiments/` for Jupyter notebooks covering:
- **EDA**: Dataset analysis, statistics, visualizations
- **BasicMLAlgos**: Traditional ML baselines (LR, RF, XGBoost)
- **Finetuning**: BERT model experiments with different configurations

👉 **[See Experiments/README.md](Experiments/README.md)** for detailed navigation

### 2️⃣ **MLPipeline** (Production & Automation)
**Best for**: Training multiple models systematically, reproducible experiments, deployment

A modular, production-ready pipeline that:
- ✅ Trains 12+ model configurations automatically
- ✅ Supports BERT (any HuggingFace model), LR, RF, XGBoost
- ✅ Generates comprehensive evaluation reports
- ✅ Includes Docker support
- ✅ Fully configurable via `config.py`

👉 **[See MLPipeline/README.md](MLPipeline/README.md)** for setup & usage

---

## 🎓 Learning Path (Recommended Order)

```
1. Start Here
   └── Experiments/EDA/
       └── Understand the dataset
       
2. Establish Baselines
   ├── Experiments/Finetuning/Before/
   │   └── Pre-trained model baseline (33.7%)
   └── Experiments/BasicMLAlgos/
       └── Traditional ML baselines (35-38%)
       
3. Deep Learning Experiments
   └── Experiments/Finetuning/After/
       ├── BERT-base (43.1%)
       ├── BERT-large (44.7%)
       └── BERT-large + CoT (45.4%)
       
4. Production Pipeline
   └── MLPipeline/
       └── Automated training & evaluation
```

---

## 📖 Quick Navigation

**Want to understand the data?**  
→ `Experiments/EDA/README.md`

**Want to see baseline approaches?**  
→ `Experiments/BasicMLAlgos/README.md`  
→ `Experiments/Finetuning/Before/README.md`

**Want to see fine-tuning experiments?**  
→ `Experiments/Finetuning/After/README.md`

**Want complete analysis?**  
→ `Experiments/README.md`

---

## 🛠️ Quick Start

### Option A: Run Production Pipeline

```bash
# Navigate to pipeline
cd MLPipeline/
pip install -r requirements.txt # Install dependencies
python pipeline.py  # # Run complete pipeline (EDA → Train all → Evaluate all → Compare)

# Or use Docker
docker build -t nli_pipeline .
docker run --gpus all -v "$(pwd)/artifacts:/app/artifacts" nli_pipeline

# Or pull image from docker
docker pull jayjajoo/anli_pipeline:latest
docker run --gpus all -v "$(pwd)/artifacts:/app/artifacts" jayjajoo/anli_pipeline:latest # Run with GPU
docker run -v "$(pwd)/artifacts:/app/artifacts" jayjajoo/anli_pipeline:latest # Run without GPU (CPU only)
```

### Option B: Run Experiments (Jupyter)

```bash
# Install dependencies
pip install transformers datasets torch scikit-learn xgboost pandas numpy matplotlib seaborn

# Navigate to experiments
cd Experiments/

# Run notebooks
jupyter notebook EDA/eai-eda-ipynb.ipynb
jupyter notebook BasicMLAlgos/train.ipynb
jupyter notebook Finetuning/After/BERTBase/bert-base-full-fine-tuning.ipynb
```

---

## 📈 Performance Progression

```
Evolution of Model Performance:

Baseline                 33.7% ███████████░░░░░░░░░░░░░░░░░░░░░
                                ↓
Traditional ML           38.7% ██████████████░░░░░░░░░░░░░░░░
(XGBoost)                       ↓ +5%
                                ↓
BERT-base                43.1% ████████████████░░░░░░░░░░░░░░
(Fine-tuned)                    ↓ +4.4%
                                ↓
BERT-large               44.7% █████████████████░░░░░░░░░░░░
(Fine-tuned)                    ↓ +1.6%
                                ↓
BERT-large + CoT         45.4% ██████████████████░░░░░░░░░░░
(Prompt Engineering)            ↓ +0.7%

Total Improvement: +11.7% absolute (34.7% relative)
```

---

## 🔑 Key Insights

### What Worked
✅ **Fine-tuning transformers** - Major improvement over baselines  
✅ **Chain-of-Thought prompting** - Model learns to ignore the COT prompt  
✅ **Larger models** - BERT-large > BERT-base  

### Challenges Observed
⚠️ **Task difficulty** - Even best model only reaches 45.4%  
⚠️ **Overfitting** - Significant train/test gap in all approaches  
⚠️ **Class imbalance** - Models struggle with Contradiction class  
⚠️ **Diminishing returns** - Performance plateaus around 45%  

### Dataset Characteristics
📊 **Adversarial by design** - Intentionally challenging examples  

---

## 🧪 Technologies Used

**Deep Learning**: PyTorch, Transformers (HuggingFace)  
**Traditional ML**: Scikit-learn, XGBoost  
**Data Processing**: Pandas, NumPy, Datasets  
**Visualization**: Matplotlib, Seaborn  
**Deployment**: Docker, MLflow  
**Development**: Jupyter, Python 3.10

---

## 📧 Project Structure Summary

| Folder | Purpose | Key Files |
|--------|---------|-----------|
| `Experiments/EDA/` | Data analysis | `eai-eda-ipynb.ipynb` |
| `Experiments/BasicMLAlgos/` | ML baselines | `train.ipynb` |
| `Experiments/Finetuning/Before/` | Pre-trained baseline | `eai_before_finetuning_py.ipynb` |
| `Experiments/Finetuning/After/` | Fine-tuning experiments | `bert-*-full-fine-tuning.ipynb` |
| `MLPipeline/` | Production pipeline | `pipeline.py`, `config.py` |

---

## 🎯 Project Goals Achieved

- [x] Comprehensive EDA with statistical analysis
- [x] Traditional ML baselines (LR, RF, XGBoost)
- [x] Pre-trained transformer baseline
- [x] BERT-base fine-tuning
- [x] BERT-large fine-tuning
- [x] Prompt engineering experiments
- [x] Production-ready modular pipeline
- [x] Docker containerization
- [x] Comprehensive documentation
- [x] Reproducible experiments

---

**Choose your path**: Start with `Experiments/` for learning, or jump to `MLPipeline/` for production!

For detailed documentation, see the README files in each subdirectory.
