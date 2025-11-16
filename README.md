# Natural Language Inference (NLI) - Complete Project

Comprehensive exploration and production implementation of Natural Language Inference models on the ANLI Round 2 dataset, ranging from exploratory notebooks to production ready pipelines.

## Project Overview

Task: Natural Language Inference (NLI)  
Dataset: Adversarial NLI (ANLI) Round 2  
Dataset Size: 45,548 train / 1,000 dev / 1,000 test  
Goal: Classify premise hypothesis relationships as:
- Entailment
- Neutral
- Contradiction

## Best Results Achieved

Updated Results with New Experiments

| Model | Accuracy | F1 (Macro) | Approach |
|-------|----------|------------|----------|
| DeBERTa v3 large | 61.8 percent | 0.6177 | Fine tuning |
| BERT large plus CoT | 45.4 percent | 0.450 | Fine tuning |
| BERT large | 44.7 percent | 0.443 | Fine tuning |
| BERT base | 43.1 percent | 0.427 | Fine tuning |
| XGBoost | 38.7 percent | 0.329 | TF IDF plus ML |
| Logistic Regression | 35.6 percent | 0.339 | TF IDF plus ML |
| DistilRoBERTa baseline | 33.7 percent | 0.242 | Pre trained |

Summary of New Improvement:
Previous best: BERT large (45 percent)  
New best: DeBERTa v3 large (61.8 percent)  
Absolute gain: 16.8 percent  
Relative gain: 37 percent  

## Repository Structure

.
├── Experiments/            Jupyter notebook based experiments  
│   ├── BasicMLAlgos/       Traditional ML baselines  
│   ├── EDA/                Exploratory data analysis  
│   └── Finetuning/         BERT and DeBERTa experiments  
└── MLPipeline/             Production ready modular pipeline  
    ├── models/             Organized model implementations  
    ├── utils/              Shared utilities  
    └── artifacts/          Training outputs and results  

## Two Ways to Use This Project

1. Experiments (Exploratory and Interactive)  
2. MLPipeline (Production and Automation)

## Learning Path

1. Start with Experiments/EDA/  
2. Try ML baselines  
3. Fine tune BERT and DeBERTa  
4. Run MLPipeline for automated experiments  

## Quick Start

Option A: Production Pipeline
cd MLPipeline/  
pip install -r requirements.txt  
python pipeline.py  

Option B: Experiments
Install dependencies and run notebooks inside Experiments/.

## Performance Progression

DistilRoBERTa baseline        33.7 percent  
Traditional ML                35 to 38 percent  
BERT base                     43.1 percent  
BERT large                    44.7 percent  
BERT large plus CoT           45.4 percent  
DeBERTa v3 large              61.8 percent  

Total improvement from baseline: 28.1 percent absolute.

## Key Insights

What Worked:
- Fine tuning transformers
- Larger models
- DeBERTa v3 large performed best

Challenges:
- ANLI R2 difficulty
- Traditional ML plateaus
- Earlier transformers overfit

New Observation:
- DeBERTa generalizes well with balanced per class performance

## Technologies Used

PyTorch, Transformers, Scikit learn, XGBoost, Pandas, NumPy, Matplotlib, Docker, MLflow, Python 3.10.

## Project Goals Achieved

All major goals completed, including new high performing DeBERTa experiments.
