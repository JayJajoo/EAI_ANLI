# NLI Model Training & Evaluation Pipeline

Complete pipeline for training and evaluating multiple Natural Language Inference (NLI) models on the ANLI dataset.

## 📁 Project Structure

```
pipeline_folder/
├── artifacts/                      # All outputs saved here
│   ├── eda/                       # EDA results and plots
│   ├── bert/                      # BERT model artifacts
│   │   ├── bert-tiny/
│   │   ├── bert-base-uncased/
│   │   └── roberta-base/
│   ├── logistic_regression/       # LR model artifacts
│   │   ├── lr-default/
│   │   ├── lr-l1-balanced/
│   │   └── lr-strong-reg/
│   ├── random_forest/             # RF model artifacts
│   │   ├── rf-default/
│   │   ├── rf-deep/
│   │   └── rf-shallow/
│   ├── xgboost/                   # XGBoost model artifacts
│   │   ├── xgb-default/
│   │   ├── xgb-deep/
│   │   └── xgb-regularized/
│   └── pipeline/                  # Pipeline logs and reports
│       ├── pipeline_log.txt
│       ├── pipeline_summary.json
│       ├── comparison_report.json
│       └── model_comparison.png
├── models/                        # Model implementations
│   ├── __init__.py
│   ├── bert/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── eval.py
│   ├── logistic_regression/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── eval.py
│   ├── random_forest/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── eval.py
│   └── xgboost/
│       ├── __init__.py
│       ├── train.py
│       └── eval.py
├── utils/                         # Shared utilities
│   ├── __init__.py
│   ├── data_loader.py            # Data loading
│   └── eda.py                    # EDA utilities
├── config.py                     # All configurations
├── pipeline.py                   # Main pipeline orchestrator
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration
└── README.md                     # This file
```

## 🚀 Quick Start

### Installation

1. **Clone/Download the project**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create empty `__init__.py` files:**
   ```bash
   touch utils/__init__.py
   touch models/__init__.py
   touch models/bert/__init__.py
   touch models/logistic_regression/__init__.py
   touch models/random_forest/__init__.py
   touch models/xgboost/__init__.py
   ```

### Running the Pipeline

**Basic usage:**
```bash
python pipeline.py
```

This will:
1. ✅ Run EDA on the ANLI dataset  
2. ✅ Train all configured models (3 BERT + 3 LR + 3 RF + 3 XGBoost = 12 models)  
3. ✅ Evaluate all models on the test set  
4. ✅ Generate comparison report and visualizations  

---

## 🐳 Using Docker

### Option 1: Build Locally

```bash
# Build image
docker build -t anli_pipeline .

# Run with GPU
docker run --gpus all -v "$(pwd)/artifacts:/app/artifacts" anli_pipeline

# Run without GPU (CPU only)
docker run -v "$(pwd)/artifacts:/app/artifacts" anli_pipeline
```

### Option 2: Pull from Docker Hub

You can directly pull the pre-built image instead of building it yourself:

```bash
# Pull the image
docker pull jayjajoo/anli_pipeline:latest

# Run with GPU
docker run --gpus all -v "$(pwd)/artifacts:/app/artifacts" jayjajoo/anli_pipeline:latest

# Run without GPU (CPU only)
docker run -v "$(pwd)/artifacts:/app/artifacts" jayjajoo/anli_pipeline:latest
```

All outputs (logs, metrics, visualizations, and models) will be saved to your local `artifacts/` directory.

---

## ⚙️ Configuration

All configurations are centralized in `config.py`:

### Adding/Modifying BERT Models

```python
# In config.py, add to BERT_CONFIGS list:
{
    'name': 'distilbert',
    'model_name': 'distilbert-base-uncased',  # Any HuggingFace model
    'max_length': 128,
    'batch_size': 32,
    'epochs': 3,
    'learning_rate': 2e-5,
}
```

### Adding/Modifying ML Models

Example for Logistic Regression:
```python
{
    'name': 'lr-custom',
    'max_iter': 1000,
    'C': 1.0,
    'solver': 'lbfgs',
    'penalty': 'l2',
    'tfidf_max_features': 10000,
    'tfidf_ngram_range': (1, 2)
}
```

---

## 📊 Output Structure

Each model configuration saves:
- `training_log.txt`  
- `evaluation_log.txt`  
- `training_results.json`  
- `evaluation_results.json`  
- `confusion_matrix.png`  
- `model.pkl` or `final_model/`

Pipeline-level outputs:
- `pipeline_log.txt`  
- `comparison_report.json`  
- `model_comparison.png`

---

## 🎯 Features

- Modular model design  
- Fully configurable via `config.py`  
- Multi-model training and comparison  
- Comprehensive evaluation metrics  
- Reproducible and well-logged runs  

---

## 📈 Baseline Metrics

| Metric | Baseline |
|--------|-----------|
| Accuracy | 0.337 |
| F1 (Macro) | 0.242 |

---

## 🐛 Troubleshooting

**Out of memory errors:**
- Reduce `batch_size` or `tfidf_max_features`

**Import errors:**
- Ensure `__init__.py` files exist

**Dataset issues:**
- Verify internet connection (dataset is downloaded from HuggingFace)

---

## 👤 Author

Jay Jajoo
