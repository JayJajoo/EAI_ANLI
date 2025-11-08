# NLI BERT Fine-tuning Pipeline
## Production-Ready Docker Pipeline

Complete automated pipeline for fine-tuning BERT models on the ANLI dataset with EDA, training, and evaluation stages.

---

## 👋 Quick Start

### Using Docker (Recommended)

```bash
# Build the image
docker build -t anli_pipeline .

# Run complete pipeline with GPU
docker run --gpus all -v "$(pwd)/artifacts:/app/artifacts" anli_pipeline

# Run complete pipeline (CPU only)
docker run -v "$(pwd)/artifacts:/app/artifacts" anli_pipeline
```

### Pull & Run for Others

```bash
# Pull the image from Docker Hub
docker pull jayjajoo/anli_pipeline:latest

# Run with GPU and local volume mapping
docker run --gpus all -v "$(pwd)/artifacts:/app/artifacts" jayjajoo/anli_pipeline:latest

# Run CPU only
docker run -v "$(pwd)/artifacts:/app/artifacts" jayjajoo/anli_pipeline:latest
```

> For Windows PowerShell:
```powershell
docker run --gpus all -v "${PWD}/artifacts:/app/artifacts" jayjajoo/anli_pipeline:latest
```

### Using Python Directly

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python pipeline.py

# Or run individual stages
python eda.py      # Stage 1: EDA
python train.py    # Stage 2: Training
python eval.py     # Stage 3: Evaluation
```

---

## 🏯 Pipeline Architecture

### Three-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     PIPELINE.PY                             │
│                  (Orchestrator)                             │
└─────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Stage 1    │  │   Stage 2    │  │   Stage 3    │
│   EDA.PY     │→ │   TRAIN.PY   │→ │   EVAL.PY    │
│              │  │              │  │              │
│ - Data stats │  │ - Fine-tune  │  │ - Test model │
│ - Plots      │  │ - Save model │  │ - Metrics    │
│ - Analysis   │  │ - Logging    │  │ - Baseline   │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Stage 1: EDA (eda.py)
- Loads ANLI Round 2 dataset
- Analyzes label distribution, text lengths, word overlap
- Generates visualizations and statistics
- **Output**: `artifacts/eda/`

### Stage 2: Training (train.py)
- Loads configuration from `config.py`
- Fine-tunes BERT model on ANLI
- Implements early stopping, gradient accumulation, mixed precision
- Saves best model checkpoint
- **Output**: `artifacts/model/`, `artifacts/training/`

### Stage 3: Evaluation (eval.py)
- Loads best trained model
- Evaluates on test set
- Compares against baseline (33.7% accuracy)
- Generates confusion matrix and detailed metrics
- **Output**: `artifacts/evaluation/`

---

## 📁 File Structure

```
.
├── config.py              # Centralized configuration
├── eda.py                 # Stage 1: Exploratory Data Analysis
├── train.py               # Stage 2: Model training
├── eval.py                # Stage 3: Model evaluation
├── pipeline.py            # Main orchestrator (runs all stages)
├── requirements.txt       # Python dependencies
└── Dockerfile             # Docker container definition
```

---

## 🛠️ Configuration

All settings are in **config.py**:

```python
# Model
MODEL_NAME = "prajjwal1/bert-tiny"  # Change to bert-base-uncased/bert-large-uncased
MAX_LENGTH = 256

# Training
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 2e-5

# Paths
ARTIFACTS_DIR = './artifacts'
MODEL_DIR = './artifacts/model'
...
```

**To change settings**: Edit `config.py` before building Docker image.

---

## 📊 Artifacts Generated

```
artifacts/
│
├── pipeline/
│   ├── pipeline_log.txt
│   └── pipeline_summary.json
│
├── eda/
│   ├── eda_log.txt
│   ├── eda_summary.json
│   ├── label_distribution.json
│   ├── similarity_stats_by_label.csv
│   ├── premise_length_distribution.png
│   ├── hypothesis_length_distribution.png
│   ├── overlap_by_label.png
│   └── tfidf_similarity_comparison.png
│
├── training/
│   ├── training_log.txt
│   ├── training_history.json
│   ├── training_summary.json
│   └── best_model.pt
│
├── model/
│   ├── final_model/
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   └── config.json
│
└── evaluation/
    ├── evaluation_log.txt
    ├── evaluation_results.json
    └── confusion_matrix.png
```

---

## 💣 Docker Usage

### Build Image

```bash
# Basic build
docker build -t anli_pipeline .

# Build with specific tag
docker build -t anli_pipeline:v1.0.0 .

# Build without cache
docker build --no-cache -t anli_pipeline .
```

### Run Pipeline

```bash
# Run with GPU (recommended)
docker run --gpus all -v "$(pwd)/artifacts:/app/artifacts" anli_pipeline

# Run with specific GPU
docker run --gpus '"device=0"' -v "$(pwd)/artifacts:/app/artifacts" anli_pipeline

# Run on CPU (slower)
docker run -v "$(pwd)/artifacts:/app/artifacts" anli_pipeline
```

### Run Individual Stages

```bash
# Run only EDA
docker run --rm anli_pipeline python eda.py

# Run only training
docker run --gpus all -v "$(pwd)/artifacts:/app/artifacts" anli_pipeline python train.py

# Run only evaluation
docker run --rm -v "$(pwd)/artifacts:/app/artifacts" anli_pipeline python eval.py
```

---

## 📈 Expected Results

### Baseline (No fine-tuning)
- Accuracy: 33.7%
- F1 Score: 0.242

### After Fine-tuning (Expected)
- Accuracy: 43-45%
- F1 Score: 0.42-0.45
- Improvement: +10-12% absolute accuracy

### Training Time (GPU)
- BERT-tiny: ~10-15 minutes
- BERT-base: ~2 hours
- BERT-large: ~3 hours

---

## 💩 Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` in `config.py`
- Increase `GRADIENT_ACCUMULATION_STEPS`

### Logs Not Showing
- Run stages individually or remove `capture_output=True`

### FileNotFoundError
- Ensure Python scripts are in the same directory

### Docker Build Fails
- Clean Docker cache: `docker system prune -a`

### Model Not Beating Baseline
- Check epochs and hyperparameters in `config.py`

---

## 🗑️ Logging System

- Logs stored in `artifacts/*/` directories
- View with `cat` or `tail -f` for real-time monitoring

---

## 🛠️ Support

- Common issues: OOM, slow training, Docker build failures, GPU detection
- Fix by adjusting config, enabling GPU, cleaning cache, or updating drivers

---

**Ready to run?**

```bash
docker run --gpus all -v "$(pwd)/artifacts:/app/artifacts" anli_pipeline
```
