# NLI BERT Fine-tuning Pipeline
## Production-Ready Docker Pipeline

Complete automated pipeline for fine-tuning BERT models on the ANLI dataset with EDA, training, and evaluation stages.

---

## 📋 Quick Start

### Using Docker (Recommended)

```bash
# Build the image
docker build -t anli_pipeline .

# Run complete pipeline with GPU
docker run --gpus all -v "$(pwd)/artifacts:/app/artifacts" anli_pipeline

# Run complete pipeline (CPU only)
docker run -v "$(pwd)/artifacts:/app/artifacts" anli_pipeline
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

## 🏗️ Pipeline Architecture

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

## 🔧 Configuration

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

After pipeline completion:

```
artifacts/
│
├── pipeline/
│   ├── pipeline_log.txt          # Overall pipeline logs
│   └── pipeline_summary.json     # Timing and status
│
├── eda/
│   ├── eda_log.txt               # EDA logs
│   ├── eda_summary.json          # Dataset statistics
│   ├── label_distribution.json
│   ├── similarity_stats_by_label.csv
│   ├── premise_length_distribution.png
│   ├── hypothesis_length_distribution.png
│   ├── overlap_by_label.png
│   └── tfidf_similarity_comparison.png
│
├── training/
│   ├── training_log.txt          # Training logs
│   ├── training_history.json     # Epoch-by-epoch metrics
│   ├── training_summary.json     # Final summary
│   └── best_model.pt             # Best checkpoint
│
├── model/
│   ├── final_model/              # Saved model (HuggingFace format)
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   └── config.json               # Training configuration
│
└── evaluation/
    ├── evaluation_log.txt        # Evaluation logs
    ├── evaluation_results.json   # Complete results + baseline comparison
    └── confusion_matrix.png      # Confusion matrix plot
```

---

## 🐳 Docker Usage

### Build Image

```bash
# Basic build
docker build -t anli_pipeline .

# Build with specific tag
docker build -t anli_pipeline:v1.0.0 .

# Build without cache (clean build)
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

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"

**Solution 1**: Reduce batch size in `config.py`
```python
BATCH_SIZE = 32  # or 16
```

**Solution 2**: Increase gradient accumulation
```python
GRADIENT_ACCUMULATION_STEPS = 4  # or 8
```

### Issue: Pipeline logs not showing

**Check**: Are you using the updated `pipeline.py`?
- Should NOT have `capture_output=True` in subprocess.run()

**Verify**: Run stages individually to see their output
```bash
python eda.py
python train.py
python eval.py
```

### Issue: "FileNotFoundError: Script not found"

**Check**: All Python files in same directory
```bash
ls -la eda.py train.py eval.py config.py pipeline.py
```

### Issue: Docker build fails

**Solution 1**: Clean Docker cache
```bash
docker system prune -a
```

### Issue: Model not beating baseline

**Check**: 
1. Enough epochs? (default: 5)

**Solution**: Adjust hyperparameters in `config.py`

---

## 📝 Logging System

### Log Files

```
artifacts/
├── pipeline/
│   └── pipeline_log.txt       # Orchestration logs, timing
├── eda/
│   └── eda_log.txt            # EDA analysis logs
├── training/
│   └── training_log.txt       # Training progress, metrics
└── evaluation/
    └── evaluation_log.txt     # Evaluation results
```

### Log Levels

- **Pipeline**: Stage transitions, timing, overall status
- **EDA**: Data loading, statistics, file saves
- **Training**: Epoch progress, loss, accuracy, model saves
- **Evaluation**: Test metrics, baseline comparison, final results

### Viewing Logs

```bash
# View all logs
cat artifacts/pipeline/pipeline_log.txt
cat artifacts/training/training_log.txt
cat artifacts/evaluation/evaluation_log.txt

# Follow training in real-time
tail -f artifacts/training/training_log.txt
```

---

## 🆘 Support

### Common Issues

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce BATCH_SIZE in config.py |
| Slow training | Enable GPU, use FP16 |
| Docker build fails | Clean cache: `docker system prune -a` |
| GPU not detected | Update NVIDIA Drivers |

### Getting Help

1. Check logs in `artifacts/*/`
2. Run stages individually to isolate issues
3. Use interactive mode for debugging
4. Review configuration in `config.py`
---

**Ready to run? Execute:** `docker run --gpus all -v "$(pwd)/artifacts:/app/artifacts" anli_pipeline`