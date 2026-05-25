# Comparative Analysis: Stance Detection Models

Comparative Analysis of several stance-detection models and hybrids (BiLSTM, CNN+Hybrid, DeBERTa hybrid, SVM, and T5 hybrid). This repository contains Jupyter notebooks and a runnable T5 hybrid script for training, evaluation, and producing comparative metrics and plots.

## Files
- [BiLSTM.ipynb](BiLSTM.ipynb) — BiLSTM model notebook.
- [Cnn_Hybrid.ipynb](Cnn_Hybrid.ipynb) — CNN + BiLSTM + Attention hybrid notebook.
- [DeBERTa_Hybrid.ipynb](DeBERTa_Hybrid.ipynb) — DeBERTa-based hybrid notebook.
- [SVM.ipynb](SVM.ipynb) — Optimized SVM baseline notebook.
- [T5_Hybrid.py](T5_Hybrid.py) — PyTorch T5 + BiLSTM + MHA hybrid script (CLI runnable).

## Dataset
This project uses the SemEval 2016 Task 6 stance dataset. Download and licensing information are available on the dataset page:

- SemEval stance dataset: https://www.saifmohammad.com/WebPages/StanceDataset.htm

Place the downloaded CSV (for example `SemEval2016-Task6-raw-annotations-stance.csv`) in the repository root or update the dataset path inside notebooks and `T5_Hybrid.py`.

## Overview
Each notebook/script handles:
- data loading and balancing (upsampling)
- tokenization / feature extraction
- model training and evaluation
- saving results: confusion matrices, ROC curves, metrics

## Requirements
Recommended Python 3.8+ and GPU for transformer models.

Quick install (example):
```
python -m venv venv
venv\Scripts\activate     # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

Suggested minimal `requirements.txt` entries:
- torch
- transformers
- seaborn
- scikit-learn
- tensorboard
- pandas
- numpy
- matplotlib
- nltk
- tensorflow  # only required for `Cnn_Hybrid.ipynb`

## Usage

Notebooks
- Open the notebooks in Jupyter / JupyterLab / Colab and run cells top-to-bottom.
- Update the dataset path cell if your CSV is in a different location.

T5 script
- Edit the dataset path at the top of `T5_Hybrid.py` if needed.
- Run:
```
python T5_Hybrid.py
```
- Outputs (models, logs, plots, metrics) are saved to an `outputs/` directory by default.

## Outputs
Common outputs produced by scripts/notebooks:
- `outputs/original_class_distribution.png`
- `outputs/balanced_class_distribution.png`
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`
- `outputs/metrics.txt`
- TensorBoard logs under `outputs/logs` or `runs/`

## Notes & Tips
- Transformer models (DeBERTa / T5) require significant GPU memory; reduce batch sizes or sequence lengths if you run out of memory.
- For reproducible experiments, set `random_state` where available and log hyperparameters to TensorBoard.
- Adjust tokenization `max_length` and model `hidden_size` to fit your compute budget.
- Use stratified splits to preserve class balance in train/validation/test.

## Contributing
- Open an issue or submit a PR for improvements (better preprocessing, hyperparameter tuning, new baselines).
- If adding models, include a short notebook demonstrating training + evaluation.


