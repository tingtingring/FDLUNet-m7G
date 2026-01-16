FTNet: Frequency-aware Transformer U-Net for RNA Modification Site Prediction

This repository provides the implementation of FTNet, a deep learning framework for RNA modification site prediction.
FTNet integrates frequency-domain decomposition, 1D U-Net, and Transformer-based contextual modeling to capture both local and global sequence patterns.

🔬 Model Overview

FTNet is designed for binary classification of RNA modification sites (e.g. m7G, m1A, m5C, A-to-I).

Key components:

One-hot encoding (1-mer) of RNA/DNA sequences

Initial CNN encoder for low-level feature extraction

Frequency-domain decomposition (FFT):

Low-frequency branch

High-frequency branch

Raw (time-domain) branch

Three parallel U-Net + Transformer branches

Feature fusion with channel-wise attention

Center-position classification head

🧠 Architecture
Input Sequence
     │
1-mer Encoding + One-hot
     │
Initial CNN
     │
FFT Decomposition
 ┌───────────┬───────────┬───────────┐
 │ Low-freq  │ High-freq │ Raw        │
 │  U-Net +  │ U-Net +   │ U-Net +    │
 │ Transformer│Transformer│Transformer│
 └───────────┴───────────┴───────────┘
     │
Feature Fusion (FC-based attention)
     │
Center Position Feature
     │
Binary Classification

📁 Project Structure
.
├── unet_cnn_ft.py        # FTNet model definition
├── train.py              # Training and evaluation script
├── data/                 # FASTA datasets
│   └── m7G/
│       ├── train_ref.fasta
│       ├── val_ref.fasta
│       └── test_ref.fasta
├── save/                 # Saved model checkpoints
├── results/              # Training logs
└── README.md

🧬 Input Data Format

FASTA format is required.

Example:
>chr19:34401574|1|train
ATGCTAGCTAGCTAGCTAG...
>chr19:34401575|0|train
CGATCGATCGATCGATCGA...


Label is parsed from FASTA header:

1 → positive sample

0 → negative sample

RNA bases (U) will be automatically converted to T

⚙️ Environment Requirements

Python ≥ 3.8

PyTorch ≥ 1.10

CUDA-enabled GPU (recommended)

Required packages
pip install torch numpy pandas scikit-learn tqdm termcolor tensorboard

🚀 Training
Step 1: Set GPU
export CUDA_VISIBLE_DEVICES=0


(or modify it directly in the code)

Step 2: Configure Hyperparameters

In train.py:

params = {
    'lr': 1e-4,
    'batch_size': 64,
    'epoch': 100,
    'seq_len': 201,
    'seed': 17,
    'patience': 10,
    'index': 10
}

Step 3: Run Training
python train.py

📊 Evaluation Metrics

The following metrics are reported:

Accuracy (ACC)

Balanced Accuracy (BACC)

Sensitivity (SE)

Specificity (SP)

Matthews Correlation Coefficient (MCC)

Area Under ROC Curve (AUC)

Early stopping is applied based on validation accuracy.

💾 Model Checkpoints

Best models are saved automatically:

save/seq_len201/seed17_YYYYMMDD_HHMMSS_acc0.XXXX.pth


Each checkpoint includes:

Model weights

Best validation accuracy

Training epoch

Hyperparameters

🧪 Supported Tasks

The framework supports multiple RNA modification datasets by changing data loaders:

m7G

m1A

m5C

A-to-I

Custom FASTA datasets

You can switch datasets by modifying the read_fasta_* function calls in evaluation_method().

📌 Reproducibility

Fixed random seeds

Deterministic CUDA settings enabled

Same sequence centering and padding strategy across datasets

📬 Contact

If you have questions or want to collaborate, feel free to open an issue or contact the author.
