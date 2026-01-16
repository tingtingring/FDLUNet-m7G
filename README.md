## Overview

In this study, we constructed a new dataset that provides a richer and more reliable basis for subsequent model training and evaluation. In addition, we developed a new interpretable deep learning model, named **FFTUNet-m7G**, to achieve accurate and stable prediction of m7G modification sites.

---

## Main Contributions

The main contributions of this work are summarized as follows:

1. We constructed a more comprehensive and multidimensional dataset of m7G modification sites and loss-of-modification variants. Compared with previous studies, this dataset alleviates the problem of limited sample size and improves applicability under diverse biological conditions.

2. We propose FFTUNet-m7G, which integrates Fast Fourier Transform on motif-level features. Through frequency-domain decomposition, the model captures variation information at different scales, enabling effective multi-scale feature representation and fusion.

3. A U-Net–based encoder–decoder architecture is introduced to extract hierarchical sequence features. This design helps suppress redundant information and noise, leading to more robust and stable prediction performance.

4. To improve model interpretability and further investigate its prediction mechanism, structural-level and mutation-level analyses are conducted using AlphaFold-based structure modeling and in silico saturation mutagenesis (ISM), providing insights into both the model behavior and biological relevance.

In addition, systematic comparative and ablation experiments demonstrate that FFTUNet-m7G consistently outperforms existing methods, and that each component contributes meaningfully to accurate prediction. The model is further evaluated on multiple RNA modification prediction tasks, indicating strong generalization ability and broad applicability.

---

## Environment Requirements

The following environment is required to run FFTUNet-m7G:

- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- CUDA-enabled GPU is recommended

Required Python packages:

```bash
pip install torch numpy pandas scikit-learn tqdm termcolor tensorboard
Usage
Training
Model training and evaluation are performed using the following script:

bash
复制代码
python train.py
Before running, please ensure that:

The correct GPU device is specified via CUDA_VISIBLE_DEVICES

Dataset paths in train.py are properly configured

Hyperparameters (e.g., learning rate, batch size, sequence length) are set in the params dictionary

Dataset Description
The dataset is provided in FASTA format and contains both positive (m7G-modified) and negative (loss-of-modification) samples.

Each sequence header encodes the label information, for example:

makefile
复制代码
>chr19:34401574|1|train
ATGCTAGCTAGCTAGCTAG...
1 indicates a positive m7G modification site

0 indicates a negative sample

RNA sequences are automatically converted from U to T during preprocessing

All sequences are centered and padded to a fixed length (default: 201 nt)

The dataset is split into training, validation, and test sets to ensure fair and reproducible evaluation.
