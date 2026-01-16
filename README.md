# 🧬 FFTUNet-m7G  
**Interpretable Deep Learning for m7G Modification Prediction**

A novel interpretable deep learning model for accurate and stable prediction of m7G modification sites, combining frequency-domain analysis and U-Net architecture for multi-scale feature representation.

---

## 📋 Table of Contents

- [✨ Model Innovation Highlights](#-model-innovation-highlights)
- [📊 Dataset Description](#-dataset-description)
- [⚙️ Environment Installation](#️-environment-installation)
- [🚀 Model Training and Usage](#-model-training-and-usage)
- [📈 Experimental Results](#-experimental-results)
- [📄 Citation](#-citation)

---

## ✨ Model Innovation Highlights

In this study, we constructed a new dataset that provides a richer and more reliable basis for subsequent model training and evaluation. Additionally, we developed a new interpretable deep learning model, **FFTUNet-m7G**, to achieve accurate and stable prediction of m7G modification sites. The main contributions of this work are as follows:

1. **More comprehensive m7G modification dataset**  
   We built a more comprehensive and multidimensional dataset of m7G modification sites and loss-of-modification variants. This relieves the problem of limited data in previous studies and greatly improves dataset applicability under diverse biological conditions.

2. **FFT-based multi-scale feature extraction**  
   FFTUNet-m7G integrates Fast Fourier Transform on motif-level features. By performing frequency-domain decomposition, the model captures variation information at different scales, facilitating multi-scale feature representation and fusion.

3. **UNet encoder–decoder architecture**  
   A U-Net architecture is introduced to capture hierarchical sequence features through an encoder–decoder framework, reducing the influence of redundant information and noise on prediction results.

4. **Multi-level model interpretability analysis**  
   To improve model interpretability and further investigate prediction mechanisms, AlphaFold3-based structural analysis and in silico saturation mutagenesis (ISM) experiments are conducted at both structural and mutation levels.

---

## 📊 Dataset Description

A precise and reliable dataset is the foundation for effective research. In this study, we constructed a new benchmark dataset for m7G modification prediction based on **RMVar 2.0**.

### Dataset Construction Strategy

- **Data Source**  
  Filtered m7G modification sites and loss-of-modification variants from RMVar 2.0.

- **Sequence Extraction**  
  Sequences were extracted by centering each site and extending **500 nt upstream and downstream**, resulting in sequences of **1001 nt** in length.

- **Sample Classification**
  - **Reference Dataset**: Central position is a G nucleotide (m7G modification sites)
  - **Alternative Dataset**: Central position is a non-G nucleotide (loss-of-modification variants)
  - **All Dataset**: Combination of reference and alternative datasets

### Dataset Statistics

| Dataset Type | Training Set | Validation Set | Test Set | Total |
|-------------|-------------|----------------|----------|-------|
| Reference Dataset | 7,018 pos + 7,018 neg | 877 pos + 877 neg | 879 pos + 879 neg | 8,774 pos + 8,774 neg |
| Alternative Dataset | 21,346 pos + 21,346 neg | 2,668 pos + 2,668 neg | 2,669 pos + 2,669 neg | 26,683 pos + 26,683 neg |
| All Dataset | 26,434 pos + 26,434 neg | 3,304 pos + 3,304 neg | 3,305 pos + 3,305 neg | 33,043 pos + 33,043 neg |

**Note:** All sequences were clustered and deduplicated using **CD-HIT** with a sequence similarity threshold of **90%**.

---

## ⚙️ Environment Installation

### System Requirements

- Python ≥ 3.8  
- CUDA ≥ 11.8 (recommended for GPU acceleration)

### Quick Installation

#### Clone the repository

```bash
git clone https://github.com/tingtingring/FFTUNet-m7G.git
cd FFTUNet-m7G
```

## 🚀 Model Training and Usage

### Training the Model

To train and evaluate the FFTUNet-m7G model, run the following command:

```bash
python train.py
```

