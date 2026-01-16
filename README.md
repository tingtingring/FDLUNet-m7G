🧬 FFTUNet-m7G - Interpretable Deep Learning for m7G Modification Prediction

A novel interpretable deep learning model for accurate and stable prediction of m7G modification sites, combining frequency domain analysis and UNet architecture for multi-scale feature representation.

📋 Table of Contents
✨ Model Innovation Highlights

📊 Dataset Description

⚙️ Environment Installation

🚀 Model Training and Usage

📈 Experimental Results

📄 Citation

✨ Model Innovation Highlights
In this study, we constructed a new dataset that provides a richer and more reliable basis for subsequent model training and evaluation. Additionally, we developed a new interpretable deep learning model, FFTUNet-m7G, to achieve accurate and stable prediction of m7G modification sites. The main contributions of this work are as follows:

(1) More comprehensive m7G modification dataset: We built a more comprehensive and multidimensional dataset of m7G modifications and loss-of-modification variants. This helps relieve the problem of limited data in previous studies and greatly improves the applicability of the dataset under diverse biological conditions.

(2) FFT-based multi-scale feature extraction: FFTUNet-m7G combines Fast Fourier Transform on motif-level features. By performing frequency-domain decomposition, the model captures variation information at different scales, which facilitates multi-scale representation and fusion of features.

(3) UNet encoder-decoder architecture: We introduce a UNet architecture, which captures sequence features through an encoder–decoder framework and reduces the influence of redundant information and noise on the prediction results.

(4) Multi-level model interpretability analysis: To improve model interpretability and further investigate its prediction mechanism, we employed AlphaFold3 and in silico saturation mutagenesis (ISM) experiments to perform structural-level and mutation-level analyses of the model and the associated data.

📊 Dataset Description
A precise and reliable dataset is the foundation and key starting point for conducting effective research. In this study, we constructed a new benchmark dataset for m7G modification based on RMVar 2.0.

Dataset Construction Strategy
Data Source: Filtered m7G modification sites and loss-of-modification variants from RMVar 2.0

Sequence Extraction: Extracted sequences centered on each site, extending 500 nt upstream and downstream (total length 1001 nt)

Sample Classification:

Reference Dataset: Central position is a G nucleotide (m7G modification sites)

Alternative Dataset: Central position is a non-G nucleotide (variants leading to loss of m7G modification)

All Dataset: Combined reference and alternative samples

Dataset Statistics
Dataset Type	Training Set	Validation Set	Test Set	Total
Reference Dataset	7,018 pos + 7,018 neg	877 pos + 877 neg	879 pos + 879 neg	8,774 pos + 8,774 neg
Alternative Dataset	21,346 pos + 21,346 neg	2,668 pos + 2,668 neg	2,669 pos + 2,669 neg	26,683 pos + 26,683 neg
All Dataset	26,434 pos + 26,434 neg	3,304 pos + 3,304 neg	3,305 pos + 3,305 neg	33,043 pos + 33,043 neg
Note: All sequences were clustered and deduplicated using CD-HIT with a sequence similarity threshold of 90%.

⚙️ Environment Installation
System Requirements
Python 3.8+

CUDA 11.8+ (for GPU acceleration)

Quick Installation
Clone the Repository

bash
git clone https://github.com/yourusername/FFTUNet-m7G.git
cd FFTUNet-m7G
Install Dependencies Using requirements.txt

bash
pip install -r requirements.txt
Main Dependencies
torch>=2.0.0

numpy>=1.24.0

pandas>=2.0.0

scikit-learn>=1.3.0

matplotlib>=3.7.0

seaborn>=0.12.0

biopython>=1.81

🚀 Model Training and Usage
Train the Model
bash
python train.py
