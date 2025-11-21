# MSP Podcast Corpus Experimental Setup

This directory contains the experimental setup for the MSP-Podcast corpus, focusing on both classification and regression tasks for emotion recognition.

## Usage

### 1. Data Preprocessing
Create the dataset using the provided Jupyter notebook:
*   `./Preprocessing_Feature_Extraction/data_preprocess_MSP.ipynb`

### 2. Training
Load the dataset and train the models using the specific scripts for each task:
*   **`train_t1.py`**: Script for Task 1 - Emotion Classification (Categorical).
*   **`train_t2.py`**: Script for Task 2 - Emotion Regression (Valence, Arousal, Dominance).

### 3. Model Architectures
Import the desired model structure from the `./Architecture/` directory. The available models include:
*   **`cnnmodel2.py`**: A 3-layer 1D CNN with SELU activation, max pooling, and batch normalization.
*   **`transformerencoder.py`**: A speech Transformer Encoder with convolutional subsampling and positional encoding.
*   **`transformerencoderBERT.py`**: A multimodal model combining a bidirectional speech Transformer (processing forward and backward audio) with a pre-trained BERT model for text embeddings.

## Note

The MSP-Podcast dataset itself is excluded from this repository due to its size and licensing restrictions. Ensure you have access to the dataset locally.
