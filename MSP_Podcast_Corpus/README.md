# MSP Podcast Corpus Experimental Setup

This directory contains the experimental setup for the MSP-Podcast corpus.

## Usage

1.  Create a dataset using the notebook:
    `./Preprocessing_Feature_Extraction/data_preprocess_MSP.ipynb`

2.  Load the dataset with:
    *   `train_t1.py` for the classification task.
    *   `train_t2.py` for the regression task.

3.  Import the desired model structure from `./Architecture/`.
    *   **`cnnmodel2.py`**: A 3-layer 1D CNN with SELU activation, max pooling, and batch normalization.
    *   **`transformerencoder.py`**: A speech Transformer Encoder with convolutional subsampling and positional encoding.
    *   **`transformerencoderBERT.py`**: A multimodal model combining a bidirectional speech Transformer (processing forward and backward audio) with a pre-trained BERT model for text embeddings.

## Note

The dataset is excluded from this repository due to its size.
