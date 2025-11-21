# Emo-DB Experimental Setup

This directory contains the experimental setup for the Berlin Database of Emotional Speech (Emo-DB).

## Directory Structure

*   `Architecture/`: Contains the PyTorch model definitions (CNN, LSTM-CNN, Transformer, etc.).
*   `Preprocessing_Feature_Extraction/`: Notebooks for data preprocessing and feature extraction.
    *   `BerlinEMO_DB_preprocessing.ipynb`: Main preprocessing pipeline.
    *   `Feature_Visualization.ipynb`: Visualization of audio features.
    *   `Open_Smile_Features/Emo_DB_arff.ipynb`: Processing of OpenSMILE features.
*   `Training/`: Contains training logs and evaluation results.
*   `Train_Berlin_EmoDB.py`: Main training script.

## Usage

### 1. Preprocessing
To create the dataset, run the preprocessing notebook:
`./Preprocessing_Feature_Extraction/BerlinEMO_DB_preprocessing.ipynb`

### 2. Training
Train a model using the main script:
```bash
python Train_Berlin_EmoDB.py
```

You can modify the script to select different model architectures from the `Architecture/` directory.

### 3. Visualization
Use `Preprocessing_Feature_Extraction/Feature_Visualization.ipynb` to visualize features and augmentation techniques.

## Note on Reproducibility

The experimental setup of Emo-DB has known reproducibility challenges. Evaluation results and tensorboard events are preserved in the `./Training/` folder for reference.
