# MRL Eye Dataset -- Drowsiness Detection (PyTorch)

A lightweight **CNN classifier** for **open vs. closed eyes** using the
**MRL Eye Dataset**.\
Includes preprocessing, subject-wise split, training, evaluation,
augmentation, and confusion matrix.

##  Dataset Download

Download from Kaggle:\
https://www.kaggle.com/datasets/muhammadkhalid/mrl-remote-ir-eye-dataset

Place extracted files into:

    data/MRL/

##  Project Summary

-   Grayscale inputs (128×128)
-   2 classes → Open (0) / Closed (1)
-   Custom CNN + Dropout
-   Subject-wise split (no leakage)
-   \~90% validation accuracy

## Model (Simple CNN)

    Conv2d → ReLU → MaxPool
    Conv2d → ReLU → MaxPool
    Flatten
    Linear → ReLU → Dropout
    Linear (2 classes)

##  Training

-   Optimizer: Adam (lr=1e-3)
-   Batch Size: 64
-   Epochs: 10
-   Augmentation: flip, rotation, normalization

Run:

    python train.py

Outputs saved to:

    results/


##  Outputs

-   Accuracy curves
-   Confusion matrix
-   Precision, Recall, F1

