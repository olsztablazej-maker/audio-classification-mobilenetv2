#  Song Title Prediction from Hummed and Whistled Audio
## Classical and Deep Learning Pipelines

---

# Overview

This project investigates multi-class audio classification for predicting song titles from 10-second hummed and whistled recordings using the **MLEnd Hums and Whistles II dataset**.

The task is challenging due to:

- Weakly structured acoustic signals  
- High variability in pitch, articulation, and loudness  
- Environmental noise and recording differences  
- Limited dataset size  

The objective was to systematically compare classical feature-based approaches with deep learning pipelines under controlled evaluation conditions.

---

# Methodology

Each pipeline follows a structured 4-stage framework:

## 1️ Input Stage
- 10-second waveform  
- Sampled at 22,050 Hz  

## 2️ Feature Transformation
- MFCC-based representations (classical pipelines)  
- Mel-spectrogram representations (deep learning pipelines)  

## 3️ Model Stage
- Classical ML models  
- Dense neural network  
- Transfer learning CNN (MobileNetV2)  

## 4️ Evaluation Stage
- Accuracy  
- Precision / Recall / F1-score  
- Confusion matrices  
- Learning curves  

Group-aware splitting was used to prevent data leakage across segments.

---

# Implemented Pipelines

## Pipeline A – Classical MFCC Baseline
- 60 MFCCs + delta + delta-delta features  
- Logistic Regression  
- SVM (Linear & RBF)  
- Random Forest  

Purpose: Establish a conventional engineered-feature baseline.

---

## Pipeline B – Extended Spectral Features
Adds:
- Chroma features  
- Spectral contrast  
- Spectral centroid  
- Spectral bandwidth  
- Spectral rolloff  
- Zero-crossing rate  

Includes a soft-voting ensemble:
- RBF SVM  
- Random Forest  
- k-Nearest Neighbours  

---

## Pipeline C – Dense Neural Network
- Input: Mel-spectrograms  
- Fully connected neural network  
- Adam optimiser  
- Early stopping applied  

Purpose: Test representation learning without convolutional inductive bias.

---

## Pipeline D – Transfer Learning (MobileNetV2)

- Mel-spectrograms resized to 224×224×3  
- Pre-trained MobileNetV2 backbone (frozen)  
- Custom classifier head with dropout  
- Sparse categorical cross-entropy loss  
- Adam optimiser  

This pipeline achieved the strongest generalisation performance.

---

# Engineering Considerations

- Group-wise dataset splitting to avoid data leakage  
- Feature normalisation for stability  
- Cross-validation for robustness  
- Error analysis to evaluate failure modes  
- Controlled comparison across modelling strategies  

---

# Tech Stack

- Python  
- PyTorch  
- scikit-learn  
- NumPy  
- pandas  
- matplotlib  

---

# Notes

Dataset not included in this repository.  
Update dataset paths in the configuration section before running the notebook.
