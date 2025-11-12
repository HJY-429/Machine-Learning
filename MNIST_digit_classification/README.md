# MNIST Digit Classification

This directory contains comprehensive handwritten digit recognition implementations using the MNIST database. The project applies Principal Component Analysis (PCA) and ridge classification to the classic 0-9 digit recognition problem.

## Project Overview

The MNIST digit classification project demonstrates dimensionality reduction techniques applied to handwritten digit recognition. It analyzes the structure of handwritten digits using PCA and compares classification performance before and after dimensionality reduction.

## Key Components

### Data Processing
- **MNIST Dataset**: Standard 28x28 pixel grayscale images (784 features)
- **Dataset Split**: 60,000 training images and 10,000 test images
- **Multi-class Classification**: 10 classes (digits 0-9)
- **Binary data loader**: Custom function to read IDX format

### Analysis and Visualization
- **2D PCA Projections**: Comparative analysis of different digit pairs
  - Digits 1 vs 8: Most distinguishable classes
  - Digits 2 vs 7: Intermediate similarity
  - Digits 3 vs 8: High similarity challenge
- **Principal Component Modes**: Visualization of first 16 PC modes capturing digit variation patterns
- **Training Image Samples**: Display of first 64 training images
- **Singular Value Energy**: Cumulative energy analysis showing dimensionality reduction effectiveness

### Dimensionality Reduction Results
- **PCA Transformation**: Projects 784D feature space to lower dimensions
- **Reconstruction Analysis**: Reconstructed images using truncated PCA modes
- **Variance Explained**: Quantitative analysis of information retained vs dimensions used

## Visualization Outputs
- `First 64 Training Images.pdf`: Sample of training data showing digit diversity
- `2D PCA Projection of Digits 1 and 8.pdf`: Most separable digit pair analysis
- `2D PCA Projection of Digits 2 and 7.pdf`: Intermediate difficulty classification
- `2D PCA Projection of Digits 3 and 8.pdf`: Most challenging classification task
- `First 16 PC Modes.pdf`: Principal component mode visualization
- `Cumulative Energy of Singular Values.pdf`: Dimensionality reduction effectiveness
- `Reconstructed Images Using 59 PC Modes.pdf`: PCA reconstruction quality demonstration

## Classification Approach
- **Ridge Regression**: Linear classification using ridge regularization parameter (alpha=1.0)
- **Dimensionality Reduction**: PCA preprocessing for noise reduction
- **Multi-class Strategy**: One-vs-rest classification for 10 digit classes

## Technical Details
- **Input Dimensions**: 28×28 = 784 pixel values per image
- **Training Set**: 60,000 images across 10 digit classes
- **Test Set**: 10,000 images for performance evaluation
- **Feature Range**: Grayscale pixel values [0, 255]
- **Data Format**: IDX format with binary header structure

## Methodology
The project demonstrates the application of unsupervised learning (PCA) for feature extraction followed by supervised learning (ridge classification) for digit recognition, showing how dimensionality reduction can improve classification performance while reducing computational requirements.
