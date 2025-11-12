# Captured Motion Recognition

This directory contains motion recognition and gesture classification implementations using Principal Component Analysis (PCA) and machine learning classifiers.

## Project Overview

The project analyzes human motion capture data to classify different types of movements including walking, jumping, and running. It includes comprehensive analysis of motion patterns using dimensionality reduction techniques and evaluates classification performance across different machine learning approaches.

## Key Components

### Data Processing
- Loads motion capture data from `.npy` files containing movement patterns
- Supports three movement types: walking, jumping, and running (5 samples each)
- Each sample contains motion capture matrices of shape (114, 100)

### Analysis and Visualization
- **Principal Component Analysis (PCA)**: Dimensionality reduction for motion pattern analysis
- **Cumulative Explained Variance**: Analysis of PCA components at different variance thresholds (70%, 80%, 90%, 95%)
- **2D and 3D PCA Projections**: Visualization of motion patterns in reduced dimensional space
- **PCA Spatial Modes**: Identification of key motion components

### Classification Results
- **Centroid vs k-NN Classifiers**: Comparative analysis of classification algorithms
- **Training Accuracy**: Performance metrics across different classifiers
- **Cross-validation**: Evaluation of model performance on motion recognition tasks

## Visualization Outputs
- `2D PCA Projection.pdf`: 2D visualization of motion patterns
- `3D PCA Projection.pdf`: 3D visualization of motion patterns
- `PCA Cumulative Energy.pdf`: Cumulative variance explained analysis
- `Comparison of Centroid and k-NN Classifiers.pdf`: Classifier performance comparison
- `Trained Classifier Accuracy.pdf`: Training accuracy metrics
- `Trained vs Test Accuracy.pdf`: Model validation results

## Methodology
The project applies machine learning techniques to motion capture data, using PCA for dimensionality reduction and various classification algorithms to distinguish between different human activities based on their motion signatures.