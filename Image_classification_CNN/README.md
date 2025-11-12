# Image Classification - CNN

This directory contains comprehensive image classification implementations using Convolutional Neural Networks (CNN). The project compares CNN performance against Fully Connected Networks (FCN) while analyzing the impact of various CNN architectural parameters.

## Project Overview

The CNN image classification project systematically evaluates different convolutional neural network architectures for image classification tasks. It demonstrates the advantages of CNNs over fully connected approaches and analyzes the effects of key architectural parameters on model performance.

## Key Components

### Data Processing
- **Custom Image Dataset**: Original image data (106 training, 12 validation, 40 test samples)
- **PyTorch Implementation**: Modern deep learning framework implementation
- **Data Preprocessing**: Standard image preprocessing and augmentation pipelines

### Network Architectures
- **Convolutional Neural Networks (CNN)**: Multi-layer CNN with varying kernel sizes and channel counts
- **Fully Connected Networks (FCN)**: Traditional dense layer approach for baseline comparison
- **PyTorch Models**: Built using PyTorch neural network modules

### Architectural Analysis
- **Kernel Size Experiments**: Analysis of 3x3 vs 5x5 convolutions in 2-layer CNN
- **Channel Width Studies**: Impact of 8, 16, 32, 64 channels in convolutional layers
- **Depth Analysis**: 2-layer vs 3-layer CNN performance comparison
- **Parameter Count**: Network capacity vs performance analysis
- **Weight Initialization**: Effects of different neural network initialization schemes

### Performance Evaluation
- **Train vs Accuracy**: Training and validation accuracy monitoring
- **Comparative Studies**: CNN vs FCN performance across 100K iterations
- **Feature Map Visualization**: Visual analysis of learned features in convolutional layers

## Visualization Outputs
- `CNN & FCN 100K Comparison.pdf`: Performance comparison over 100K training iterations
- `Different CNN Kernels (2 Convolutional Layers).pdf`: Kernel size impact analysis (3x3 vs 5x5)
- `Different CNN Weights (2 Convolutional Layers).pdf`: Channel width effects in 2-layer CNN
- `Different CNN Weights (3 Convolutional Layers).pdf`: Channel width effects in 3-layer CNN
- `Different CNN Weights Comparison.pdf`: Comprehensive weight analysis comparison
- `Different FCN Weights 2 Layers.pdf`: Fully connected network weight analysis
- `Feature Maps 2C.pdf`: Feature visualization for 2-layer CNN
- `Feature Maps 3C.pdf`: Feature visualization for 3-layer CNN

## Implementation Details
- **Framework**: PyTorch with CUDA support
- **Activation Function**: ReLU for convolutional layers
- **Optimization**: Standard gradient descent with various learning rates
- **Convolution Parameters**: Stride 1, padding preserving spatial dimensions
- **Pooling**: Max pooling for spatial downsampling

## Technical Architecture
The project demonstrates the superiority of CNNs for image processing tasks through systematic comparison with FCN approaches, showing how convolutional architectures achieve better performance with fewer parameters by leveraging spatial locality and parameter sharing.
