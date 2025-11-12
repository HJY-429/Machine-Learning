# Image Classification - DNN

This directory contains comprehensive image classification implementations using traditional Deep Neural Networks (DNN). The project systematically analyzes the impact of key neural network hyperparameters including initialization methods, learning rates, optimizers, batch normalization, and dropout regularization.

## Project Overview

The DNN image classification project provides a thorough analysis of fully-connected neural network performance for image classification tasks. It benchmarks different optimization strategies and architectural choices through controlled experiments on image data.

## Key Components

### Data Processing
- **Image Dataset**: Custom image data organized in training and testing splits
- **PyTorch Framework**: Implementation using PyTorch neural network modules
- **Data Preprocessing**: Standard normalization and data preparation pipelines

### Neural Network Architecture
- **Multi-layer DNN**: Fully connected feedforward networks with hidden layers
- **Baseline Configuration**: Standard architecture for controlled experiments
- **Sequential Design**: Progressive layer structure with ReLU activations

### Hyperparameter Analysis
- **Learning Rate Studies**: Performance analysis across different learning rates
- **Optimizer Comparison**: Systematic evaluation of gradient descent variants
- **Initialization Methods**: Weight initialization impact on training dynamics
- **Batch Normalization**: Effects of normalization layers on convergence
- **Dropout Regularization**: Regularization impact with different dropout rates

### Benchmark Experiments
- **Baseline Performance**: Reference performance with standard hyperparameters
- **Learning Rate Sensitivity**: Cross-validated learning rate comparison
- **Optimizer Selection**: Choice of gradient descent algorithms
- **Regularization Effects**: Dropout and batch normalization impact

## Visualization Outputs
- `Figure BASELINE_LR0.05 EP75 HL3 SGD.pdf`: Baseline model with standard hyperparameters
- `Figure BN_use_bn.pdf`: Batch normalization effect analysis
- `Figure DROP_analysis_Dropouts.pdf`: Dropout regularization study
- `Figure INIT_Initialization.pdf`: Weight initialization impact
- `Figure OPTM_Adam LR.pdf`: Adam optimizer with different learning rates
- `Figure OPTM_RMSprop LR.pdf`: RMSprop optimizer with different learning rates
- `Figure OPTM_SGD LR.pdf`: Stochastic Gradient Descent with different learning rates
- `test.pdf`: Additional performance analysis

## Technical Implementation
- **Framework**: PyTorch deep learning framework
- **Architecture**: Multi-layer perceptron with custom layer dimensions
- **Activation Function**: ReLU for hidden layers
- **Learning Rates**: Comparative study across multiple learning rate values
- **Training Episodes**: Extended training (75 epochs) for convergence analysis
- **Hidden Layers**: Multiple hidden layer configurations

## Hyperparameter Studies
### Learning Rate Analysis
- Systematic comparison across optimizers
- Convergence behavior analysis
- Performance stability evaluation

### Regularization Effects
- **Batch Normalization**: Impact on training stability and convergence
- **Dropout Rates**: Regularization strength and generalization effects
- **Initialization Strategies**: Weight initialization effect on optimization

### Optimization Methods
- **Adam**: Adaptive moment estimation
- **RMSprop**: Root mean square propagation
- **SGD**: Stochastic gradient descent comparison

## Methodology
The project demonstrates systematic hyperparameter tuning for deep neural networks, providing insights into the sensitivity of DNN performance to various architectural and training choices. It serves as a comprehensive guide for neural network hyperparameter selection in image classification tasks.
