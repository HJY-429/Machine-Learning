# Machine Learning Algorithms and Implementations

This directory contains comprehensive machine learning implementations for educational purposes, covering fundamental algorithms, deep learning basics, and systematic performance analysis. Each module provides hands-on experience with core ML concepts through practical implementation.

## Code Subdirectory - Core Implementations

The `code/` directory contains structured implementations organized by algorithm type and complexity. Each module includes educational implementations with detailed documentation, practical exercises, and comparative analysis.

### 🧮 Fundamental Algorithms

#### [poly_regression/](code/poly_regression/)
**Polynomial Regression and Learning Curve Analysis**
- `polyreg.py`: Complete polynomial regression implementation with polynomial feature expansion
- `linreg_closedform.py`: Closed-form solution for linear regression using normal equations
- `polyreg.py` includes customizable degree polynomial fitting with L2 regularization
- Learning curve generation showing bias-variance tradeoff
- Systematic evaluation of model complexity vs performance

#### [ridge_regression_mnist/](code/ridge_regression_mnist/)
**Ridge Regression on MNIST Dataset**
- `ridge_regression.py`: Ridge regression for handwritten digit classification
- MNIST feature extraction and preprocessing pipeline
- Regularization parameter tuning and cross-validation
- Performance evaluation on 10-class classification problem

#### [log_regression/](code/log_regression/)
**Binary Logistic Regression**
- `binary_log_regression.py`: Complete logistic regression with gradient-based optimization
- `binary_log_regression_1.py`: Alternative implementation with different optimization strategies
- Sigmoid activation and negative log-likelihood loss
- Gradient descent implementation with convergence criteria

### 🧠 Deep Learning Foundations

#### [intro_pytorch/](code/intro_pytorch/)
**PyTorch Basics and Neural Network Components**
- **Layers**: Custom implementations of fundamental neural network layers
  - `linear.py`: Fully connected layer with weight initialization
  - `relu.py`: ReLU activation function
  - `sigmoid.py`: Sigmoid activation function
  - `softmax.py`: Softmax activation for multi-class outputs

- **Loss Functions**: Essential loss functions for different tasks
  - `losses/MSE.py`: Mean Squared Error (MSE) loss
  - `losses/CrossEntropy.py`: Cross-entropy loss for classification

- **Optimizers**: Basic optimization algorithms
  - `optimizers/SGD.py`: Stochastic Gradient Descent implementation
  - `train.py`: Comprehensive training loop for model optimization

#### [neural_network_mnist/](code/neural_network_mnist/)
**Multi-layer Neural Network for MNIST Classification**
- `main.py`: Complete neural network implementation from scratch
- Custom F1 neural network class with backpropagation
- Weight initialization using uniform distribution
- Forward pass and backward pass implementation
- MNIST digit classification with hidden layers

### 📊 Statistical Learning Methods

#### [kernel_bootstrap/](code/kernel_bootstrap/)
**Bootstrap Sampling and Kernel Methods**
- `main.py`: Bootstrap implementation for uncertainty quantification
- Kernel density estimation and non-parametric statistical analysis
- Confidence interval construction through resampling
- Application to machine learning model evaluation

#### [lasso/](code/lasso/)
**Least Absolute Shrinkage and Selection Operator (LASSO)**
- `ISTA.py`: Iterative Shrinkage-Thresholding Algorithm for LASSO
- `crime_data_lasso.py`: LASSO regression on crime dataset for feature selection
- L1 regularization for sparse model selection
- Coordinate descent and proximal gradient methods

#### [vanilla_vs_numpy/](code/vanilla_vs_numpy/)
**Performance Comparison and Implementation Studies**
- `vanilla_vs_numpy.py`: Benchmark comparison between pure Python and NumPy implementations
- Matrix multiplication and vector operations performance analysis
- Algorithm optimization demonstration using vectorized operations
- Educational example of computational efficiency considerations

#### [clt_with_cdfs/clt_with_cdfs.py](code/clt_with_cdfs/clt_with_cdfs.py)
**Central Limit Theorem Demonstration**
- Empirical verification of Central Limit Theorem
- Cumulative distribution function analysis
- Sampling distributions and convergence properties
- Statistical theory verification through simulation

## 🔧 Technical Utilities

### Utilities Package (`utils/`)
- `load_data.py`: Standardized data loading functions for consistent dataset access
- `tag_decorator.py`: Problem tagging system for homework assignment organization
- `sst_problem_util.py`: Custom utility functions for specific problem types
- `lower_level_utils.py`: Low-level utility functions and helpers

### Development Tools
- **setup.py**: Package configuration for `cse446utils` distribution
- **environment.yaml**: Conda environment specification for reproducible setup
- **tasks.py**: Task automation and project management utilities
- **tests/**: Comprehensive test suite with public API validation

### Results Directory (`results/`)
- Organized output storage for analysis results and intermediate outputs
- Generated plots, model performance metrics, and evaluation summaries

### Data Directory (`data/`)
- Sample datasets and input files for algorithm demonstrations
- Standardized data format for consistent processing across modules

### Tests Directory (`tests/`)
- Some test samples to identify if each part of the function is correct

## 📚 Educational Focus

This machine learning codebase emphasizes understanding through implementation:

- **Algorithm Fundamentals**: Core ML algorithms implemented from scratch
- **Hands-on Learning**: Complete implementations with detailed documentation
- **Comparison Studies**: Performance analysis between different approaches
- **Real-world Applications**: Practical implementations on standard datasets
- **Practical Insights**: Understanding computational considerations and optimization

Each module includes comprehensive testing through the `tests/` directory, ensuring correctness and reliability of implementations while serving as additional educational examples.

All implementations follow consistent coding standards and include comprehensive documentation for educational use and practical application.
