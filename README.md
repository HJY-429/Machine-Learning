# Machine Learning Projects Repository

This repository contains comprehensive machine learning implementations and research projects spanning computer vision, signal processing, and deep learning applications. Each project demonstrates different aspects of machine learning theory and practical implementation.

## Repository Structure

### Foundational Projects

#### [machine_learning/](machine_learning/)
**Machine Learning Algorithms from Scratch**
- Systematic implementation of fundamental ML algorithms including polynomial regression, logistic regression, and ridge regression
- Complete neural network construction with custom layers (Linear, ReLU, Sigmoid, Softmax), loss functions (MSE, Cross-Entropy), and optimizers (SGD)
- Advanced statistical learning methods: LASSO regression with ISTA optimization, bootstrap sampling for uncertainty quantification
- Performance benchmarking studies: vanilla Python vs NumPy implementations for computational efficiency analysis
- Educational focus on mathematical foundations, algorithm behavior analysis, and practical implementation considerations

### Computer Vision Projects

#### [MNIST_digit_classification/](MNIST_digit_classification/)
**Handwritten Digit Recognition using PCA and Ridge Classification**
- Implements dimensionality reduction on the classic MNIST dataset
- Compares 2D PCA projections across different digit pairs (1 vs 8, 2 vs 7, 3 vs 8)
- Analyzes cumulative energy of singular values for optimal dimensionality selection
- Visualizes first 16 principal component modes capturing digit variation patterns

#### [Image_classification_CNN/](Image_classification_CNN/)
**Convolutional Neural Network Architecture Analysis**
- Systematic comparison of CNN vs Fully Connected Networks (FCN)
- Hyperparameter studies on kernel sizes (3x3 vs 5x5) and channel configurations (8, 16, 32, 64)
- Network depth analysis (2-layer vs 3-layer CNN architectures)
- Feature map visualization for understanding learned representations
- PyTorch implementation with 100K iteration training

#### [Image_classification_DNN/](Image_classification_DNN/)
**Deep Neural Network Hyperparameter Optimization**
- Comprehensive analysis of learning rate effects across different optimizers (Adam, RMSprop, SGD)
- Systematic evaluation of batch normalization impact on training convergence
- Dropout regularization analysis with varying dropout rates
- Weight initialization method comparison and their training effects
- Benchmark studies on optimizer learning rate sensitivity

### Signal Processing Projects

#### [Finding_submarine/](Finding_submarine/)
**3D Sonar Signal Processing and Submarine Detection**
- Advanced frequency domain analysis using Fast Fourier Transform (FFT)
- 3D signal processing on 64×64×64 grid data for underwater detection
- Dominant frequency identification and 3D trajectory visualization
- Spectral analysis with k-space coordinates (KX, KY, KZ)
- Multi-dimensional frequency domain visualization using isosurface rendering
- Submarine tracking through acoustic signature analysis

#### [Captured_motion_recognition/](Captured_motion_recognition/)
**Human Motion Recognition using PCA and Classification**
- Motion capture data analysis for walking, jumping, and running movements
- Principal Component Analysis for motion pattern dimensionality reduction
- Classification comparison between centroid and k-NN classifiers
- 2D and 3D PCA projection analysis
- Cumulative explained variance analysis across different motion types
- Motion signature extraction and pattern recognition

### Additional Projects

#### [Code_Data_Analysis/](Code_Data_Analysis/)
Data analysis projects focused on code repositories, programming patterns, and software metrics analysis.

#### [pytorch_example/](pytorch_example/)
PyTorch deep learning implementations and examples for various applications.

## Technical Implementation

### Frameworks and Libraries
- **PyTorch**: Deep learning framework for neural network implementations
- **Scikit-learn**: Traditional machine learning algorithms and preprocessing
- **NumPy/Matplotlib**: Numerical computing and visualization
- **SciPy**: Signal processing and FFT implementations
- **Plotly**: Interactive 3D visualizations for complex data

### Methodology
Each project follows a systematic approach:
1. **Data Processing**: Loading, preprocessing, and validation
2. **Model Development**: Algorithm implementation and architecture design
3. **Parameter Analysis**: Hyperparameter tuning and optimization studies
4. **Performance Evaluation**: Validation, testing, and comparative analysis
5. **Visualization**: Comprehensive plotting and analysis results

## Key Research Areas
- **Dimensionality Reduction**: PCA applications in computer vision and signal processing
- **Computer Vision**: CNN architectures and their comparative analysis
- **Signal Processing**: FFT-based frequency domain analysis for pattern detection
- **Machine Learning**: Traditional ML algorithms vs deep learning approaches
- **Data Visualization**: Multi-dimensional data representation and analysis

## Getting Started

1. **Install Dependencies**: Each project may have specific requirements detailed in individual README files
2. **Data Preparation**: Some projects use local datasets while others use standard datasets like MNIST
3. **Run Analysis**: Execute notebooks or Python scripts to reproduce results
4. **Visualize Results**: Generated analysis plots and performance metrics

## Project Structure
Each subdirectory includes:
- **README.md**: Detailed project description and methodology
- **Core Implementation**: Python scripts or Jupyter notebooks
- **Analysis Results**: Generated plots, visualizations, and performance metrics
- **Documentation**: Comprehensive analysis of findings and technical details

Each project provides standalone implementations with educational focus, suitable for understanding machine learning concepts and their practical applications.
