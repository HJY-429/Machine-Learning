# Finding Submarine

This directory contains submarine detection and sonar signal processing implementations using 3D frequency domain analysis and Fast Fourier Transform techniques.

## Project Overview

The project analyzes 3D sonar data to detect underwater objects (submarines) through frequency domain analysis. It processes simulated sonar signals in 3D space to identify the location and trajectory of underwater targets based on their acoustic signatures.

## Key Components

### Data Processing
- **3D Signal Processing**: Analyzes 3D sonar data arrays (64x64x64 grid) in frequency space
- **Spectral Analysis**: Applies Fast Fourier Transform (FFT) to convert spatial data to frequency domain
- **Frequency Grid Analysis**: Uses 3D frequency coordinates for comprehensive signal analysis

### Analysis and Visualization
- **Dominant Frequency Detection**: Identifies primary frequency components in 3D sonar data
- **Trajectory Analysis**: Tracks submarine movement paths in 2D and 3D space
- **Frequency Domain Visualization**: 3D isosurface plots of frequency spectrum
- **Domain Representation**: Maps spatial domain [-10, 10] with 64 grid points

### Signal Processing Techniques
- **Fast Fourier Transform (FFTN)**: 3D FFT for frequency domain conversion
- **Frequency Shift (FFTSHIFT)**: Centers frequency spectrum for analysis
- **Isosurface Rendering**: 3D visualization of frequency thresholds

## Visualization Outputs
- `Frequency Domain.pdf`: 3D visualization of frequency spectrum (KX, KY, KZ coordinates)
- `2D Trajectory graph_objs.pdf`: 2D path visualization of submarine movement
- `3D Trajectory graphs_objs.pdf`: 3D path visualization with objects
- `2D Path.pdf`: Simplified 2D trajectory analysis
- `3D Trajectory.pdf`: Clean 3D path visualization
- `Dominant Frequency.pdf`: Primary frequency component analysis

## Technical Details
- **Spatial Domain**: [-10, 10] with N=64 grid points per dimension
- **Frequency Resolution**: K-grid analysis in 3D Fourier space
- **Data Format**: Binary numpy files containing 3D sonar signal arrays
- **Visualization**: Interactive 3D plots showing frequency domain isosurfaces

## Methodology
The project demonstrates advanced signal processing techniques for underwater detection, showing how 3D FFT analysis can reveal submarine locations through their acoustic signatures in frequency space.
