# Project Title: Image Steganography with GANs

## Overview
This project implements a steganography technique using Generative Adversarial Networks (GANs) to hide secret images within cover images. The model is designed to effectively embed and extract images while maintaining the visual quality of the cover image.

## Project Structure
```
project
├── data
│   ├── __init__.py
│   ├── dataset.py
│   └── transforms.py
├── models
│   ├── __init__.py
│   ├── chase.py
│   ├── inn_block.py
│   ├── haar_wavelet.py
│   └── affine_coupling.py
├── utils
│   ├── __init__.py
│   ├── chaos_permutation.py
│   ├── losses.py
│   ├── visualization.py
│   └── checkpoint.py
├── train.py
├── config.py
└── README.md
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd project
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Prepare your dataset:
   - Place your cover images in the `data/cover` directory.
   - Place your secret images in the `data/secret` directory. Ensure that the filenames match for corresponding cover and secret images.

2. Configure the training parameters in `config.py`.

3. Run the training script:
   ```
   python train.py
   ```

## Model Description
- **SteganographyDataset**: Handles loading and preprocessing of cover and secret images.
- **CHASE**: The main model that implements the hiding and revealing processes.
- **INNBlock**: Represents a single invertible neural network block used in the model.
- **HaarWavelet**: Implements Haar wavelet transformations for image processing.
- **AffineCouplingLayer**: Implements the affine coupling layer used in the model.

## Loss Functions
- **Hiding Loss**: Measures the difference between the cover and stego images.
- **Reconstruction Loss**: Measures the difference between the original secret and the reconstructed secret.

## Visualization
The project includes functions for visualizing the training progress and comparing the cover and stego images.

## Checkpoints
Model checkpoints are saved during training to allow for resuming training or evaluating the model at different stages.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.