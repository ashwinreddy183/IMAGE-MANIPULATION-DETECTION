 MVSSNet: Multi-View Spatial-Spectral Network

This repository implements **MVSSNet**, a deep learning framework for image feature extraction and segmentation using ResNet50, Sobel edge detection, and Bayar convolution-based constrained filtering.

---

## Features
- **ResNet50 Backbone** for hierarchical feature extraction
- **Sobel Filtering** for edge detection
- **BayarConv2d** for constrained noise extraction
- **ERB Blocks** for enhanced residual processing
- **Position & Channel Attention Modules**
- Supports **RGB and grayscale images**
- Pretrained **ImageNet weights** available

---

## Installation
```bash
git clone <repo_url>
cd <repo_folder>
pip install torch torchvision Cython numpy pillow matplotlib scikit-image opencv-python
Usage
python
Copy code
import torch
from mvssnet import get_mvss, rgb2gray

# Initialize model
model = get_mvss(
    backbone='resnet50',
    pretrained_base=True,
    nclass=1,
    sobel=True,
    n_input=3,
    constrain=True
)

# Forward pass
input_tensor = torch.randn(1, 3, 224, 224)
res1, x0 = model(input_tensor)

print("Edge Feature Map Shape:", res1.shape)
print("Segmentation Output Shape:", x0.shape)
Modules Overview
Sobel Module: Detects edges in feature maps

BayarConv2d: Constrained convolution for high-pass noise extraction

ERB: Residual blocks with convolution for enhanced features

ResNet50: Backbone network

Position & Channel Attention: Captures spatial & channel dependencies

_DAHead: Dual-attention segmentation head

Notes
Toggle Sobel and Bayar constraints using sobel=True/False and constrain=True/False.

Input images should be normalized if using pretrained weights.

forward returns:

res1: Feature map after ERB + Sobel

x0: Segmentation map resized to input

References
He et al., Deep Residual Learning for Image Recognition, CVPR 2016

Bayar & Stamm, Constrained Convolutional Layer for Image Manipulation Detection, 2016

Position and Channel Attention Modules for segmentation
