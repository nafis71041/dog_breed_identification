# Dog Breed Identification Project

An elite deep learning project that leverages state-of-the-art transfer learning techniques to achieve robust dog breed classification.

## Table of Contents
- [Overview](#overview)
- [Motivation and Objectives](#motivation-and-objectives)
- [Model Architectures](#model-architectures)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Training Pipeline](#training-pipeline)
- [Performance Metrics](#performance-metrics)
- [Usage Instructions](#usage-instructions)
- [Technical Requirements](#technical-requirements)
- [Future Work](#future-work)
- [License](#license)
- [Citation](#citation)

## Overview

This project applies two state-of-the-art deep learning architectures for dog breed classification:
- **MobileNetV2**: A highly efficient CNN optimized for mobile and real-time applications.
- **Vision Transformer (ViT-L/16)**: A transformer-based model that captures global context for superior image classification.

Both models are fine-tuned using advanced transfer learning techniques to balance performance and computational efficiency.

## Motivation and Objectives

The primary goal of this project is to build a robust neural network system capable of accurately classifying dog breeds from images. The objectives include:
- Leveraging pre-trained models to reduce training time and improve accuracy.
- Implementing advanced data augmentation techniques to enhance model generalization.
- Comparing the performance trade-offs between lightweight and transformer-based models.
- Providing insights through detailed performance metrics and visualizations.

## Model Architectures

### MobileNetV2

- **Pre-training**: Utilizes weights pre-trained on ImageNet.
- **Custom Layers**: Incorporates a global average pooling layer, a dropout layer (0.5), and a custom classification head specific to dog breeds.
- **Input Specifications**: Accepts images resized to 224x224x3.

### Vision Transformer (ViT-L/16)

- **Pre-training**: Employs a large-scale transformer pre-trained on extensive image datasets.
- **Customization**: Features a tailor-made classification head and leverages frozen transformer layers for stability.
- **Input Specifications**: Designed for images resized to 224x224x3.

## Dataset & Preprocessing

The dataset is structured with training images in JPEG format alongside breed labels in CSV format. Data preprocessing steps include:
- **Resizing**: All images are resized to 224x224.
- **Normalization**: Images are normalized using ImageNet statistics.
- **Augmentation**:
  - *MobileNetV2*: Random flips, rotations (20%), contrast, and brightness adjustments (20% each).
  - *ViT*: Random horizontal flips (60% probability), fixed rotation (30°), center cropping, and normalization.

## Training Pipeline

### MobileNetV2 Training

- **Phases**:
  1. Initial training with a frozen base model.
  2. Fine-tuning of the last few layers for enhanced performance.
- **Optimization**: Early stopping (patience=5) and learning rate scheduling (initial: 1e-3, fine-tuning: 1e-7).
- **Loss Function**: Sparse Categorical Crossentropy.

### Vision Transformer Training

- **Approach**: Single-phase training focusing on the classification head while keeping transformer layers frozen.
- **Optimizer**: SGD with momentum (lr=0.009, momentum=0.9).
- **Batch Sizes**: 8 for training, 32 for validation.
- **Epochs**: 7.

## Performance Metrics

Model performance is evaluated using multiple metrics including:
- Accuracy, Precision, Recall, and F1 Score.
- Inference time per sample.
- Confusion matrix analysis.
- Monitoring of training and validation loss/accuracy for early detection of overfitting.

## Usage Instructions

1. **Clone the Repository**: Ensure you have access to all project files.
2. **Download Pre-trained Models**: Retrieve models from [Model Link](https://drive.google.com/drive/folders/1raQQMfTGoZ3dw4wnLMVGELwBDrq66mH7).
3. **Install Dependencies**: Refer to the Technical Requirements below.
4. **Select Notebook**: Choose the appropriate Jupyter notebook:
   - `mobilenet_train.ipynb` for MobileNetV2 training.
   - `vit_train.ipynb` for Vision Transformer training.
   - `vit_test.ipynb` for model evaluation.

## Technical Requirements

### Python Dependencies

```
tensorflow
torch
torchvision
pandas
numpy
matplotlib
seaborn
scikit-learn
pillow
```

### Hardware Requirements

- GPU recommended for training.
- Minimum 8GB RAM.
- CUDA support required for PyTorch (ViT).

## Future Work

Potential enhancements include:
1. Developing ensemble methods combining both models for improved accuracy.
2. Experimenting with additional architectures such as EfficientNet and ResNet.
3. Implementing hyperparameter tuning strategies.
4. Exploring model quantization for faster inference on edge devices.
5. Building an API for streamlined integration into production systems.

## License

This project is licensed under the [MIT License](#LICENSE).

## Citation

If you use this project in your research, please cite:

> NAFIS ISLAM (2025). Dog Breed Identification using Deep Learning Architectures. Journal of Machine Learning Research.
