# Detect Pixelated Image and Correct it

## Introduction

This repository contains project focused on detecting pixelated images using MobileNetV2 and correcting them using SRCNN (Super-Resolution Convolutional Neural Network). The models are designed to enhance image quality for various applications.

## Pixelated Image Detection (MobileNetV2)

### Overview
The detection model utilizes MobileNetV2 to classify images as pixelated or high-resolution. It's efficient and suitable for deployment in resource-constrained environments.

### Dataset

The dataset used can be found on [Kaggle](https://www.kaggle.com/datasets/aleenasaj/pix-og)
The dataset consists of two folders:
- `Pixelated`: Contains pixelated images.
- `Original`: Contains high-resolution images.

### Model Training
- **Architecture**: MobileNetV2 with custom dense layers for binary classification.
- **Training**: Compiled with Adam optimizer and binary cross-entropy loss over 10 epochs.

## Pixelated Image Correction (SRCNN)

### Overview
The correction model uses SRCNN to convert pixelated images to high-resolution versions, improving image clarity and detail.

### Dataset
The dataset used can be found on [Kaggle](https://www.kaggle.com/datasets/aleenasaj/image-pro)
The dataset includes:
- `Original`: High-resolution images.
- `Pixelated`: Corresponding pixelated images.

### Model Architecture
- **SRCNN**: Three convolutional layers designed for super-resolution tasks.

### Training
- **Objective**: Minimizing Mean Squared Error (MSE) loss using Adam optimizer.

### Inference
- Trained model can be used to enhance new pixelated images.

## Team Members
The Team members of this project:
- J Jenolin Jeba
- Aleena Saji
- Prajusha
- Jenolin Esther
- Bettina

And we thank our college mentor Dr. M.Rajeswari and the mentors from Intel Unnati Industry for their support and guidance.

## Future Work
- Need improvement in correcting the quality of image.
- Implement additional evaluation metrics such as SSIM.
- Deploy models for broader applications.
