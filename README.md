# 🧠 AlzMRI-Net: Classify Alzheimer's stages from MRI scans.

This project demonstrates a deep learning pipeline to classify MRI images into four stages of Alzheimer's disease using a custom model built upon a pre-trained EfficientNet-V2-L model. The dataset contains MRI scans labeled as **MildDemented**, **ModerateDemented**, **NonDemented**, and **VeryMildDemented**. The model training and evaluation are conducted using PyTorch, with mixed precision training to optimize performance.

## 📚 Table of Contents

- [🌟 Introduction](#-introduction)
- [✨ Features](#-features)
- [⚙️ Installation](#️-installation)
- [📂 Dataset](#-dataset)
- [🛠️ Training](#%EF%B8%8F-training)
- [📊 Results](#-results)
- [📝 License](#-license)

## 🌟 Introduction

The goal of this project is to classify MRI images into four different stages of Alzheimer's disease: **MildDemented**, **ModerateDemented**, **NonDemented**, and **VeryMildDemented**. We leverage transfer learning with the EfficientNet-V2-L model pre-trained on ImageNet and fine-tune it on our Alzheimer's MRI dataset. The model training and evaluation are conducted using mixed precision training to reduce memory usage and improve training speed.

## ✨ Features

- **Transfer Learning:** Utilizes the EfficientNet-V2-L model pre-trained on ImageNet, with custom layers for Alzheimer's classification.
- **Mixed Precision Training:** Reduces memory consumption and accelerates training.
- **Dataset Augmentation:** Includes random horizontal and vertical flips, rotations, color jitter, and resizing to improve model generalization.
- **Gradient Accumulation:** Handles larger batch sizes without exceeding GPU memory.
- **Custom Model Architecture:** A custom model built upon EfficientNet-V2-L with additional fully connected layers tailored to Alzheimer's classification.
- **Training, Validation, and Test Phases:** Clearly separated phases for training, validation, and testing to monitor performance.

## ⚙️ Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.7+
- PyTorch
- Torchvision
- Other dependencies: `numpy`, `matplotlib`, `tqdm`, `scikit-learn`, `gc`

## 📂 Dataset

The dataset contains MRI scans labeled into four classes related to Alzheimer's disease progression:

- **MildDemented**
- **ModerateDemented**
- **NonDemented**
- **VeryMildDemented**

It is divided into training, validation, and test sets:

- **Train Directory:** Contains the images used for training the model.
- **Validation:** A portion (15%) of the training dataset is set aside for validation to monitor model performance and avoid overfitting.
- **Test Directory:** Contains the images used for testing the model.

You can download the dataset from Kaggle using the following link: [Medical Scan Classification Dataset](https://www.kaggle.com/datasets/arjunbasandrai/medical-scan-classification-dataset).

## 🛠️ Training

The model is trained using a PyTorch script that includes functions for:

- **Loading and Transforming Data:** Data augmentation techniques such as flipping, rotation, and color jitter are applied, and images are resized to 256x256 pixels.
- **Custom Model Definition:** A custom model is built by adding fully connected layers to the EfficientNet-V2-L model.
- **Training Loop:** Implements gradient accumulation for better memory management and mixed precision training to optimize speed and resource usage.
- **Validation:** A separate validation set, comprising 15% of the training data, is used to evaluate the model after each epoch to monitor validation accuracy and loss.
- **Saving Best Model:** The model with the best validation accuracy is saved.

## 📊 Results

- The best model weights based on validation accuracy are saved during training.
- After training, the model was evaluated on the test set to determine its classification accuracy.
- **Test Accuracy:** The model achieved an accuracy of **99.19%** on the test set.
- **Confusion Matrix, Precision, Recall, F1 Score, Specificity, and AUC:** Detailed evaluation metrics, including a confusion matrix, precision, recall, F1 score, specificity, and AUC, were computed.

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
