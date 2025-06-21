# 🧠 Brain Tumor Detection using CNN

This project uses **Convolutional Neural Networks (CNN)** to detect brain tumors from MRI images. The model is built using **TensorFlow** and **Keras** and trained on a labeled dataset of MRI scans.

---

## 🚀 Project Overview

Brain tumor detection is a crucial step in early diagnosis and treatment. This deep learning model automates tumor detection using MRI images, which can significantly assist medical professionals in making faster and more accurate decisions.

---

## 📁 Dataset

The dataset used in this project contains MRI images categorized into two classes:

- **Yes** – Brain tumor present
- **No** – No tumor

Each class has been divided into training and testing sets.

> Dataset Source: Included in the repository under the `/Training` and `/Testing` folders.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV

---

## 📊 Model Architecture

The model uses a simple CNN architecture:

- Convolutional Layers
- MaxPooling Layers
- Dropout for regularization
- Dense layers
- Softmax for classification

```python
model = Sequential([
    Conv2D(...),
    MaxPooling2D(...),
    ...
    Dense(2, activation='softmax')
])
