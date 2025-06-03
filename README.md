## 🧠 Deep Convolutional Neural Network (DCNN) - 52 Layer Architecture

This MATLAB project implements a custom **52-layer deep CNN** designed for **binary classification** of chest X-ray images into `NORMAL` and `PNEUMONIA`. The network is trained and evaluated using a curated dataset of 200×200 resized X-ray images, with automated preprocessing.

### 🗂️ Dataset Structure

Dataset/
└── new/
├── train/
│ ├── NORMAL/
│ └── PNEUMONIA/
└── test/
├── NORMAL/
└── PNEUMONIA/


### 🧼 Preprocessing Steps
- Resize images to 200x200x3
- Normalize pixel values (Min-Max)
- Apply CLAHE (contrast enhancement)
- Gaussian filtering to remove noise

### 🧠 CNN Architecture
- 52 convolutional layers (3×3 filters, 64 feature maps)
- Batch normalization and ReLU after each
- Max pooling every 13 layers
- Fully connected layer (512 units) + dropout
- Final classification layer with softmax

### ✅ Performance
- **Test Accuracy**: 94.69%

