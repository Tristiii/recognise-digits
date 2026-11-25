# Recognize Digits
A machine learning model that recognizes written digits.

## 1. Introduction

From the beginning of neural networks, one of the most basic problems of machine learning is recognizing handwriting, in particular digits. This has become so popular, that the MNIST-database was created, which contains a lot of handwritten digits to train models on.

This model aims to achieve a precise classification of this dataset.

## 2. Methods

### 2.1 Dataset

This model is trained on the mnist_784 dataset from OpenML via scikit-learn, found in the [following link](https://www.openml.org/search?type=data&sort=runs&id=554&status=active).

Distribution of the dataset:
- Training: 60.000 samples.
- Testing: 10.000 samples.

### 2.2 Model

The LeNet convolutional neural network (CNN) was applied on the dataset (LeCun et al., 1998). The PyTorch library was used to implement this neural network.

## 3. Results

The model obtains an accuracy of 99,3%, a precision of 99,3%, recall of 99,3%, F1-score of 0,993 and an AUROC of 1.000.