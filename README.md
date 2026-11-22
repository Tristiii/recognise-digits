# RecogniseDigits
A machine learning model that recognises written digits.

## Introduction

This model is trained on the mnist_784 dataset from OpenML via scikit-learn, found in [this link](https://www.openml.org/search?type=data&sort=runs&id=554&status=active).

To use the model, run main.py in the src folder.

## Methods

A convolutional neural network (CNN) was applied on the dataset. The PyTorch library was used to implement the CNN, with the following hyperparameters:
- Architecture of the CNN: LeNet (LeCun et al., 1998).
- Loss function: cross entropy loss.
- Optimizer: adaptive moment estimation (Adam).
- Test size: 20%.
- Batch size: 64.
- Epochs: 25.

## Results

The model obtains an accuracy of ~98%.