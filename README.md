# Neural Network for Handwritten Digit Classification

This project implements a simple feedforward neural network to classify handwritten digits from the **MNIST dataset**. The neural network uses **ReLU activation** in the hidden layer and **softmax activation** in the output layer to predict digit classes (0-9).

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model Overview](#model-overview)
- [Functions](#functions)
- [Training](#training)



## Introduction

This project builds a two-layer neural network from scratch using **NumPy** and **Pandas** for handling the dataset. The model is trained using the gradient descent optimization technique and backpropagation. The model accuracy improves over a specified number of epochs, and after training, predictions on individual images can be visualized.

## Installation

1. Clone this repository:
    ```bash
    git clone <repo_url>
    ```
2. Install the required Python packages:
    ```bash
    pip install numpy pandas matplotlib
    ```

## Usage

1. Prepare your dataset: 
    - Download the MNIST training data (`train.csv`) and save it in the project folder. This dataset can be downloaded from [Kaggle MNIST Dataset](https://www.kaggle.com/datasets/animatronbot/mnist-digit-recognizer).

2. Run the script:
    ```bash
    python NeuralNetwork.py
    ```
3. The script will train the model, print accuracy every 10 epochs, and show visualized predictions for selected test samples.

## Model Overview

- **Input Layer**: The input layer accepts a flattened 28x28 grayscale image (784 features).
- **Hidden Layer**: Contains 10 neurons with **ReLU activation**.
- **Output Layer**: Contains 10 neurons (for the 10 digit classes) with **Softmax activation** to output probability scores for each class.

## Functions

### 1. `init_params()`
   Initializes the weights and biases for the network.

### 2. `ReLU(val)`
   Applies the **ReLU activation function**.

### 3. `softmax(x)`
   Applies the **softmax function** for the output layer.

### 4. `forward_prop(w1, b1, w2, b2, X)`
   Performs forward propagation to compute the activations of the hidden and output layers.

### 5. `one_hot(a)`
   Converts class labels into one-hot encoded vectors for backpropagation.

### 6. `back_prop(Z1, A1, Z2, A2, W2, X, Y)`
   Computes the gradients of weights and biases using backpropagation.

### 7. `update_params(W1, b1, W2, b2, dW1, dB1, dW2, dB2, alpha)`
   Updates the parameters using the computed gradients and a learning rate (`alpha`).

### 8. `get_predictions(A2)`
   Returns the predicted class for each input based on the output activations.

### 9. `get_Accuracy(predictions, Y)`
   Calculates the accuracy of predictions compared to true labels.

### 10. `Gradient_descent(X, Y, epochs, alpha)`
   Trains the neural network using gradient descent and backpropagation.

### 11. `make_predictions(X, W1, b1, W2, b2)`
   Returns predictions for new input data.

### 12. `test_prediction(index, W1, b1, W2, b2)`
   Tests the model’s prediction on a specific training image and visualizes the result.

## Training

Training is done via gradient descent over a fixed number of epochs. The accuracy of the model is printed every 10 epochs. The model is trained on the MNIST dataset using the following steps:

- Data normalization: The pixel values (0-255) are scaled to the range (0-1).
- Loss minimization: The cross-entropy loss is minimized using backpropagation.
- Parameters are updated using a learning rate of `0.1` (modifiable).

Source : https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras

