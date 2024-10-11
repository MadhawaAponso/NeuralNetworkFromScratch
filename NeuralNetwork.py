import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load the dataset from CSV
data = pd.read_csv("train.csv")

# Convert the dataframe to a numpy array
data = np.array(data)

# Get the dimensions of the data (number of rows m, and number of columns n)
m, n = data.shape

# Shuffle the data to mix the rows randomly
np.random.shuffle(data)

# Extract 1000 examples for testing and separate X (features) and y (labels)
data_test = data[0:1000].T  # Transpose the data
y_test = data_test[0]       # Labels are the first row
X_test = data_test[1:n]     # Features are the rest of the rows
X_test = X_test / 255       # Normalize the pixel values (0-255) to (0-1)

# Extract the remaining data for training
data_train = data[1000:m].T
y_train = data_train[0]      # Labels for training data
X_train = data_train[1:n]    # Features for training data
X_train = X_train / 255      # Normalize training data as well

# Initialize weights and biases
def init_params():
    W1 = np.random.rand(10, 784) - 0.5  # Weights for the first layer
    b1 = np.random.rand(10, 1) - 0.5    # Biases for the first layer
    W2 = np.random.rand(10, 10) - 0.5   # Weights for the second layer
    b2 = np.random.rand(10, 1) - 0.5    # Biases for the second layer
    return W1, b1, W2, b2

# ReLU activation function
def ReLU(val):
    return np.maximum(0.0, val)

# Softmax activation function for output layer
def softmax(x):
    return np.exp(x) / sum(np.exp(x))

# Forward propagation function
def forward_prop(w1, b1, w2, b2, X):
    Z1 = w1.dot(X) + b1       # Linear combination for first layer
    A1 = ReLU(Z1)             # Activation for the first layer (ReLU)
    Z2 = w2.dot(A1) + b2      # Linear combination for second layer
    A2 = softmax(Z2)          # Activation for the second layer (Softmax)
    
    return Z1, A1, Z2, A2     # Return intermediate results for backpropagation

# Convert labels into one-hot encoded vectors
def one_hot(a):
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    return b.T

# Derivative of ReLU function
def der_relu(Z):
    return Z > 0

# Backpropagation function to compute gradients
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)  # Convert labels to one-hot vectors
    dZ2 = A2 - one_hot_Y    # Error at output layer
    dW2 = 1 / m * dZ2.dot(A1.T)  # Gradient for W2
    dB2 = 1 / m * np.sum(dZ2)    # Gradient for b2
    dZ1 = W2.T.dot(dZ2) * der_relu(Z1)  # Backpropagation to hidden layer
    dW1 = 1 / m * dZ1.dot(X.T)   # Gradient for W1
    dB1 = 1 / m * np.sum(dZ1)    # Gradient for b1
    return dW1, dB1, dW2, dB2

# Update parameters using the gradients and learning rate
def update_params(W1, b1, W2, b2, dW1, dB1, dW2, dB2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * dB1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * dB2
    return W1, b1, W2, b2

# Get predictions by choosing the class with the highest probability
def get_predictions(A2):
    return np.argmax(A2, axis=0)

# Calculate accuracy by comparing predictions with true labels
def get_Accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

# Gradient descent function to train the model
def Gradient_descent(X, Y, epochs, alpha):
    W1, b1, W2, b2 = init_params()  # Initialize weights and biases
    for i in range(epochs):
        # Perform forward and backward propagation
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, dB1, dW2, dB2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, dB1, dW2, dB2, alpha)
        
        # Print accuracy every 10 epochs
        if i % 10 == 0:
            print("Epochs:", i)
            print("Accuracy:", get_Accuracy(get_predictions(A2), Y))
    
    return W1, b1, W2, b2

# Train the model using gradient descent
W1, b1, W2, b2 = Gradient_descent(X_train, y_train, 500, 0.1)  # Training for 500 epochs

# Function to make predictions on a new input
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

# Test and visualize the predictions for a specific index
def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]  # Extract the image data for the given index
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = y_train[index]  # True label for the given index
    
    print("Prediction:", prediction)
    print("Label:", label)
    
    # Reshape and scale the image back for visualization
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')  # Plot the image
    plt.show()

# Test predictions for a few samples
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)
