import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()
x_train = (x_train / 255).reshape(-1, 784)
x_test = (x_test / 255).reshape(-1, 784)

def sigmoid(z, d=False):
    return sigmoid(z) * (1 - sigmoid(z)) + 1e-12 if d else 1 / (1 + np.exp(-z))

def relu(z, d=False):
    return (z > 0)+1e-12 if d else  z * (z > 0)

layers = [
    # activation, shape:(out,in)
    {"act":relu, "shape":(1024,784)},
    {"act":relu, "shape":(50,1024)},
    {"act":relu, "shape":(1024,50)},
    {"act":sigmoid, "shape":(784,1024)}
]

# global properties
layerslength = len(layers)
errors = []
epochs = 5

# adam properties
learningrate = 0.002
beta1 = 0.9
beta2 = 0.999
momentum_term, uncorrected_moving_avg_gradient, momentum_term_2, uncorrected_moving_avg_sqaured_gradient = {},{},{},{}

# layer properties
activation, weight, biases, function = {},{},{},{}
for i, layer in zip(range(1, layerslength+1), layers):
    n_out, n_in = layer["shape"]
    function[i] = layer["act"]

    # Xavier Initialization of weights
    weight[i] = np.random.randn(n_out, n_in) / n_in**0.5
    biases[i], momentum_term_2[i], uncorrected_moving_avg_sqaured_gradient[i] = [np.zeros((n_out,1)) for i in [1,2,3]]
    momentum_term[i], uncorrected_moving_avg_gradient[i] = [np.zeros((n_out, n_in)) for i in [1,2]]

for t in range(1, epochs+1):
    # Train
    for batch in np.split(x_train, 30):

        # Forward pass
        activation[0] = batch.T
        for i in range(1,layerslength+1):
            activation[i] = function[i]((weight[i] @ activation[i-1]) + biases[i])
            
        # Backpropagation
        backprop_activation, backprop_weight, backprop_biases = {},{},{}
        for i in range(1,layerslength+1)[::-1]:
            backprop_error = weight[i+1].T @ backprop_activation[i+1] if layerslength-i else 0.5*(activation[layerslength]-activation[0])
            backprop_activation[i] = backprop_error * function[i](activation[i],d=1)
            backprop_weight[i] = backprop_activation[i] @ activation[i-1].T
            backprop_biases[i] = np.sum(backprop_activation[i], 1, keepdims=True)

        # Adam updates
        def adam(moving_avg, momentum, weight_bias, backprop, i):
            moving_avg[i] = beta1 * moving_avg[i] + (1 - beta1) * backprop[i]
            momentum[i] = beta2 * momentum[i] + (1 - beta2) * backprop[i]**2
            moving_avg_hat = moving_avg[i] / (1. - beta1**t)
            momentum_hat = momentum[i] / (1. - beta2**t) 
            weight_bias[i] -= learningrate * moving_avg_hat / (momentum_hat**0.5 + 1e-12)
            
        for i in range(1,layerslength+1):
            adam(uncorrected_moving_avg_gradient, momentum_term, weight, backprop_weight, i)
            adam(uncorrected_moving_avg_sqaured_gradient, momentum_term_2, biases, backprop_biases, i)

    # Validate
    activation[0] = x_test.T
    for i in range(1,layerslength+1):
        activation[i] = function[i]((weight[i] @ activation[i-1]) + biases[i])
    errors += [np.mean((activation[layerslength]-activation[0])**2)]
    print("Val loss - ", errors[-1])

prediction = []
activation[0] = x_train[:20].T

#forward pass
for i in range(1,layerslength+1):
    activation[i] = function[i](weight[i] @ activation[i-1] + biases[i])
prediction = activation[layerslength]

plt.figure(figsize=(20,5))

for i in range(20):
    plt.subplot(3, 20, i + 1)
    plt.imshow(x_train[i].reshape(28,28), cmap="gray")
    plt.axis("off")

for i in range(20):
    plt.subplot(3, 20, i + 1 + 20)
    plt.imshow(activation[layerslength-2].T[i].reshape(5,-1), cmap="gray")
    plt.axis("off")
    
for i in range(20):
    plt.subplot(3, 20, i + 1 + 40)
    plt.imshow(prediction.T[i].reshape(28,28), cmap="gray") 
    plt.axis("off")

plt.show()
