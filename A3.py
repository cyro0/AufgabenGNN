import numpy as np
import tensorflow as tf
import pygame
import random
import threading

# Loading MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Autoencoder network architecture (vis 0 will be declared in 'train_network' function as the input layer)
vis1 = np.zeros((784))
hid0 = np.zeros((100))
hid1 = np.zeros((100))

# Global properties
learning_rate = 0.01
c = np.random.rand(100)
b = np.random.rand(784)
weights = np.random.rand(100, 784) 

# Sigmoid activation function
def sigmoid(x):
    return np.divide(1, np.add(np.exp(-x, x), 1, x), x) # this calculation of sigmoid prevents overflow
   
def train_network(input):
    vis0 = input.flatten() # Inputlayer

    # Positive Phase
    np.matmul(weights, vis0, hid0)
    np.add(hid0, c, hid0)
    sigmoid(hid0)

    # Negative Phase
    np.matmul(weights.T, hid0, vis1)
    np.add(vis1, b, vis1)
    sigmoid(vis1)

    # Positive Phase again
    np.matmul(weights, vis1, hid1)
    np.add(hid1, c, hid1)
    sigmoid(hid1)
    
    # Update weights
    deltaW = learning_rate * (np.outer(hid0, vis0) - np.outer(hid1, vis1))
    deltaC = learning_rate * (hid0 - hid1)
    deltaB = learning_rate * (vis0 - vis1)
    np.add(weights, deltaW, weights)
    np.add(c, deltaC, c)
    np.add(b, deltaB, b)

# Processing function for pygame and reconstruction
def process(input):
    proc_vis0 = input.flatten()
    proc_vis1 = np.zeros((784))
    proc_hid0 = np.zeros((100))

    # Poitive Phase
    np.matmul(weights, proc_vis0, proc_hid0)
    np.add(proc_hid0, c, proc_hid0)
    sigmoid(proc_hid0)

    # Reconstruction
    np.matmul(weights.T, proc_hid0, proc_vis1)
    np.add(proc_vis1, b, proc_vis1)
    sigmoid(proc_vis1)

    return proc_vis1

# Convert grayscale to 3D rgb array
def gray(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret

# Train network with input from MNIST
def train(n: int):
    for i in range(n):
        input = random.choice(x_test)
        train_network(input)

# Pygame initialization
pygame.init()
screen = pygame.display.set_mode((560, 280))
clock = pygame.time.Clock()

train_thread = threading.Thread(target=train, args=(100_000,))
train_thread.start()

first_iteration = True

# Original and Reconstructed Image output with Pygame
while True:
    screen.fill((20, 20, 20))
    input = random.choice(x_test)
    output = process(input)
    screen.blit(pygame.transform.scale(pygame.pixelcopy.make_surface(gray((input.transpose() * 255).astype(np.uint8))), (280, 280)), (0, 0))
    screen.blit(pygame.transform.scale(pygame.pixelcopy.make_surface(gray((output.reshape(28, 28).transpose() * 255).astype(np.uint8))), (280, 280)), (280, 0))
    pygame.display.update()
    clock.tick(0.5)