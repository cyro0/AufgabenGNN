import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Laden der MNIST-Daten
(x_train, _), (x_test, _) = mnist.load_data()

# Normalisieren der Daten
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28*28))
x_test = np.reshape(x_test, (len(x_test), 28*28))

# Definition der RBM-Klasse
class RBM(object):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.W = tf.Variable(tf.random.normal([input_size, output_size], stddev=0.01))
        self.h_bias = tf.Variable(tf.zeros([output_size]))
        self.v_bias = tf.Variable(tf.zeros([input_size]))

    def prob_h_given_v(self, visible, W, h_bias):
        return tf.nn.sigmoid(tf.matmul(visible, W) + h_bias)

    def prob_v_given_h(self, hidden, W, v_bias):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(W)) + v_bias)

    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs))))

    def gibbs_sampling(self, visible):
        h_sample = self.sample_prob(self.prob_h_given_v(visible, self.W, self.h_bias))
        v_sample = self.sample_prob(self.prob_v_given_h(h_sample, self.W, self.v_bias))
        return v_sample

    def train(self, X, lr=0.1, k=1):
        positive_grad = tf.matmul(tf.transpose(X), self.prob_h_given_v(X, self.W, self.h_bias))
        for i in range(k):
            hidden = self.sample_prob(self.prob_h_given_v(X, self.W, self.h_bias))
            visible = self.sample_prob(self.prob_v_given_h(hidden, self.W, self.v_bias))
            negative_grad = tf.matmul(tf.transpose(visible), self.prob_h_given_v(visible, self.W, self.h_bias))
            self.W += lr * (positive_grad - negative_grad) / tf.cast(tf.shape(X)[0], dtype=tf.float32)
            self.v_bias += lr * tf.reduce_mean(X - visible, 0)
            self.h_bias += lr * tf.reduce_mean(self.prob_h_given_v(X, self.W, self.h_bias) - self.prob_h_given_v(visible, self.W, self.h_bias), 0)

# Hyperparameter
learning_rate = 0.1
training_epochs = 50
batch_size = 64
n_visible = 28*28
n_hidden = 128

# Erstellen und trainieren der RBM
rbm = RBM(n_visible, n_hidden)
for epoch in range(training_epochs):
    for i in range(0, len(x_train), batch_size):
        batch_xs = x_train[i:i+batch_size]
        rbm.train(batch_xs, lr=learning_rate)

    # Ausgabe während des Trainings
    print("Epoch:", '%d' % (epoch+1))

# Rekonstruktion der Daten
reconstructed_imgs = []
for i in range(len(x_test)):
    reconstructed_img = rbm.gibbs_sampling(tf.reshape(x_test[i], (1, n_visible)))
    reconstructed_imgs.append(reconstructed_img)

reconstructed_imgs = np.reshape(reconstructed_imgs, (len(x_test), 28, 28))

# Beispielrekonstruktion anzeigen
import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Originalbild
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Rekonstruiertes Bild
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed_imgs[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
