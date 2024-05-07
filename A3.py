import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt

x, y, x_test, y_test = mnist.mnist()

#(x_train, _), (x_test, _) = mnist.load_data()

# # Normalisieren der Daten auf Werte zwischen 0 und 1
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255

# # Anzahl der sichtbaren Einheiten (entspricht der Anzahl der Pixel in einem MNIST-Bild)
# n_visible = 784

# # Anzahl der versteckten Einheiten
# n_hidden = 128

# # Initialisierung der Gewichte und Biases
# W = tf.Variable(tf.random.normal((n_visible, n_hidden), mean=0.0, stddev=0.01))
# v_bias = tf.Variable(tf.zeros(shape=(n_visible,)))
# h_bias = tf.Variable(tf.zeros(shape=(n_hidden,)))

# def gibbs_step(v):
#   # Aktivierung der versteckten Einheiten
#   h = tf.nn.sigmoid(tf.matmul(v, W) + h_bias)

#   # Rekonstruktion der sichtbaren Einheiten
#   v_prime = tf.nn.sigmoid(tf.matmul(h, tf.transpose(W)) + v_bias)

#   # Erneute Aktivierung der versteckten Einheiten
#   h_prime = tf.nn.sigmoid(tf.matmul(v_prime, W) + h_bias)

#   return v_prime, h_prime

# # Funktion zum Ausführen von n Gibbs-Sampling-Schritten
# def gibbs_sampling(v, n_steps):
#   for _ in range(n_steps):
#     v, h = gibbs_step(v)
#   return v

# # Optimierer
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# # Trainingsloop
# for epoch in range(10):
#   for i in range(len(x_train)):
#     # Rekonstruktion der Eingabe
#     v_prime = gibbs_sampling(x_train[i], n_steps=10)

#     # Rekonstruktionsfehler berechnen
#     reconstruction_error = tf.reduce_mean(tf.square(v_prime - x_train[i]))

#     # Gradienten berechnen und Parameter aktualisieren
#     with tf.GradientTape() as tape:
#       loss = reconstruction_error
#     gradients = tape.gradient(loss, [W, v_bias, h_bias])
#     optimizer.apply_gradients(zip(gradients, [W, v_bias, h_bias]))

#   # Testen der Rekonstruktionsleistung
#   reconstructed_digits = gibbs_sampling(x_test, n_steps=10)
#   reconstruction_error = tf.reduce_mean(tf.square(reconstructed_digits - x_test))
#   print(f"Epoche {epoch}: Rekonstruktionsfehler: {reconstruction_error.numpy()}")

# # Rekonstruieren Sie einige Testbilder und visualisieren Sie sie
# reconstructed_digits = gibbs_sampling(x_test[:10], n_steps=10)

# import matplotlib.pyplot as plt

# for i in range(10):
#   plt.imshow(reconstructed_digits[i].reshape(28, 28))
#   plt.show()
