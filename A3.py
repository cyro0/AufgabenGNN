import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

# Normalisierung der Daten auf Werte zwischen 0 und 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Anzahl der sichtbaren Neuronen (entspricht der Anzahl der Pixel in einem MNIST-Bild)
num_visible = 28

# Anzahl der verborgenen Neuronen
num_hidden = 28

# Gewichte und Biases initialisieren
W = np.random.randn(num_visible, num_hidden) * 0.01
vb = np.zeros(num_visible)
hb = np.zeros(num_hidden)

# Lernrate
learning_rate = 0.01

# Anzahl der Epochen
num_epochs = 10

# Trainingsschleife
for epoch in range(num_epochs):
  for i in range(len(x_train)):
    # Aktivierung der sichtbaren Schicht
    v = x_train[i]

    # Transponieren von v, um die Dimensionen an W anzupassen
    v = v.T

    # Aktivierung der verborgenen Schicht
    h = 1 / (1 + np.exp(-np.dot(v, W) - hb))

    # Rekonstruktion der sichtbaren Schicht
    v_prime = 1 / (1 + np.exp(-np.dot(h, W.T) - vb))

    # Fehler berechnen
    error = np.mean((v - v_prime)**2)

    # Update der Gewichte und Biases
    dW = np.dot(v.T, h) - np.dot(v_prime.T, h)
    dvb = np.mean(v - v_prime)
    dhb = np.mean(h - v_prime)

    W += learning_rate * dW
    vb += learning_rate * dvb
    hb += learning_rate * dhb

# Testen der RBM
rekonstruierte_bilder = []
for i in range(10):
  # Aktivierung der sichtbaren Schicht
  v = x_test[i]

  # Transponieren von v, um die Dimensionen an W anzupassen
  v = v.T

  # Aktivierung der verborgenen Schicht
  h = 1 / (1 + np.exp(-np.dot(v, W) - hb))

  # Rekonstruktion der sichtbaren Schicht
  v_prime = 1 / (1 + np.exp(-np.dot(h, W.T) - vb))

  # Rekonstruiertes Bild visualisieren
  rekonstruierte_bilder.append(v_prime.T)  # Transponieren von v_prime zurück

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i in range(10):
  axes[i // 5, i % 5].imshow(x_test[i], cmap='gray')
  axes[i // 5, i % 5].axis('off')

fig.suptitle('Originale MNIST-Bilder')

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i in range(10):
  axes[i // 5, i % 5].imshow(rekonstruierte_bilder[i], cmap='gray')
  axes[i // 5, i % 5].axis('off')

fig.suptitle('Rekonstruierte Bilder mit RBM')
plt.show()
