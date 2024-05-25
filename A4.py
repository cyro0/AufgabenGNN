import numpy as np
import matplotlib.pyplot as plt

def max_pooling_2d(image, filter_size, stride):
  """
  Führt 2D Max-Pooling auf einem Bild mit der angegebenen Filtergröße und Schrittweite durch.

  Args:
    image: Das Eingabebild (zweidimensionales Array).
    filter_size: Die Größe des Pooling-Filters (z.B. 2x2).
    stride: Die Schrittweite für die Filterbewegung (z.B. 2).

  Returns:
    Das gepoolte Bild (zweidimensionales Array).
  """

  image_height, image_width = image.shape

  # Berechne die Ausgabegröße des gepoolten Bildes
  output_height = int((image_height - filter_size + 1) / stride)
  output_width = int((image_width - filter_size + 1) / stride)

  # Initialisiere das gepoolte Bild
  pooled_image = np.zeros((output_height, output_width))

  # Durchlaufe das Eingabebild in Blöcken der Filtergröße
  for y in range(0, output_height):
    for x in range(0, output_width):
      # Extrahiere den aktuellen Block aus dem Eingabebild
      block = image[y * stride:(y + filter_size) * stride,
                    x * stride:(x + filter_size) * stride]

      # Berechne den maximalen Wert im Block
      pooled_image[y, x] = np.max(block)

  return pooled_image

# Lade ein Schwarz-Weiß-Bild (ersetzen Sie 'image_path' mit dem Pfad zu Ihrem Bild)
image = plt.imread('A4.JPG')  # Annahme: Graustufenbild

# Wähle die Pooling-Parameter
filter_size = 2
stride = 2

# Führe Max-Pooling auf dem Bild durch
pooled_image = max_pooling_2d(image, filter_size, stride)

# Visualisiere das gepoolte Bild
plt.imshow(pooled_image, cmap='gray')
plt.title('Gepooltes Bild')
plt.show()
