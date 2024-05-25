import numpy as np
import matplotlib.pyplot as plt

# Load Image
image = plt.imread('A4.JPG')  # Assumption: Grayscale image

# Pooling parameters
filter_size = 2
stride = 2

def max_pooling(image, filter_size, stride):
  image_height, image_width = image.shape

  # Calculate the output size of the pooled image
  output_height = int((image_height - filter_size + 1) / stride)
  output_width = int((image_width - filter_size + 1) / stride)

  # Initialize pooled image
  pooled_image = np.zeros((output_height, output_width))

  # Iterate through input image in blocks of the filter size
  for y in range(0, output_height):
    for x in range(0, output_width):
      # Extract the current block from the input image
      block = image[y * stride:(y + filter_size) * stride,
                    x * stride:(x + filter_size) * stride]

      # Calculate maximum value in the block
      pooled_image[y, x] = np.max(block)

  return pooled_image

# Perform max-pooling on the image
pooled_image = max_pooling(image, filter_size, stride)

# Visualize pooled Image
plt.imshow(pooled_image, cmap='gray')
plt.title('Pooled Image')
plt.show()
