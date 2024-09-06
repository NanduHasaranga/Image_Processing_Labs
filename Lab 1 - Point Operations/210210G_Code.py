import matplotlib.pyplot as plt
import numpy as np
import cv2

# Create a 3x2 grid of subplots with a specified figure size
fig, axes = plt.subplots(3, 2, figsize=(10, 8))
image_path = "210210G_SrcImage.jpg"

# Load the image using OpenCV and convert it to RGB
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Extract the R, G, and B channels
R, G, B = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]

# Convert the image to grayscale using the luminosity method
grayscale_image = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

# Brighten the image by adding 20% of the maximum pixel value
brightness_increase = int(255 * 0.2)
brightened_image = np.clip(grayscale_image + brightness_increase, 0, 255).astype(np.uint8)

# Calculate the current min and max pixel values
current_min, current_max = grayscale_image.min(), grayscale_image.max()

# Define the new contrast range
new_min, new_max = 125, 175

# Normalize and adjust contrast
normalized_image = (grayscale_image - current_min) / (current_max - current_min)
contrast_reduced_image = np.clip(normalized_image * (new_max - new_min) + new_min, new_min, new_max).astype(np.uint8)

# Reduce to 4 bits per pixel (4bpp)
four_bpp_image = (contrast_reduced_image // 16) * 16

# Create a mirrored version of the grayscale image
mirror_image = np.fliplr(grayscale_image)

# Create the negative image
negative_image = 255 - grayscale_image

# Plot and display the images
axes[0, 0].imshow(grayscale_image, cmap="gray")
axes[0, 0].set_title("Grayscale image")
axes[0, 1].imshow(negative_image, cmap="gray")
axes[0, 1].set_title("Negative image")
axes[1, 0].imshow(brightened_image, cmap="gray")
axes[1, 0].set_title("Brightened image")
axes[1, 1].imshow(contrast_reduced_image, cmap="gray")
axes[1, 1].set_title("Contrast reduced image")
axes[2, 0].imshow(four_bpp_image, cmap="gray")
axes[2, 0].set_title("4bpp image")
axes[2, 1].imshow(mirror_image, cmap="gray")
axes[2, 1].set_title("Mirror image")

# Save the processed images
images = [
    grayscale_image,
    negative_image,
    brightened_image,
    contrast_reduced_image,
    four_bpp_image,
    mirror_image,
]

# Save each image to a file
for idx, img in enumerate(images):
    cv2.imwrite(f"210210G_OPImage_{idx}.jpg", img)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
