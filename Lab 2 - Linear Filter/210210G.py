import cv2
import numpy as np

# Load the image
image_path = "./road10.png"
image = cv2.imread(image_path)

# Convert the image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image_rgb.shape)

# Extract the R, G, B channels
R = image_rgb[:, :, 0]
G = image_rgb[:, :, 1]
B = image_rgb[:, :, 2]

# Convert to grayscale using the given formula
grayscale_image = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

# Enhance contrast
min_val = np.min(grayscale_image)
max_val = np.max(grayscale_image)
contrast_enhanced_image = ((grayscale_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# Save the contrast-enhanced image
cv2.imwrite("original.jpg", contrast_enhanced_image)

# Define filters
filter_A = np.array([[0, -1, -1, -1, 0],
                    [-1, 2, 2, 2, -1],
                    [-1, 2, 8, 2, -1],
                    [-1, 2, 2, 2, -1],
                    [0, -1, -1, -1, 0]])

filter_B = np.array([[1, 4, 6, 4, 1],
                    [4, 16, 24, 16, 4],
                    [6, 24, 36, 24, 6],
                    [4, 16, 24, 16, 4],
                    [1, 4, 6, 4, 1]])

filter_C = np.array([[5, 5, 5, 5, 5],
                    [5, 5, 5, 5, 5],
                    [5, 5, 5, 5, 5],
                    [5, 5, 5, 5, 5],
                    [5, 5, 5, 5, 5]])

filter_D = np.array([[0, -1, -1, -1, 0],
                    [-1, 2, 2, 2, -1],
                    [-1, 2, 16, 2, -1],
                    [-1, 2, 2, 2, -1],
                    [0, -1, -1, -1, 0]])

# Function to normalize a filter
def normalize_filter(filter_matrix):
    filter_sum = np.sum(filter_matrix)
    if filter_sum != 0:
        filter_matrix = filter_matrix / filter_sum
    return filter_matrix

# Normalize the filters
filter_A = normalize_filter(filter_A)
filter_B = normalize_filter(filter_B)
filter_C = normalize_filter(filter_C)
filter_D = normalize_filter(filter_D)

# Function to apply a filter to an image
def apply_filter(image, filter_matrix, filter_name):
    # Get the dimensions of the image and filter
    image_height, image_width = image.shape
    filter_size = filter_matrix.shape[0]
    padding_size = filter_size // 2
    padded_image = np.pad(image, padding_size, mode='constant', constant_values=0)
    output_image = np.zeros_like(image)
    
    # Apply the filter to the image
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+filter_size, j:j+filter_size]
            output_value = np.sum(region * filter_matrix)            
            output_image[i, j] = output_value
    
    # Clip the output values to be in the valid range (0 to 255) for image pixels
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    output_filename = f"{filter_name}.jpg"
    cv2.imwrite(output_filename, output_image)
    
    return output_image

# Function to compute RMS difference between two images
def compute_rms_difference(image1, image2):
    diff = np.sqrt(np.mean((image1 - image2) ** 2))
    return diff

# Apply the filters
filtered_image_A = apply_filter(contrast_enhanced_image, filter_A, "filter_A")
filtered_image_B = apply_filter(contrast_enhanced_image, filter_B, "filter_B")
filtered_image_C = apply_filter(contrast_enhanced_image, filter_C, "filter_C")
filtered_image_D = apply_filter(contrast_enhanced_image, filter_D, "filter_D")

# Compute RMS differences
rms_A = compute_rms_difference(contrast_enhanced_image, filtered_image_A)
rms_B = compute_rms_difference(contrast_enhanced_image, filtered_image_B)
rms_C = compute_rms_difference(contrast_enhanced_image, filtered_image_C)
rms_D = compute_rms_difference(contrast_enhanced_image, filtered_image_D)

# Print RMS differences
print(f"RMS Difference for Filter A: {rms_A}")
print(f"RMS Difference for Filter B: {rms_B}")
print(f"RMS Difference for Filter C: {rms_C}")
print(f"RMS Difference for Filter D: {rms_D}")
