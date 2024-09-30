import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the image (ensure the file path is correct)
img = cv2.imread('C:/Users/spram/Documents/IPNMV/images/emma.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if img is None:
    print("Error: Image not found or path is incorrect")
else:
    # Step 2: Define the intensity transformation (example: square function)
    def intensity_transform(pixel_value):
        return (pixel_value / 255) ** 2 * 255

    # Apply the transformation to the entire image
    transformed_img = np.array([intensity_transform(pixel) for pixel in img.flat]).reshape(img.shape)

    # Step 3: Display the original and transformed images
    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')  # Hide axes

    # Transformed image
    plt.subplot(1, 2, 2)
    plt.imshow(transformed_img, cmap='gray')
    plt.title('Transformed Image')
    plt.axis('off')  # Hide axes

    # Show the result
    plt.show()
