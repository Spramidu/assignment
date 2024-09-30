import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to apply Sobel filter using OpenCV
def apply_sobel_opencv(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel filter on x-axis
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel filter on y-axis
    sobel_combined = cv2.sqrt(sobelx**2 + sobely**2)      # Magnitude of gradients
    return sobel_combined

# Function to manually apply Sobel filter
def apply_sobel_manual(image):
    # Define Sobel kernels (3x3)
    sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Apply Sobel filter on x and y direction manually
    sobelx_manual = cv2.filter2D(image.astype(np.float32), -1, sobel_x_kernel)
    sobely_manual = cv2.filter2D(image.astype(np.float32), -1, sobel_y_kernel)

    # Compute the magnitude of the gradient (cast to float)
    sobel_combined_manual = cv2.sqrt(sobelx_manual**2 + sobely_manual**2)
    
    return sobel_combined_manual

# Load the image in grayscale
img = cv2.imread('C:\\Users\spram\Documents\IPNMV\images\einstein.png', cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if img is None:
    print("Error: Image not found or path is incorrect")
else:
    # Step 1: Apply Sobel filter using OpenCV
    sobel_opencv = apply_sobel_opencv(img)

    # Step 2: Apply Sobel filter manually
    sobel_manual = apply_sobel_manual(img)

    # Plot the results
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    # Sobel using OpenCV
    plt.subplot(1, 3, 2)
    plt.imshow(sobel_opencv, cmap='gray')
    plt.title('Sobel Filter (OpenCV)')

    # Sobel using manual implementation
    plt.subplot(1, 3, 3)
    plt.imshow(sobel_manual, cmap='gray')
    plt.title('Sobel Filter (Manual)')

    plt.show()
