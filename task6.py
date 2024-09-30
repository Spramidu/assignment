import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to perform selective histogram equalization
def selective_histogram_equalization(image, mask):
    # Apply histogram equalization to the foreground only
    foreground = cv2.bitwise_and(image, image, mask=mask)
    equalized_foreground = cv2.equalizeHist(foreground)

    # Use the mask to extract the background
    background = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))

    # Combine the equalized foreground with the original background
    final_image = cv2.add(background, equalized_foreground)
    
    return final_image

# Load the grayscale image
img = cv2.imread('C:\\Users\spram\Documents\IPNMV\images\jeniffer.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if img is None:
    print("Error: Image not found or path is incorrect")
else:
    # Step 1: Threshold the image to create a mask for the foreground
    _, mask = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)  # Binary mask where 1 represents foreground

    # Step 2: Apply selective histogram equalization using the mask
    result_img = selective_histogram_equalization(img, mask)

    # Plot the original, mask, and result images
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    # Mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Foreground Mask')

    # Result image
    plt.subplot(1, 3, 3)
    plt.imshow(result_img, cmap='gray')
    plt.title('Selective Histogram Equalization')

    plt.show()
