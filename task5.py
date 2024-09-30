import cv2
import matplotlib.pyplot as plt

# Function for histogram equalization
def equalize_histogram(image):
    return cv2.equalizeHist(image)

# Load a grayscale image
img = cv2.imread('C:\\Users\spram\Documents\IPNMV\images\shells.tif', cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if img is None:
    print("Error: Image not found or path is incorrect")
else:
    # Apply histogram equalization
    equalized_img = equalize_histogram(img)

    # Plot the histograms of the original and equalized images
    plt.figure(figsize=(12, 6))

    # Histogram of the original image
    plt.subplot(2, 2, 1)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.title('Original Histogram')

    # Original image
    plt.subplot(2, 2, 2)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    # Histogram of the equalized image
    plt.subplot(2, 2, 3)
    plt.hist(equalized_img.ravel(), 256, [0, 256])
    plt.title('Equalized Histogram')

    # Equalized image
    plt.subplot(2, 2, 4)
    plt.imshow(equalized_img, cmap='gray')
    plt.title('Equalized Image')

    plt.show()
