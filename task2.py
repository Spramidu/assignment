import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the brain image
img = cv2.imread('C:\\Users\\spram\\Documents\\IPNMV\\images\\brain_proton_density_slice.png', cv2.IMREAD_GRAYSCALE)

# Function to accentuate white matter (high intensity)
def enhance_white_matter(image):
    white_matter_img = np.where(image > 150, 255, image)  # Example threshold for white matter
    return white_matter_img

# Function to accentuate gray matter (mid-range intensity)
def enhance_gray_matter(image):
    gray_matter_img = np.where((image >= 100) & (image <= 150), image * 1.5, image)  # Example range
    return gray_matter_img

# Enhance white and gray matter
white_matter_img = enhance_white_matter(img)
gray_matter_img = enhance_gray_matter(img)

# Plot the original, white matter enhanced, and gray matter enhanced images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Brain Image')

plt.subplot(1, 3, 2)
plt.imshow(white_matter_img, cmap='gray')
plt.title('White Matter Enhanced')

plt.subplot(1, 3, 3)
plt.imshow(gray_matter_img, cmap='gray')
plt.title('Gray Matter Enhanced')

plt.show()
