import cv2
import numpy as np
import matplotlib.pyplot as plt

# Nearest-Neighbor Interpolation
def nearest_neighbor_zoom(image, scale_factor):
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    zoomed_image = np.zeros((new_h, new_w), dtype=image.dtype)

    for i in range(new_h):
        for j in range(new_w):
            # Find the nearest pixel in the original image
            src_x = int(i / scale_factor)
            src_y = int(j / scale_factor)
            zoomed_image[i, j] = image[src_x, src_y]
    
    return zoomed_image

# Bilinear Interpolation
def bilinear_zoom(image, scale_factor):
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    zoomed_image = np.zeros((new_h, new_w), dtype=image.dtype)

    for i in range(new_h):
        for j in range(new_w):
            # Find coordinates of the pixel in the original image
            x = i / scale_factor
            y = j / scale_factor

            # Find the surrounding four pixels
            x1 = int(np.floor(x))
            y1 = int(np.floor(y))
            x2 = min(x1 + 1, h - 1)
            y2 = min(y1 + 1, w - 1)

            # Compute the weights
            wx = x - x1
            wy = y - y1

            # Bilinear interpolation
            top = (1 - wy) * image[x1, y1] + wy * image[x1, y2]
            bottom = (1 - wy) * image[x2, y1] + wy * image[x2, y2]
            zoomed_image[i, j] = (1 - wx) * top + wx * bottom
    
    return zoomed_image

# Function to compute the normalized sum of squared differences (SSD)
def compute_ssd(image1, image2):
    return np.sum((image1 - image2) ** 2) / (image1.size)

# Load the image
img = cv2.imread('C:\\Users\\spram\\Documents\\IPNMV\\images\\a1q8images\\im03small.png', cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if img is None:
    print("Error: Image not found or path is incorrect")
else:
    # Apply Nearest-Neighbor Interpolation
    scale_factor = 4
    zoomed_nearest = nearest_neighbor_zoom(img, scale_factor)

    # Apply Bilinear Interpolation
    zoomed_bilinear = bilinear_zoom(img, scale_factor)

    # Compute SSD by comparing the zoomed images with the original scaled back
    ssd_nearest = compute_ssd(cv2.resize(zoomed_nearest, img.shape[::-1]), img)
    ssd_bilinear = compute_ssd(cv2.resize(zoomed_bilinear, img.shape[::-1]), img)

    # Plot the results
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    # Nearest-Neighbor Zoomed image
    plt.subplot(1, 3, 2)
    plt.imshow(zoomed_nearest, cmap='gray')
    plt.title(f'Nearest-Neighbor Zoom (SSD: {ssd_nearest:.4f})')

    # Bilinear Zoomed image
    plt.subplot(1, 3, 3)
    plt.imshow(zoomed_bilinear, cmap='gray')
    plt.title(f'Bilinear Zoom (SSD: {ssd_bilinear:.4f})')

    plt.show()
