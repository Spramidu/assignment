import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to apply GrabCut segmentation
def grabcut_segmentation(image):
    # Initialize mask
    mask = np.zeros(image.shape[:2], np.uint8)
    
    # Define the rectangle around the object (adjust this based on your image)
    rect = (50, 50, image.shape[1] - 50, image.shape[0] - 50)
    
    # Allocate memory for the background and foreground models
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # Apply GrabCut
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create a mask where foreground pixels are marked with 1 and background with 0
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Extract the segmented foreground
    foreground = cv2.bitwise_and(image, image, mask=mask2)
    
    return foreground, mask2

# Function to blur the background
def blur_background(image, mask):
    # Apply Gaussian blur to the entire image
    blurred_image = cv2.GaussianBlur(image, (21, 21), 0)
    
    # Use the mask to keep the foreground sharp and the background blurred
    # Mask is 1 for foreground, so we invert it to get 0 for foreground, 1 for background
    background = cv2.bitwise_and(blurred_image, blurred_image, mask=1-mask)
    foreground = cv2.bitwise_and(image, image, mask=mask)
    
    # Combine the blurred background and sharp foreground
    final_image = cv2.add(background, foreground)
    
    return final_image

# Load the image
img = cv2.imread('C:\\Users\spram\Documents\IPNMV\images\daisy.jpg')

# Check if the image was loaded successfully
if img is None:
    print("Error: Image not found or path is incorrect")
else:
    # Step 1: Apply GrabCut segmentation
    segmented_foreground, mask = grabcut_segmentation(img)

    # Step 2: Blur the background
    final_image = blur_background(img, mask)

    # Plot the results
    plt.figure(figsize=(18, 6))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Segmented foreground
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(segmented_foreground, cv2.COLOR_BGR2RGB))
    plt.title('Segmented Foreground')
    plt.axis('off')

    # Final image with blurred background
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    plt.title('Blurred Background')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
