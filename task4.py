import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to enhance vibrance
def enhance_vibrance(saturation_channel, a, sigma=70):
    enhanced_sat = saturation_channel + a * 128 * np.exp(-((saturation_channel - 128) ** 2) / (2 * sigma ** 2))
    # Clip values to be in the range [0, 255] and convert to uint8 to match original image type
    enhanced_sat = np.clip(enhanced_sat, 0, 255).astype(np.uint8)
    return enhanced_sat

# Load the image
img = cv2.imread('C:\\Users\spram\Documents\IPNMV\images\spider.png')

# Check if the image was loaded successfully
if img is None:
    print("Error: Image not found or path is incorrect")
else:
    # Convert the image to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_img)  # Split into Hue, Saturation, and Value channels

    # Apply vibrance enhancement to the Saturation channel
    a = 0.5  # Adjust the value of 'a' to control the enhancement strength
    enhanced_S = enhance_vibrance(S, a)

    # Recombine the enhanced Saturation channel with the original Hue and Value channels
    enhanced_hsv = cv2.merge([H, enhanced_S, V])

    # Convert back to BGR color space for displaying
    enhanced_img = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    # Display the original and vibrance-enhanced images
    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Vibrance-enhanced image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
    plt.title('Vibrance Enhanced Image')
    plt.axis('off')

    plt.show()
