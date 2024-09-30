import cv2
import numpy as np
import matplotlib.pyplot as plt

# Gamma correction function
def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Load a color image
img_color = cv2.imread('C:\\Users\spram\Documents\IPNMV\images\highlights_and_shadows.jpg')

# Check if the image was loaded successfully
if img_color is None:
    print("Error: Image not found or path is incorrect")
else:
    # Convert to L*a*b* color space
    lab_image = cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab_image)

    # Apply gamma correction to the L channel
    gamma_value = 2.2  # Adjust this value to test different gamma corrections
    corrected_L = gamma_correction(L, gamma_value)

    # Merge the corrected L channel with the original A and B channels
    corrected_lab = cv2.merge([corrected_L, A, B])

    # Convert back to the BGR color space
    corrected_img = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

    # Plot the histograms of the original and gamma-corrected L channels
    plt.figure(figsize=(12, 6))

    # Original L channel histogram
    plt.subplot(1, 2, 1)
    plt.hist(L.ravel(), 256, [0, 256])
    plt.title('Original L* Histogram')

    # Gamma-corrected L channel histogram
    plt.subplot(1, 2, 2)
    plt.hist(corrected_L.ravel(), 256, [0, 256])
    plt.title('Gamma Corrected L* Histogram')

    plt.show()

    # Display the original and gamma-corrected images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB))
    plt.title('Gamma Corrected Image')

    plt.show()
