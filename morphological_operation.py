import cv2
import numpy as np


def create_ellipse_kernel(size):
    """Create an elliptical kernel for morphological operations."""
    kernel = np.zeros((size[0], size[1]), dtype=np.uint8)
    center = (size[0] // 2, size[1] // 2)
    axes = (size[0] // 2, size[1] // 2)

    for i in range(size[0]):
        for j in range(size[1]):
            if ((i - center[0]) / axes[0]) ** 2 + ((j - center[1]) / axes[1]) ** 2 <= 1:
                kernel[i, j] = 1
    return kernel


def morphological_close(image, kernel):
    """Apply morphological closing (dilation followed by erosion) to the image."""
    dilated = cv2.dilate(image, kernel, iterations=1)
    closed = cv2.erode(dilated, kernel, iterations=1)
    return closed


def morphological_open(image, kernel):
    """Apply morphological opening (erosion followed by dilation) to the image."""
    eroded = cv2.erode(image, kernel, iterations=1)
    opened = cv2.dilate(eroded, kernel, iterations=1)
    return opened


"""
def detect_fire(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image.")
        return

    # Step 1: Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 2: Define the color range for detecting fire (adjust as necessary)
    lower_fire = np.array([18, 50, 50], dtype=np.uint8)
    upper_fire = np.array([35, 255, 255], dtype=np.uint8)

    # Step 3: Threshold the HSV image to get only fire colors
    fire_mask = cv2.inRange(hsv_image, lower_fire, upper_fire)

    # Step 4: Apply morphological operations to remove noise
    kernel = create_ellipse_kernel((5, 5))
    fire_mask = morphological_close(fire_mask, kernel)
    fire_mask = morphological_open(fire_mask, kernel)

    # Step 5: Display the results
    cv2.imshow("Original Image", image)
    cv2.imshow("HSV Image", hsv_image)
    cv2.imshow("Fire Mask", fire_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
detect_fire("./images/fire_image_1.jpg")

"""