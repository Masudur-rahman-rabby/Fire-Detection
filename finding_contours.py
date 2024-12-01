import cv2
import numpy as np

def find_contours(binary_image):
    """Find contours in a binary image using edge detection and contour tracing."""
    # Step 1: Use the Canny edge detector to find edges
    edges = cv2.Canny(binary_image, 100, 200)

    # Step 2: Find contours using the edges
    contours = []
    visited = np.zeros_like(binary_image)

    def trace_contour(i, j):
        contour = []
        stack = [(i, j)]
        while stack:
            x, y = stack.pop()
            if visited[x, y] == 0:
                visited[x, y] = 1
                contour.append((x, y))
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < binary_image.shape[0] and 0 <= ny < binary_image.shape[1]:
                            if edges[nx, ny] == 255 and visited[nx, ny] == 0:
                                stack.append((nx, ny))
        return contour

    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if edges[i, j] == 255 and visited[i, j] == 0:
                contour = trace_contour(i, j)
                if contour:
                    contours.append(np.array(contour))

    return contours


def bounding_rect(contour):

    # Initialize min and max coordinates
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')

    # Find the min and max x, y coordinates
    for point in contour:
        x, y = point
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y

    # Calculate width and height
    width = max_x - min_x
    height = max_y - min_y

    return min_x, min_y, width, height
