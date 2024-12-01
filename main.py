import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from morphological_operation import create_ellipse_kernel, morphological_open, morphological_close
from finding_contours import find_contours,bounding_rect

class FireDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fire Detection")
        self.root.geometry("1120x1000")
        self.root.configure(bg='#ffffff')

        self.panel_hsv = None
        self.panel_contours = None
        self.panel_mask = None
        self.panel_thresholded = None
        self.panel_output = None

        # Adding title label
        self.title_label = tk.Label(root, text="Fire Detection Application", font=("Arial", 24), bg='#ffffff')
        self.title_label.grid(row=0, column=0, columnspan=3, pady=10, sticky="nsew")

        # Adding buttons and result label
        self.btn_load = tk.Button(root, text="Load Image", command=self.load_image, font=("Arial", 14), bg='#4CAF50',
                                  fg='white', padx=10, pady=5)
        self.btn_load.grid(row=1, column=1, padx=1, pady=(0, 1), sticky="nsew")

        self.label_result = tk.Label(root, text="", font=("Arial", 16), bg='#ffffff')
        self.label_result.grid(row=5, column=0, columnspan=3, pady=1, sticky="nsew")

        # Configure grid row and column weights to center contents
        for i in range(8):  # Assuming 8 rows/columns
            self.root.grid_rowconfigure(i, weight=1)
            self.root.grid_columnconfigure(i, weight=1)

    def load_image(self):
        try:
            image_path = filedialog.askopenfilename()
            if image_path:
                self.clear_panels()
                self.detect_fire(image_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def clear_panels(self):
        if self.panel_hsv:
            self.panel_hsv.destroy()
            self.panel_hsv = None
        if self.panel_contours:
            self.panel_contours.destroy()
            self.panel_contours = None
        if self.panel_mask:
            self.panel_mask.destroy()
            self.panel_mask = None
        if self.panel_thresholded:
            self.panel_thresholded.destroy()
            self.panel_thresholded = None
        if self.panel_output:
            self.panel_output.destroy()
            self.panel_output = None

    def detect_fire(self, image_path):
        try:
            image = cv2.imread(image_path)

            if image is None:
                messagebox.showerror("Error", "Could not load image")
                return

            # Step 1: Convert the image to the HSV color space
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Step 2: Define the color range for detecting fire (adjust as necessary)
            lower_fire = np.array([18, 50, 50], dtype=np.uint8)
            upper_fire = np.array([35, 255, 255], dtype=np.uint8)

            # Step 3: Threshold the HSV image to get only fire colors
            fire_mask1 = cv2.inRange(hsv_image, lower_fire, upper_fire)

            # Step 4: Apply morphological operations to remove noise
            kernel = create_ellipse_kernel((5, 5))
            fire_mask2 = morphological_close(fire_mask1, kernel)
            fire_mask3 = morphological_open(fire_mask2, kernel)

            # Step 5: Find contours in the mask
            contours = find_contours(fire_mask3)

            # Step 6: Draw bounding boxes around detected fire regions
            fire_detected = False
            for contour in contours:
                if cv2.contourArea(contour) > 500:
                    x, y, w, h = bounding_rect(contour) ########################################
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    fire_detected = True

            # Step 7: Display intermediate images and final result sequentially in one row
            self.display_image(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR), "HSV Image", 2, 1)
            self.display_image(fire_mask1, "Non Processed Binary Mask", 2, 2, is_gray=True)
            self.display_image(fire_mask2, "Fire Mask after Morphological Closing", 3, 0, is_gray=True)
            self.display_image(fire_mask3, "Fire Mask after Morphological Opening", 3, 1, is_gray=True)
            self.display_contours(image, contours, "Contour Detection", 3, 2)
            self.display_output_image(image, "Output Image with/without Fire Detection", 4, 1)
            self.display_input_image(image_path, "Input RGB image", 2, 0)

            if fire_detected:
                self.label_result.config(text="Fire Detected", fg='red')
            else:
                self.label_result.config(text="No Fire Detected", fg='green')


        except Exception as e:
            messagebox.showerror("Error", f"Failed to detect fire: {e}")

    def display_input_image(self, image_path, title, row, col):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (300, 180))
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        frame = tk.Frame(self.root, bg='#ffffff')
        frame.grid(row=row, column=col, padx=20, pady=10, sticky="nsew")

        label_title = tk.Label(frame, text=title, font=("Arial", 14), bg='#ffffff')
        label_title.pack(side="top", padx=10, pady=(10, 0))

        label_image = tk.Label(frame, image=image)
        label_image.image = image
        label_image.pack(side="top", padx=10, pady=(0, 10))

    def display_image(self, image, title, row, col, is_gray=False):
        image = cv2.resize(image, (300, 180))
        if is_gray:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        frame = tk.Frame(self.root, bg='#ffffff')
        frame.grid(row=row, column=col, padx=20, pady=10, sticky="nsew")

        label_title = tk.Label(frame, text=title, font=("Arial", 14), bg='#ffffff')
        label_title.pack(side="top", padx=10, pady=(10, 0))

        label_image = tk.Label(frame, image=image)
        label_image.image = image
        label_image.pack(side="top", padx=10, pady=(0, 10))

        if title == "HSV Image":
            self.panel_hsv = frame
        elif title == "Fire Mask":
            self.panel_mask = frame



    def display_contours(self, image, contours, title, row, col):
        image_with_contours = image.copy()
        cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
        image_with_contours = cv2.resize(image_with_contours, (300, 180))
        image_with_contours = Image.fromarray(image_with_contours)
        image_with_contours = ImageTk.PhotoImage(image_with_contours)

        frame = tk.Frame(self.root, bg='#ffffff')
        frame.grid(row=row, column=col, padx=20, pady=10, sticky="nsew")

        label_title = tk.Label(frame, text=title, font=("Arial", 14), bg='#ffffff')
        label_title.pack(side="top", padx=10, pady=(10, 0))

        label_image = tk.Label(frame, image=image_with_contours)
        label_image.image = image_with_contours
        label_image.pack(side="top", padx=10, pady=(0, 10))

        self.panel_contours = frame

    def display_output_image(self, image, title, row, col):
        image = cv2.resize(image, (300, 180))
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        frame = tk.Frame(self.root, bg='#ffffff')
        frame.grid(row=row, column=col, padx=20, pady=20, sticky="nsew")

        label_title = tk.Label(frame, text=title, font=("Arial", 14), bg='#ffffff')
        label_title.pack(side="top", padx=10, pady=(10, 0))

        label_image = tk.Label(frame, image=image)
        label_image.image = image
        label_image.pack(side="top", padx=10, pady=(0, 10))

        self.panel_output = frame


if __name__ == "__main__":
    root = tk.Tk()
    app = FireDetectionApp(root)
    root.mainloop()
