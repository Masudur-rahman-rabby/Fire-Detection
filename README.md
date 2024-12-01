# Fire-Detection
This Python project implements a fire detection system using digital image processing techniques. It analyzes static images to identify fire-like regions based on their distinctive color and shape characteristics. The process involves:  

1. Preprocessing: 
   - Convert the image to a suitable color space (e.g., HSV or YCbCr) for effective color segmentation.  
   - Apply smoothing techniques to reduce noise.  

2. Fire Detection:  
   - Use color thresholding to isolate regions resembling fire based on specific color ranges.  
   - Perform edge detection and contour analysis to determine the shape and size of the detected regions.  

3. Verification: 
   - Validate potential fire regions by analyzing features like roundness and intensity.  
   - Filter out false positives using predefined rules.  

The system is lightweight, avoids machine learning, and is suitable for real-time applications in monitoring and safety systems.
