# Image Segmentation using GrabCut

This repository contains a Python script for image segmentation using the GrabCut algorithm. GrabCut is an iterative algorithm that separates the foreground and background of an image based on user-defined rectangular regions.

## Requirements
- Python 3.x
- NumPy
- OpenCV
- Matplotlib

## Installation
```bash
pip install numpy opencv-python matplotlib
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Run the script:
   ```bash
   python grabcut_segmentation.py
   ```

## Script Overview
```python
import numpy as np 
import cv2 
from matplotlib import pyplot as plt

# Load the image
imgnom = cv2.imread('woman-Palestine.jpg') 

# Display the original image
plt.imshow(imgnom)

# Convert the image to RGB format
img = cv2.cvtColor(imgnom, cv2.COLOR_BGR2RGB)
plt.imshow(img)

# Create an initial mask
mask = np.zeros(img.shape[:2], np.uint8) 

# Create background and foreground models for GrabCut
backgroundModel = np.zeros((1, 65), np.float64) 
foregroundModel = np.zeros((1, 65), np.float64)

# Define a rectangular region for GrabCut
rectangle = (200, 210, 600, 800)

# Apply GrabCut algorithm
cv2.grabCut(img, mask, rectangle, backgroundModel, foregroundModel, 3, cv2.GC_INIT_WITH_RECT) 

# Generate the final mask
mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8') 

# Apply the mask to the original image
image = img * mask2[:, :, np.newaxis]

# Display the segmented image with a colorbar
plt.imshow(image) 
plt.colorbar() 
plt.show()
```

## Explanation
- The script loads an image and converts it to RGB format.
- It initializes a mask and creates background and foreground models.
- A rectangular region is defined for GrabCut.
- GrabCut algorithm is applied to segment the image.
- The final mask is generated and applied to the original image.
- The segmented image is displayed with a colorbar.

Feel free to modify the script and experiment with different images and parameters to observe the segmentation results.
