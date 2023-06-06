import cv2
import numpy as np

# Load camera intrinsic parameters
cameraMatrix = np.load(open("cameraMatrix.npz","rb"))
dist = np.load(open("dist.npz","rb"))

# Load test image from your camera
img = cv2.imread('img10.png')
h,  w = img.shape[:2]

# calculate new transform matrix
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

# Undistort test image
new_img = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# Snow images
cv2.imshow("Original",img)
cv2.imshow("Transformed",new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()