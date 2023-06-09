import numpy as np
import cv2
import glob

chessboardSize = (8,5)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('images/calib*.png')

print(f"objpoints:{len(objpoints)} , imgpoints:{len(imgpoints)}")
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv2.imshow('img', img)
    cv2.waitKey(1)
cv2.destroyAllWindows()
print(f"objpoints:{len(objpoints)} , imgpoints:{len(imgpoints)}")

# calculate destortion matrix
ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# save to file
np.save(open("cameraMatrix.npz", "wb" ),cameraMatrix)
np.save(open("dist.npz", "wb" ),dist)