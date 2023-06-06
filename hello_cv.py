import cv2

image = cv2.imread("images/cb_gray.png", cv2.IMREAD_UNCHANGED)

print(f"Image size:{image.shape}")
print(f"Image size:{image.dtype}")
print(image)

cv2.namedWindow("Display",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Display",500,500)
cv2.imshow("Display",image)

cv2.waitKey(0)
cv2.destroyAllWindows()
