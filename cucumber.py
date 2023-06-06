import cv2
import numpy as np

cucumber_color = cv2.imread('images/cucumber.png')
cv2.imshow("cucumber", cucumber_color)
cucumber_gray = cv2.cvtColor(cucumber_color, cv2.COLOR_BGR2GRAY)
_, cucumber_th = cv2.threshold(cucumber_gray, 200, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("cucumber_th", cucumber_th)
contours, _ = cv2.findContours(cucumber_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in contours:
    rect = cv2.boundingRect(cnt)
    cv2.rectangle(cucumber_color, rect, (0,255,0),2)

    if(rect[2] > 10):
        coin_type = f"length:{rect[2]}"
    else:
        coin_type = ""

    cv2.putText(cucumber_color, coin_type, (rect[0], rect[1]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255))

cv2.imshow("detected", cucumber_color)
cv2.waitKey(0)