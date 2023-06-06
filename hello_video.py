import cv2

# From file
cap = cv2.VideoCapture("images/tomato01.mp4")

# From Webcam
# cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("End of VIDEO")
        break

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()

