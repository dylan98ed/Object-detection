import cv2

cap = cv2.VideoCapture('/dev/video0')

while cap.isOpened():
    status, frame = cap.read()
    if not status:
        print("Error: No se puede acceder a la cámara.")
        break

    cv2.imshow("frame", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
