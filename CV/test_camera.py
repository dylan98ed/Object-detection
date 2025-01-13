from picamera2 import Picamera2
import cv2

# Inicia Picamera2
picam2 = Picamera2()
picam2.start()

while True:
    frame = picam2.capture_array()
    cv2.imshow("frame", frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()