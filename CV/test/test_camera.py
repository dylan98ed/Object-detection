from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
picam2.start()

# Define el codec y crea el objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))  # Ajusta la resolución según tus necesidades

while True:
    frame = picam2.capture_array()
    out.write(frame)  # Escribe cada frame en el archivo de video

    #cv2.imshow('Frame', frame)  # Muestra el frame en pantalla

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

picam2.stop()
out.release()
cv2.destroyAllWindows()
