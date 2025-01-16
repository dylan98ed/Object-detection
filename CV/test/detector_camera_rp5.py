import cv2
import torch
from picamera2 import Picamera2

# Cargar el modelo YOLOv5
model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)

def detector():
    # Inicializar la cámara de Raspberry Pi
    picam2 = Picamera2()
    picam2.start()

    while True:
        # Capturar un frame de la cámara
        frame = picam2.capture_array()

        # Inferencia con YOLOv5
        pred = model(frame)
        
        # Convertir predicciones a un DataFrame y filtrar por confianza
        df = pred.pandas().xyxy[0]
        df = df[df["confidence"] > 0.5]

        # Dibujar las cajas de detección en el frame
        for i in range(df.shape[0]):
            bbox = df.iloc[i][["xmin", "ymin", "xmax", "ymax"]].values.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame,
                        f"{df.iloc[i]['name']}: {round(df.iloc[i]['confidence'], 4)}",
                        (bbox[0], bbox[1] - 15),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 255, 255),
                        2)

        # Mostrar el frame procesado
        cv2.imshow("frame", frame)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detector()
