import cv2
from object_detection import ObjectDetectionProcessor  # Asegúrate de que el archivo esté correctamente nombrado como `object_detection.py`
from ultralytics import YOLO
import os

def main():

    yolo_weights = "train_model/train/weights/best.pt"  # Nueva ruta del modelo entrenado
    yolo_path = "train_model/train/weights/best_ncnn_model"

    if not os.path.exists(yolo_path):
        model = YOLO(yolo_weights)  # Cargar el modelo desde la nueva ubicación
        model.export(format="ncnn")  # Exportar modelo a formato NCNN


    # Crear una instancia del procesador de detección de objetos
    processor = ObjectDetectionProcessor(yolo_path)

    # Cargar el video
    cap = cv2.VideoCapture("data/pista con niebla.mp4")#0)#"data/Señal de Stop.mp4")  # Puedes cambiar por `0` para usar la cámara de la notebook

    while cap.isOpened():
        status, frame = cap.read()  # Leer un frame del video

        if not status:
            break

        # Procesar el frame para la detección de objetos
        processed_frame = processor.process_image(frame)

        # Mostrar el frame procesado
        cv2.imshow("Object Detection", processed_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()