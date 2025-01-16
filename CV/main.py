import cv2
from object_detection import ObjectDetectionProcessor  # Asegúrate de que el archivo esté correctamente nombrado como `object_detection.py`

def main():
    # Crear una instancia del procesador de detección de objetos
    processor = ObjectDetectionProcessor()

    # Cargar el video
    cap = cv2.VideoCapture(2)#"data/Señal de Stop.mp4")  # Puedes cambiar por `0` para usar la cámara de la notebook

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