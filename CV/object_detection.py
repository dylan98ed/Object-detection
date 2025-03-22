import cv2
import torch
from ultralytics import YOLO

class ObjectDetectionProcessor:

    def __init__(self, yolo_path):
        # Cargar el modelo YOLOv11 personalizado
        self.model_path = yolo_path
        self.model = YOLO(self.model_path)  
        #self.model = YOLO('train_model/train/weights/best.pt')        # Ruta al modelo entrenado
        self.class_names = self.model.names # Obtener los nombres de las clases

        # Dimensiones reales promedio de una señal de stop (en metros)
        self.real_width = 0.06  # señal de stop: 6 cm de ancho real
        self.focal_length = 480  # Focal length en píxeles de webcam

    def calculate_distance(self, bbox_width):
        """ Calcula la distancia a la señal de stop """
        distance_m = (self.real_width * self.focal_length) / bbox_width
        return distance_m * 100  # Convertir a centímetros

    def process_image(self, cv_image):
        """
        Procesa la imagen de entrada para detectar señales de stop y mostrar la distancia a ellas.
        """
        # Realizar la inferencia con YOLOv11
        # Inference
        results = self.model(cv_image)

        # Get detections and draw bounding boxes
        for result in results:
            boxes = result.boxes.xyxy.numpy()
            confidences = result.boxes.conf.numpy()
            labels = result.boxes.cls.numpy().astype(int)

            valid_distance = True  # Default

            for i, bbox in enumerate(boxes):
                if confidences[i] < 0.5:
                    continue

                class_name = self.class_names[labels[i]]

                bbox = bbox.astype(int)
                bbox_width = bbox[2] - bbox[0]
                distance_cm = self.calculate_distance(bbox_width)

                bbox_color = (255, 0, 0) if distance_cm < 40 else (0, 0, 255)   #blue bbox if the distance is less than 40cm, red if it is greater
                valid_distance = distance_cm >= 40

                # Draw bbox
                cv2.rectangle(cv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, 2)

                # Draw text with name, confidence, and distance
                cv2.putText(cv_image,
                            f"{class_name}: {round(confidences[i], 4)}, Dist: {round(distance_cm, 2)}cm",
                            (bbox[0], bbox[1] - 15),
                            cv2.FONT_HERSHEY_PLAIN,
                            1,
                            (255, 255, 255),
                            2)

        return cv_image
