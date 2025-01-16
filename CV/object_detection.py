import cv2
import torch

class ObjectDetectionProcessor:

    def __init__(self):
        # Cargar el modelo YOLOv5 nano
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)
        # Dimensiones reales promedio de una señal de stop (en metros)
        self.real_width = 0.06  # Ejemplo: 6 cm de ancho real
        self.focal_length = 480  # Focal length en píxeles (deberías calibrar tu cámara)

    def calculate_distance(self, bbox_width):
        """
        Calcula la distancia a la señal de stop usando la relación entre el ancho real y el ancho en la imagen.

        Args:
            bbox_width (int): Ancho de la caja delimitadora en píxeles.

        Returns:
            float: Distancia a la señal de stop en metros.
        """
        distance_m = (self.real_width * self.focal_length) / bbox_width
        distance_cm = distance_m * 100  # Convertir a centímetros
        return distance_cm

    def process_image(self, cv_image):
        """
        Procesa la imagen de entrada para detectar solo señales de stop y muestra la distancia a ellas.

        Args:
            cv_image (np.ndarray): Imagen de entrada.

        Returns:
            np.ndarray: Imagen procesada con cuadros delimitadores y distancias.
        """
        # Inferencia con YOLOv5
        pred = self.model(cv_image)
        
        # Convertir predicciones a un DataFrame y filtrar por confianza
        df = pred.pandas().xyxy[0]
        df = df[(df["confidence"] > 0.5) & (df["name"] == "stop sign")]

        # Dibujar las cajas de detección en el frame
        for i in range(df.shape[0]):
            bbox = df.iloc[i][["xmin", "ymin", "xmax", "ymax"]].values.astype(int)
            bbox_width = bbox[2] - bbox[0]
            distance_cm = self.calculate_distance(bbox_width)
            
            cv2.rectangle(cv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(cv_image,
                        f"Stop Sign: {round(df.iloc[i]['confidence'], 4)}, Dist: {round(distance_cm, 2)}cm",
                        (bbox[0], bbox[1] - 15),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 255, 255),
                        2)

        return cv_image
