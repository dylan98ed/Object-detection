import cv2
import torch

model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

def detector():
    
    cap = cv2.VideoCapture('/dev/video0')  # Dirección de la cámara de la Raspberry Pi
    

    while cap.isOpened():
        status, frame = cap.read()          #frame es un array numpy por ej (1080,1920,3)
        
        if not status:
            break
        
        # Inferencia
        pred = model(frame)
        # xmin,ymin,xmax,ymax
        df = pred.pandas().xyxy[0]
        # Filtrar por confidence
        df = df[df["confidence"] > 0.5]

        for i in range(df.shape[0]):
            bbox = df.iloc[i][["xmin","ymin","xmax","ymax"]].values.astype(int) #para los bbox el tipo de dato tiene que ser entero
            
            #print bboxes: frame -> (xmin,ymin), (xmax,ymax), color = azul (255,0,0), grosor de linea = 2
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),(255,0,0),2)

            #print text
            cv2.putText(frame,
                        f"{df.iloc[i]['name']}: {round(df.iloc[i]['confidence'],4)}", # nombre de la calse y la confianza redondeada a 4 decimales
                        (bbox[0], bbox[1] - 15),    # posicion del texto
                        cv2.FONT_HERSHEY_PLAIN,     # tipo de letra
                        1,                          # grosor de la letra
                        (255,255,255),              # color
                        2)                          # tamaño

        cv2.imshow("frame", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()

if __name__ == '__main__':
    detector()

