import cv2
import time
from ultralytics import YOLO


# Cargar un modelo ligero (Nano)
model = YOLO('/home/pi/Documents/pi_test/UNO_FINAL.pt')

# Configurar la captura de video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Ventana redimensionable
cv2.namedWindow('Deteccion en Tiempo Real', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Deteccion en Tiempo Real', 800, 600)

prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar inferencia con confianza optimizada
    results = model(frame, conf=0.5)


    # Dibujar y mostrar los resultados
    annotated_frame = results[0].plot()
    cv2.imshow('Deteccion en Tiempo Real', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
