import os
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2

model = tf.lite.Interpreter(model_path='hand_gesture_recognition.lite')
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()

class_names = ['fist', 'five', 'okay', 'two']
conf_threshold = 0.5  # puedes ajustarlo

cap = cv2.VideoCapture(0)  # 0 = webcam por defecto

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

print("Pulsa 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer un frame de la cámara")
        break

    # frame: BGR uint8, tamaño libre
    # 1) redimensionar a 320x320
    frame_resized = cv2.resize(frame, (320, 320))

    # 2) pasar a RGB + normalizar a [0,1]
    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    image = img_rgb.astype(np.float32) / 255.0
    sample = np.expand_dims(image, axis=0)  # (1, 320, 320, 3)

    # 3) inferencia
    model.set_tensor(input_details[0]['index'], sample)
    model.invoke()
    output = model.get_tensor(output_details[0]['index'])  # (1, 2100, 8)
    detections = output[0]  # (2100, 8)

    # 4) mejor detección
    class_scores = detections[:, 4:]              # (2100, 4)
    confs = np.max(class_scores, axis=1)          # (2100,)
    class_ids = np.argmax(class_scores, axis=1)   # (2100,)

    best_idx = int(np.argmax(confs))
    best_conf = float(confs[best_idx])
    best_class_id = int(class_ids[best_idx])

    # Solo dibujamos si pasa el umbral
    if best_conf >= conf_threshold:
        # 5) decodificar bbox como [x_min, y_min, x_max, y_max] normalizados
        x1_n, y1_n, x2_n, y2_n = detections[best_idx, :4]

        img_h, img_w = 320, 320  # frame_resized size

        x1 = int(x1_n * img_w)
        y1 = int(y1_n * img_h)
        x2 = int(x2_n * img_w)
        y2 = int(y2_n * img_h)

        # clamp
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        label = f"{class_names[best_class_id]} ({best_conf:.2f})"

        # Dibujamos sobre el frame redimensionado (BGR)
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_resized, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 6) mostrar ventana
    cv2.imshow("Hand gesture detection", frame_resized)

    # salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()