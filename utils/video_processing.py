import cv2
import torch


def detect_knives(video_path: str, model):
    cap = cv2.VideoCapture(video_path)
    detected = False
    processed_video_path = f"processed_{video_path}"

    # Procesamiento de video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(processed_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        if results.pred[0].size(0) > 0:  # Si hay detecciones
            detected = True
            # Dibujar cuadros alrededor de los objetos detectados
            for *box, conf, cls in results.pred[0].cpu().numpy():
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    return detected, processed_video_path
