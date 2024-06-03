
import torch
from fastapi import UploadFile
from datetime import datetime
import shutil


from utils.video_processing import detect_knives

model = torch.hub.load('ultralytics/yolov8',
                       'yolov8n')  # Asegurarse de que el modelo esté entrenado para detectar cuchillos


async def process_video(file: UploadFile):
    # Guardar el archivo de video temporalmente
    video_path = f"data/{file.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Procesar el video y detectar cuchillos
    detected, processed_video_path = detect_knives(video_path, model)

    # Registrar la detección (aquí puede agregar la lógica para guardar en la base de datos)
    detection = {
        "timestamp": datetime.now().isoformat(),
        "detected": detected,
        "video_path": processed_video_path
    }

    return detection
