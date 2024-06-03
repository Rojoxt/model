from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import requests
import json
import asyncio

# Inicializar FastAPI
app = FastAPI()

# Inicializar la cámara con una resolución más baja
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Detalles de la API de Ultralytics
api_url = 'https://api.ultralytics.com/v1/predict/vY89GpzxNSze7BxrYy72'
api_key = 'zJeZNa2ijEJzXmVDxTU5'

async def get_predictions(image):
    _, img_encoded = cv2.imencode('.jpg', image)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        'image': img_encoded.tobytes().hex()
    }
    response = requests.post(api_url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()
    else:
        return None

async def gen_frames():  # Generador de frames de la cámara
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Obtener predicciones de la API
            results = await get_predictions(frame)

            if results and 'predictions' in results:
                for result in results['predictions']:
                    x1, y1, x2, y2 = result['box']
                    confidence = result['confidence']
                    cls = result['class']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{cls} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Codificar el frame en formato JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # Limitar la tasa de fotogramas a 10 FPS
            await asyncio.sleep(0.1)

@app.get('/video_feed')
async def video_feed():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/")
def read_root():
    return {"message": "Welcome to the Object Detection API"}

# Ejecutar el servidor:
# uvicorn main:app --reload
