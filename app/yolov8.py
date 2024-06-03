from ultralytics import YOLO

def train_model():
    model = YOLO('models/yolov8n.pt')  # Carga el modelo preentrenado
    model.train(data='data.yaml', epochs=20, imgsz=(640, 640), batch=16, optimizer='Adam')
    model.save('models/best.pt')  # Guarda el modelo entrenado

if __name__ == "__main__":
    train_model()