from ultralytics import YOLO
import torch

if __name__ == "__main__":
    # Enable cuDNN benchmark and deterministic mode
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    model = YOLO('models/yolov8n.pt')  # Carga el modelo preentrenado
    results = model.train(data='../data.yaml', epochs=100, imgsz=(640), batch=16)
    #model = YOLO('../models/best.pt')  # Carga el modelo preentrenado
    #results = model.val(data='../data.yaml', conf=1.0)
    #results = model.val(data='../data.yaml', conf=0.9)
    #results = model.val(data='../data.yaml', conf=0.8)
    #results = model.val(data='../data.yaml', conf=0.7)
    #results = model.val(data='../data.yaml', conf=0.6)
    #results = model.val(data='../data.yaml', conf=0.5)
    #results = model.val(data='../data.yaml', conf=0.4)
    #results = model.val(data='../data.yaml', conf=0.3)
    #results = model.val(data='../data.yaml', conf=0.2)
    #results = model.val(data='../data.yaml', conf=0.1)
    #results = model.val(data='../data.yaml', conf=0)