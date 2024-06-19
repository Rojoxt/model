from ultralytics import YOLO
import torch

if __name__ == "__main__":
    # Habilitar cuDNN benchmark y cudnn deterministic
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    model = YOLO('models/yolov8n.pt')  # Carga el modelo preentrenado

    #Inicio de entrenamiento con los parámetros establecidos
    model.train(data='../data.yaml', epochs=100, imgsz=(640), batch=16, optimizer='Adam')

    #El código de abajo es para validar con el valor conf de 0 a 0.9 y manualmente hallar el AP (Precisión promedio)
    #model = YOLO('runs/detect/train/weights/best.pt')  # Carga el modelo entrenado
    # results = model.train(data='../data.yaml', epochs=100, imgsz=(640), batch=16)
    #results = model.val(data='../data.yaml', conf=0)
    #results = model.val(data='../data.yaml', conf=0.1)
    #results = model.val(data='../data.yaml', conf=0.2)
    #results = model.val(data='../data.yaml', conf=0.3)
    #results = model.val(data='../data.yaml', conf=0.4)
    #results = model.val(data='../data.yaml', conf=0.5)
    #results = model.val(data='../data.yaml', conf=0.6)
    #results = model.val(data='../data.yaml', conf=0.7)
    #results = model.val(data='../data.yaml', conf=0.8)
    #results = model.val(data='../data.yaml', conf=0.9)
