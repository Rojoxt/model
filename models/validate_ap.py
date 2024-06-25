from ultralytics import YOLO

# Cargar el modelo YOLOv8
model = YOLO('best.pt')

# Definir el threshold
threshold = 0.1

# Ruta al archivo de configuración del dataset
data_config = '../data.yaml'

# Validar el modelo en el conjunto de datos especificado
results = model.val(data=data_config, conf=threshold)

# Imprimir los resultados de la validación
print(results)
