import os
import json


def is_detection_label(label_path):
    """
    Check if the label file follows the detection format.
    """
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            return False
        try:
            float_parts = [float(x) for x in parts[1:]]
        except ValueError:
            return False
    return True


def is_segmentation_label(label_path):
    """
    Check if the label file follows the segmentation format (assuming COCO JSON format).
    """
    try:
        with open(label_path, 'r') as f:
            data = json.load(f)
        if "annotations" in data and "segmentation" in data["annotations"][0]:
            return True
    except (json.JSONDecodeError, KeyError, IndexError):
        return False
    return False


def check_dataset_labels(label_dir):
    """
    Check all label files in the directory to determine if they are for detection or segmentation.
    """
    detection_labels = []
    segmentation_labels = []

    for root, _, files in os.walk(label_dir):
        for file in files:
            label_path = os.path.join(root, file)
            if file.endswith('.txt') and is_detection_label(label_path):
                detection_labels.append(label_path)
            elif file.endswith('.json') and is_segmentation_label(label_path):
                segmentation_labels.append(label_path)

    return detection_labels, segmentation_labels


# Directorio de etiquetas
label_dir = 'C:/Users/Niro/Documents/UPC Academico/Taller de Proyecto/model/datasets/dataset/images/val/labels'

detection_labels, segmentation_labels = check_dataset_labels(label_dir)

print(f"Detección: {len(detection_labels)} archivos")
print(f"Segmentación: {len(segmentation_labels)} archivos")