import torch
from pathlib import Path
from PIL import Image
import numpy as np

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device
from yolov5.utils.dataloaders import letterbox

# Charger le modèle YOLOv5
model_path = Path("models/yolov5s.pt")
device = select_device('')  # '' = CPU ou GPU si dispo
model = DetectMultiBackend(str(model_path), device=device)
model.eval()

def detect_car(image_path):
    """
    Détecte une voiture dans l'image et retourne un crop (PIL.Image) de la voiture.
    Si aucune voiture détectée, retourne l'image entière.
    """
    # Charger l'image et la convertir en numpy array
    img0 = Image.open(image_path).convert('RGB')
    img0_np = np.array(img0)

    # Prétraitement : resize + padding (letterbox)
    img, ratio, pad = letterbox(img0_np, new_shape=640)
    img = img.transpose((2, 0, 1))  # HWC → CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    img = img.unsqueeze(0)  # batch size 1

    # Inference
    pred = model(img, augment=False, visualize=False)

    # NMS : filtrer les voitures uniquement (classes 2, 5, 7)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=[2, 5, 7], agnostic=False)

    if pred[0] is not None and len(pred[0]):
        det = pred[0]

        # Redimensionner les coordonnées vers l'image d'origine
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0_np.shape).round()

        # Trier par score de confiance décroissant
        det = det[det[:, 4].argsort(descending=True)]

        # Prendre la bbox la plus confiante
        x1, y1, x2, y2 = map(int, det[0, :4])

        # Ajouter un padding de 10 % autour de la bbox
        w, h = x2 - x1, y2 - y1
        pad_w, pad_h = int(0.1 * w), int(0.1 * h)

        x1 = max(x1 - pad_w, 0)
        y1 = max(y1 - pad_h, 0)
        x2 = min(x2 + pad_w, img0.width)
        y2 = min(y2 + pad_h, img0.height)

        car_crop = img0.crop((x1, y1, x2, y2))
        return car_crop

    # Sinon, retourner image complète
    return img0
