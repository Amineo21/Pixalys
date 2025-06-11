import torch
import torch.nn as nn
import torchvision  # Ajoute cette ligne
from torchvision import models, transforms
from PIL import Image
import json

# Préparer le device (GPU si dispo)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger noms des classes
with open("class_name.json", "r") as f:
    class_names = json.load(f)
class_keys = sorted(class_names.keys())  

num_classes = len(class_names)
model = torchvision.models.resnet50(num_classes=num_classes)
checkpoint = torch.load("models/resnet50_compcars.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Transformation de l'image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(  # Normalisation standard ImageNet, utile si ton modèle a été entraîné avec
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def classify_car(image: Image.Image) -> str:
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
    key = class_keys[predicted.item()]  # Utilise l'indice comme index dans la liste des clés
    return class_names[key]