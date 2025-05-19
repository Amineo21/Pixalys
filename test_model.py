import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

# Charger les noms de classes
with open("class_names.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)

# Charger le modèle ResNet50
model = models.resnet50(weights=None)
num_classes = len(class_names)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("models/resnet50_compcars.pth", map_location=torch.device('cpu')))
model.eval()

# Prétraitement de l'image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Charger l'image à prédire
image_path = "Voiture2.jpg"  # Assure-toi qu'elle est bien dans le dossier
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)

# Prédiction
with torch.no_grad():
    output = model(input_tensor)
    predicted_idx = output.argmax(1).item()
    predicted_class = class_names[predicted_idx]

print("Prédiction :", predicted_class)
