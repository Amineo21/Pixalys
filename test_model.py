import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

# Charger les noms de classes
with open("class_name.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)

# Charger le modèle ResNet50
model = models.resnet50(weights=None)
num_classes = len(class_names)
model.fc = nn.Linear(model.fc.in_features, num_classes)
checkpoint = torch.load("models/resnet50_compcars.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Prétraitement de l'image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Charger l'image à prédire
image_path = "uploads/AS3Sedan.jpg"
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)

# Prédiction
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    top3 = torch.topk(probs, 3)
    print("Top 3 prédictions :")
    filtered = []
    model_ids = sorted(class_names.keys())
    for idx, score in zip(top3.indices[0], top3.values[0]):
        idx = idx.item()
        model_id = model_ids[idx]
        name = class_names.get(model_id, "Unknown")
        print(f"Indice : {idx} | model_id : {model_id} | Classe : {name} | Score : {score:.4f}")
        if name != "Unknown":
            filtered.append((model_id, name, score.item()))

    if filtered:
        # Prédiction finale : la classe connue avec le score le plus élevé dans le top 3
        final_model_id, final_name, final_score = max(filtered, key=lambda x: x[2])
        print(f"\nPrédiction finale (classe connue) : {final_name} (model_id : {final_model_id}, score : {final_score:.4f})")
    else:
        print("\nPrédiction finale : Aucune classe connue dans le top 3")
