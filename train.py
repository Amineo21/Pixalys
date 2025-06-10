import os
os.environ["TORCH_HOME"] = "D:/Stock/torch_cache"
os.environ["HF_HOME"] = "D:/Stock/torch_cache"

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import CompCarsDataset  # Assure-toi que ton dataset utilise bien class_name.json

# Configuration
root_dir = 'CompCars/image'
split_file = 'CompCars/train_test_split/classification/train.txt'
val_split_file = 'CompCars/train_test_split/classification/val.txt'
batch_size = 32
num_epochs = 30
learning_rate = 0.0001
model_save_path = 'resnet50_compcars.pth'

# Charger le mapping des classes
with open("class_name.json", "r", encoding="utf-8") as f:
    id_to_name = json.load(f)
class_to_idx = {class_id: idx for idx, class_id in enumerate(sorted(id_to_name.keys()))}

# Prétraitement
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset & DataLoader
train_dataset = CompCarsDataset(
    root_dir=root_dir,
    split_file=split_file,
    transform=transform,
    class_to_idx=class_to_idx  # si besoin
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CompCarsDataset(
    root_dir=root_dir,
    split_file=val_split_file,
    transform=transform,
    class_to_idx=class_to_idx
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))

# Fine-tuning
for name, param in model.named_parameters():
    param.requires_grad = False
    if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
        param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate
)

# Reprise éventuelle
if os.path.exists(model_save_path):
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Modèle chargé depuis {model_save_path}, reprise à l'époque {start_epoch}")
else:
    start_epoch = 0

model.to(device)
model.train()
print(f"Démarrage de l'entraînement pour {num_epochs} époques...")

best_val_loss = float('inf')  # À placer avant la boucle d'entraînement

for epoch in range(start_epoch, num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    print(f"[Époque {epoch + 1}/{num_epochs}] Perte : {epoch_loss:.4f}")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_save_path)
    print(f"Modèle sauvegardé dans {model_save_path}")

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    val_loss /= len(val_loader)
    val_acc = correct / total if total > 0 else 0
    print(f"[Époque {epoch + 1}/{num_epochs}] Perte validation : {val_loss:.4f} | Accuracy : {val_acc:.4f}")

    # Sauvegarde du meilleur modèle
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, "best_resnet50_compcars.pth")
        print(f"Meilleur modèle sauvegardé (val_loss={val_loss:.4f}) dans best_resnet50_compcars.pth")

    model.train()

print("\nEntraînement terminé.")


