import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import CompCarsDataset

# Configuration
root_dir = 'CompCars/image'
split_file = 'CompCars/train_test_split/classification_clean/train_all_clean.txt'
batch_size = 32
num_epochs = 10
learning_rate = 0.0001  # plus bas car on fine-tune plusieurs couches
model_save_path = 'resnet50_compcars.pth'

# Prétraitement des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset & DataLoader
train_dataset = CompCarsDataset(root_dir=root_dir, split_file=split_file, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Charger le modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Appareil utilisé : {device}")

weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

# Adapter la dernière couche
num_classes = len(train_dataset.class_to_idx)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Dégeler layer4 et fc
for name, param in model.named_parameters():
    param.requires_grad = False  # tout geler d'abord
    if name.startswith("layer4") or name.startswith("fc"):
        param.requires_grad = True

# Charger les poids s’ils existent
start_epoch = 0
if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    print(f"Modèle chargé depuis {model_save_path}")

model.to(device)

# Fonction de perte et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate
)

# Entraînement
model.train()
print(f"Démarrage de l'entraînement pour {num_epochs} époques...")

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

    # Sauvegarde après chaque époque
    torch.save(model.state_dict(), model_save_path)
    print(f"Modèle sauvegardé dans {model_save_path}")

print("\n Entraînement terminé.")

