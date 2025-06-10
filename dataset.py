import os
from PIL import Image
from torch.utils.data import Dataset

class CompCarsDataset(Dataset):
    def __init__(self, root_dir, split_file, transform=None, class_to_idx=None):
        self.root_dir = root_dir
        self.transform = transform

        with open(split_file, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines() if line.strip() != '']

        # Détecter si les chemins sont absolus ou relatifs
        self.paths_are_absolute = all(os.path.isabs(p) for p in self.image_paths)

        # Pour récupérer les classes, il faut extraire la bonne partie du chemin
        # Si absolu, on doit extraire la partie relative à root_dir
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            classes = set()
            for path in self.image_paths:
                if self.paths_are_absolute:
                    # Ex: path = "D:/Python/CompCars/image/37/1993/2010/d344ef4be1ebf8.jpg"
                    # On enlève root_dir + os.sep pour garder "37/1993/2010/d344ef4be1ebf8.jpg"
                    rel_path = os.path.relpath(path, self.root_dir)
                else:
                    rel_path = path

                parts = rel_path.split(os.sep)
                if len(parts) < 3:
                    raise ValueError(f"Ligne mal formée : {rel_path}")
                model_id = parts[1]  # model_id = 2ème dossier après 'image'
                classes.add(model_id)

            classes = sorted(list(classes))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Si chemin absolu, on l'utilise tel quel
        if not self.paths_are_absolute:
            img_path = os.path.join(self.root_dir, img_path)

        image = Image.open(img_path).convert('RGB')

        # Extraction du class_name de la partie relative
        if self.paths_are_absolute:
            rel_path = os.path.relpath(img_path, self.root_dir)
        else:
            rel_path = img_path

        parts = rel_path.split(os.sep)
        if len(parts) < 3:
            raise ValueError(f"Ligne mal formée : {rel_path}")
        model_id = parts[1]

        label = self.class_to_idx[model_id]

        if self.transform:
            image = self.transform(image)

        return image, label



