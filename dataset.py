import os
from PIL import Image
from torch.utils.data import Dataset

class CompCarsDataset(Dataset):
    def __init__(self, root_dir, split_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        with open(split_file, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines() if line.strip() != '']

        classes = set()
        for path in self.image_paths:
            parts = path.split('/')
            if len(parts) < 3:
                print(f"[Warning] Ligne mal formée dans split_file : '{path}'")
                continue
            class_name = parts[1]  # ex: '1101'
            classes.add(class_name)

        classes = sorted(list(classes))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_rel_path = self.image_paths[idx]
        img_path = os.path.join(self.root_dir, img_rel_path)
        image = Image.open(img_path).convert('RGB')

        class_name = img_rel_path.split('/')[1]
        label = self.class_to_idx[class_name]

        if self.transform:
            image = self.transform(image)

        return image, label


