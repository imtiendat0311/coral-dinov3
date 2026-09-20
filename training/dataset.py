import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CoralDataset(Dataset):
    """
    Dataset for loading individual coral image patches from a CSV file.
    Expects columns: 'label', 'e_label', 'path', 'file_name', 'source'
    """
    def __init__(self, csv_file: str, transform=None, root_dir: str = None):
        self.coral_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.coral_frame)

    def __getitem__(self, idx):
        row = self.coral_frame.iloc[idx]
        label = str(row["label"])
        e_label = int(row["e_label"])
        file_name = str(row["file_name"])
        source = str(row.get("source", "unknown"))

        base_path = self.root_dir if self.root_dir else str(row["path"])
        full_img_path = os.path.join(base_path, label, file_name)

        if not os.path.exists(full_img_path):
            # Fallback if path column already included label
            fallback = os.path.join(base_path, file_name)
            if os.path.exists(fallback):
                full_img_path = fallback

        image = Image.open(full_img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label, torch.tensor(e_label, dtype=torch.long), source, file_name
