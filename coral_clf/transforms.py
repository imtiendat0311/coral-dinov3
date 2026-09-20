import torch
from torchvision.transforms import v2

def make_transform(resize_size: int = 256, is_train: bool = False):
    """
    Standard transform pipeline for ConvNextVIT coral classifier.
    Matches the original training pipeline with optional data augmentation.
    """
    transforms = [
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
    ]

    if is_train:
        transforms.extend([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
        ])

    transforms.extend([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])

    return v2.Compose(transforms)
