#!/usr/bin/env python3
"""
Single-Image & Batch Inference Script for Coral Classifier
Usage:
    python infer.py --image path/to/coral.jpg --checkpoint path/to/model-9.pth
    python infer.py --image path/to/coral.jpg --mock   # dry-run without backbones/weights
"""

import argparse
import os
import torch
from PIL import Image

from coral_clf.model import build_coral_classifier
from coral_clf.transforms import make_transform
from coral_clf.helper import load_class_labels

def parse_args():
    parser = argparse.ArgumentParser(description="Coral Species Classifier")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Path to model checkpoint (.pth or .safetensors)")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face repo ID to pull weights from")
    parser.add_argument("--classes", type=str, default="classes.json", help="Path to classes.json or test.csv")
    parser.add_argument("--topk", "-k", type=int, default=5, help="Number of top predictions to display")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mock", action="store_true", help="Run with mock backbones for testing/dry-run")
    return parser.parse_args()

def classify_single_image(image_path: str, model, transform, class_labels, topk: int, device: str):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor, tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_probs, top_indices = torch.topk(probs, min(topk, len(probs)))

    print(f"\nPredictions for: {image_path}")
    print("-" * 50)
    for rank, (p, idx) in enumerate(zip(top_probs, top_indices), start=1):
        idx_val = idx.item()
        label_name = class_labels.get(idx_val, f"Class_{idx_val}")
        print(f"  {rank}. {label_name:<30} {p.item() * 100:6.2f}%")
    print("-" * 50)

def main():
    args = parse_args()

    # 1. Load class labels
    labels_file = args.classes if os.path.exists(args.classes) else None
    class_labels = load_class_labels(classes_json_path=labels_file)

    # 2. Build model
    ckpt_path = args.checkpoint
    if not ckpt_path and not args.hf_repo and not args.mock:
        for candidate in ["model-9.pth", "checkpoints/model-9.pth"]:
            if os.path.exists(candidate):
                ckpt_path = candidate
                print(f"[INFO] Auto-detected local checkpoint: {ckpt_path}")
                break

    print(f"[INFO] Initializing model on device: {args.device}...")
    model, _ = build_coral_classifier(
        checkpoint_path=ckpt_path,
        hf_repo_id=args.hf_repo,
        device=args.device,
        use_mock_backbones=args.mock,
    )
    transform = make_transform(resize_size=256, is_train=False)

    # 3. Predict
    if os.path.isdir(args.image):
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        files = [
            os.path.join(args.image, f)
            for f in sorted(os.listdir(args.image))
            if f.lower().endswith(image_extensions)
        ]
        print(f"[INFO] Found {len(files)} images in directory: {args.image}")
        for img_p in files:
            classify_single_image(img_p, model, transform, class_labels, args.topk, args.device)
    else:
        classify_single_image(args.image, model, transform, class_labels, args.topk, args.device)

if __name__ == "__main__":
    main()
