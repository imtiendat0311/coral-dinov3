#!/usr/bin/env python3
"""
Standby Tool: Convert PyTorch Training Checkpoint to Hugging Face Format
Extracts weights into 'model.safetensors' and generates 'config.json'.

Usage:
    python tools/export_weights.py --checkpoint path/to/model-9.pth --output-dir ./exported_model
"""

import os
import json
import argparse
import torch
from safetensors.torch import save_file

def parse_args():
    parser = argparse.ArgumentParser(description="Export checkpoint to safetensors & config.json")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Path to input .pth checkpoint")
    parser.add_argument("--output-dir", "-o", type=str, default="./exported_model", help="Directory to save exported model")
    parser.add_argument("--classes", type=str, default="classes.json", help="Path to classes.json")
    return parser.parse_args()

def main():
    args = parse_args()

    # Auto-detect checkpoint path if not provided
    ckpt_path = args.checkpoint
    if not ckpt_path:
        for candidate in ["model-9.pth", "checkpoints/model-9.pth", "train_model/output_27/checkpoints/model-9.pth"]:
            if os.path.exists(candidate):
                ckpt_path = candidate
                break

    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found. Searched candidate paths ('model-9.pth', 'checkpoints/model-9.pth'). "
            f"Please specify using --checkpoint <path>"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Extract state_dict
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # Clean DDP module. prefixes and unshare aliased memory
    clean_dict = {
        (k[7:] if k.startswith("module.") else k): v.clone().contiguous()
        for k, v in state_dict.items()
    }

    # 1. Save safetensors
    safetensors_path = os.path.join(args.output_dir, "model.safetensors")
    save_file(clean_dict, safetensors_path)
    size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
    print(f"[INFO] Successfully exported: {safetensors_path} ({size_mb:.2f} MB)")

    # 2. Build config.json
    id2label = {}
    if os.path.exists(args.classes):
        with open(args.classes, "r") as f:
            id2label = json.load(f)

    label2id = {v: int(k) for k, v in id2label.items()}

    config = {
        "model_type": "convnext_vit_coral",
        "num_classes": len(id2label) if id2label else 27,
        "image_size": 256,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "latent_dim": 512,
        "dropout_rate": 0.5,
        "backbones": {
            "vit": "dinov3_vit7b16",
            "cnn": "dinov3_convnext_large",
            "source": "facebookresearch/dinov3"
        },
        "mlp": [
            [1024, 512, "BATCHNORM", "RELU", "DROP 0.5"],
            [512, 256, "BATCHNORM", "RELU", "DROP 0.5"],
            [256, 128, "BATCHNORM", "RELU", "DROP 0.5"],
            [128, 27]
        ],
        "id2label": id2label,
        "label2id": label2id
    }

    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] Saved configuration to: {config_path}")

    print("\n[SUCCESS] Model is ready for Hugging Face Hub upload! Run tools/upload_to_hf.py next.")

if __name__ == "__main__":
    main()
