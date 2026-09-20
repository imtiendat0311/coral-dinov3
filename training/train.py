#!/usr/bin/env python3
"""
Training Pipeline for ConvNextVIT Coral Classification Model
Uses Hugging Face Accelerate for multi-GPU and single-GPU execution.

Usage:
    python -m training.train --config training/train_config.json
    accelerate launch -m training.train --config training/train_config.json
"""

import os
import itertools
try:
    import json5
except ImportError:
    import json as json5
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from accelerate import Accelerator

from coral_clf.model import build_coral_classifier
from coral_clf.transforms import make_transform
from training.dataset import CoralDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvNextVIT Coral Classifier")
    parser.add_argument("--config", "-c", type=str, default="training/train_config.json", help="Path to config.json")
    parser.add_argument("--mock-backbones", action="store_true", help="Use mock backbones for fast dry-run")
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator()
    print(f"[INFO] Accelerate device: {accelerator.device}")

    with open(args.config, "r") as f:
        config = json5.load(f)

    # 1. Instantiate Model
    model, _ = build_coral_classifier(
        config=config,
        device=str(accelerator.device),
        use_mock_backbones=args.mock_backbones,
    )

    # Ensure backbones are frozen and only projection + head train
    for p in model.model_vit.parameters():
        p.requires_grad = False
    for p in model.model_conv.parameters():
        p.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"]
    )
    criterion = nn.CrossEntropyLoss()

    # 2. Datasets & Dataloaders
    train_transform = make_transform(resize_size=256, is_train=True)
    eval_transform = make_transform(resize_size=256, is_train=False)

    train_ds = CoralDataset(config["train_path"], transform=train_transform)
    val_ds = CoralDataset(config["val_path"], transform=eval_transform)
    test_ds = CoralDataset(config["test_path"], transform=eval_transform)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )

    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )

    # 3. Setup Output Directories
    exp_dir = os.path.join(config["output_dir"], config["output_dir_name"])
    if accelerator.is_main_process:
        os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "val_tracker"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "test_tracker"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "confusion_matrix_val"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "confusion_matrix_test"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "curves"), exist_ok=True)

    logs = pd.DataFrame(columns=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"])

    # 4. Training Loop
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss, train_acc = 0.0, 0.0

        for i, (img, _, e_labels, _, _) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(img, img)
            loss = criterion(outputs, e_labels)
            accelerator.backward(loss)
            optimizer.step()

            y_pred = torch.argmax(outputs, dim=1)
            acc = (e_labels == y_pred).float().mean().item()
            train_loss += loss.item()
            train_acc += acc

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        print(f"[Epoch {epoch+1}/{config['num_epochs']}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        # --- Validation Loop ---
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for img, _, e_labels, _, _ in val_loader:
                outputs = model(img, img)
                loss = criterion(outputs, e_labels)
                val_loss += loss.item()
                val_acc += (e_labels == torch.argmax(outputs, dim=1)).float().mean().item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        print(f"               Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # --- Test Loop ---
        test_loss, test_acc = 0.0, 0.0
        with torch.no_grad():
            for img, _, e_labels, _, _ in test_loader:
                outputs = model(img, img)
                loss = criterion(outputs, e_labels)
                test_loss += loss.item()
                test_acc += (e_labels == torch.argmax(outputs, dim=1)).float().mean().item()

        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        print(f"               Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.4f}")

        # Save Checkpoint
        if accelerator.is_main_process and config.get("save_model", True):
            if epoch % config.get("save_every", 1) == 0 or epoch == config["num_epochs"] - 1:
                ckpt_path = os.path.join(exp_dir, "checkpoints", f"model-{epoch}.pth")
                unwrapped = accelerator.unwrap_model(model)
                accelerator.save(unwrapped.state_dict(), ckpt_path)
                print(f"[INFO] Saved checkpoint to: {ckpt_path}")

    print("\n[INFO] Training complete!")

if __name__ == "__main__":
    main()
