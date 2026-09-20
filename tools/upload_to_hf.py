#!/usr/bin/env python3
"""
Standby Tool: Upload Exported Model to Hugging Face Hub
Creates the model repository and uploads model.safetensors, config.json, and the Model Card.

Usage:
    python tools/upload_to_hf.py --repo-id YOUR_USERNAME/coral-convnext-vit
"""

import os
import argparse
from huggingface_hub import HfApi

MODEL_CARD_TEMPLATE = """---
pipeline_tag: image-classification
tags:
  - coral
  - marine-biology
  - underwater-vision
  - dinov3
  - convnext
  - pytorch
license: mit
datasets:
  - gv3-coral
---

# Robust In-Situ Coral Image Classification with a DINOv3 Hybrid Vision Architecture

Official model weights for the paper:  
**"Robust In-Situ Coral Image Classification with a DINOv3 Hybrid Vision Architecture"**  
*Dat Nguyen, Minh Tran, Denton Bobeldyk, Jonathan P. Leidig* (Grand Valley State University)

## Model Overview
- **Architecture**: Dual-encoder DINOv3 ViT-7B/16 + ConvNeXt-Large with a trained MLP head
- **Task**: 27-class Caribbean in situ coral species classification
- **Benchmark Accuracy**: **91.04% top-1 accuracy** on the 27-class GV3 benchmark (+8.9% over prior ResNet+ViT)
- **Input Resolution**: 224x224 / 256x256

## Usage via Official GitHub Repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/coral-dinov3.git
cd coral-dinov3
pip install -r requirements.txt

# Automatic weight download and inference:
python infer.py --image sample_coral.jpg --hf-repo {repo_id}
```

## Citation
```bibtex
@article{{nguyen2026robust,
  title={{Robust In-Situ Coral Image Classification with a DINOv3 Hybrid Vision Architecture}},
  author={{Nguyen, Dat and Tran, Minh and Bobeldyk, Denton and Leidig, Jonathan P.}},
  year={{2026}},
  institution={{Grand Valley State University}}
}}
```
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Upload Coral Classifier to Hugging Face Hub")
    parser.add_argument("--repo-id", "-r", type=str, required=True, help="Target HF repo ID (e.g. username/coral-convnext-vit)")
    parser.add_argument("--model-dir", "-d", type=str, default="./exported_model", help="Directory containing model.safetensors and config.json")
    parser.add_argument("--token", type=str, default=None, help="Optional Hugging Face Write Token")
    return parser.parse_args()

def main():
    args = parse_args()
    api = HfApi(token=args.token)

    safetensors_file = os.path.join(args.model_dir, "model.safetensors")
    config_file = os.path.join(args.model_dir, "config.json")

    if not os.path.exists(safetensors_file) or not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Required files not found in {args.model_dir}.\n"
            f"Please run `python tools/export_weights.py` first to generate model.safetensors and config.json."
        )

    print(f"[INFO] Creating or verifying model repository on Hugging Face: {args.repo_id}...")
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)

    # 1. Upload Model Card (README.md)
    readme_content = MODEL_CARD_TEMPLATE.format(repo_id=args.repo_id)
    readme_path = os.path.join(args.model_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)

    print(f"[INFO] Uploading Model Card...")
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )

    # 2. Upload config.json
    print(f"[INFO] Uploading config.json...")
    api.upload_file(
        path_or_fileobj=config_file,
        path_in_repo="config.json",
        repo_id=args.repo_id,
        repo_type="model",
    )

    # 3. Upload model.safetensors
    print(f"[INFO] Uploading model.safetensors (this may take a moment)...")
    api.upload_file(
        path_or_fileobj=safetensors_file,
        path_in_repo="model.safetensors",
        repo_id=args.repo_id,
        repo_type="model",
    )

    print(f"\n[SUCCESS] Model successfully uploaded to Hugging Face Hub!")
    print(f"Model URL: https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()
