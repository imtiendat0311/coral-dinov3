# Robust In-Situ Coral Image Classification with a DINOv3 Hybrid Vision Architecture

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Weights-yellow.svg" alt="Hugging Face"></a>
  <a href="https://github.com/facebookresearch/dinov3"><img src="https://img.shields.io/badge/Backbone-Meta%20DINOv3-black" alt="Meta DINOv3"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
</p>

This is the official codebase for the paper:  
**"Robust In-Situ Coral Image Classification with a DINOv3 Hybrid Vision Architecture"**  
*Dat Nguyen, Minh Tran, Denton Bobeldyk, Jonathan P. Leidig*  
**Grand Valley State University**, Allendale, MI, United States  

### 📬 Contacts & Authors
* **Dat Nguyen** (`nguyedat@mail.gvsu.edu`)
* **Minh Tran** (`tranmq@mail.gvsu.edu`)
* **Prof. Denton Bobeldyk** (`bobeldyd@gvsu.edu`)
* **Prof. Jonathan P. Leidig** (`leidijon@gvsu.edu`)

---

## 📌 Abstract

Coral reef ecosystems are under severe threat from recurrent mass bleaching and climate-driven decline, requiring scalable automated monitoring for effective conservation. This study evaluates **DINOv3**, a self-supervised vision foundation model, as a frozen feature extractor for automated *in situ* coral classification.

Eight model configurations spanning convolutional networks, Vision Transformers, a Mamba-Transformer backbone, and self-supervised backbones were evaluated on identical data partitions of the **GV3 dataset** comprising **78,012 in situ coral images extracted from 309 underwater videos covering 27 species** under strict video-level separation.

A dual-encoder model combining a frozen **DINOv3 ViT-7B/16** transformer encoder and a frozen **DINOv3 ConvNeXt-Large** convolutional encoder with a multi-layer perceptron head attains the highest accuracy on every subset, reaching **91.04% top-1 accuracy on the 27-class task** against 82.11% for a previously developed ResNet+ViT model (+8.9% absolute gain).

---

## 🏆 Key Benchmark Results (Table II)

All models evaluated on identical partitions of the GV3 dataset under strict video-level separation:

| Model Architecture | Backbone Pretraining | 10 Classes (Subset A) | 17 Classes (Subset B) | 27 Classes (Subset C) |
| :--- | :--- | :---: | :---: | :---: |
| **ResNet-50** | ImageNet-1K (Supervised) | 84.78% | 80.81% | 75.32% |
| **ViT-B/16** | ImageNet-1K (Supervised) | 85.41% | 83.49% | 77.25% |
| **ViT-L/16** | ImageNet-1K (Supervised) | 88.73% | 83.04% | 79.11% |
| **MambaVision-L2** | ImageNet-1K (Supervised) | 93.02% | 85.62% | 80.24% |
| **ResNet+ViT** *(prior)* | ImageNet-1K + ImageNet-21k | 94.89% | 87.10% | 82.11% |
| **DINOv3 ViT-B/16** | Meta LVD-1689M (Self-Supervised) | 96.27% | 91.79% | 87.43% |
| **DINOv3 ViT-H+/16** | Meta LVD-1689M (Self-Supervised) | 99.48% | 96.54% | 89.18% |
| **DINOv3 ViT+ConvNeXt (Ours)** | **Meta LVD-1689M (Self-Supervised)** | **99.64%** | **97.17%** | **91.04%** |

---

## 🏛 Architecture & Read-Out Width (Fig. 1 & Fig. 2)

```
                       Input Coral Crop (224x224 / 256x256)
                                        │
                       ┌────────────────┴────────────────┐
                       ▼                                 ▼
         [ViT-7B/16: Frozen DINOv3]        [ConvNeXt-Large: Frozen DINOv3]
             Class Token (CLS)               Global Average Pooling (GAP)
                 (4,096-d)                             (1,536-d)
                       │                                 │
               Linear(4096, 2048)                Linear(1536, 2048)
                       └────────────────┬────────────────┘
                                        ▼
                             Concatenate (4,096-dim)
                                        │
                                        ▼
                   Trainable Multi-Layer Perceptron (MLP) Head
                    4096 ──> 2048 ──> 1024 ──> 512 ──> 256 ──> N
                                        │
                                        ▼
                    Species Prediction (27 Caribbean Taxa)
```

Unlike prior methods that flatten the full patch token grid (150,528 dimensions), our class-token read-out operates on 4,096 dimensions (**36.8× narrower**), ensuring lightweight training and rapid convergence without updating frozen foundation weights.

---

## 🌊 End-to-End Reef Monitoring Pipeline (Fig. 3)

The classifier serves as the automated taxonomic labeling engine within a real-world benthic survey workflow:

```
[Underwater Video / Orthomosaic] 
             │
             ▼
[Segment Anything (SAM 3)]  ──> Zero-shot promptable colony segmentation
             │
             ▼
[DINOv3 Hybrid Classifier]  ──> Species-level identification (91.04% Top-1)
             │
             ▼
[TagLab Human-in-the-Loop]  ──> Expert validation & differential contour export
```

---

## 📊 The GV3 Dataset

The **GV3 dataset** contains **78,012 images** across **27 Caribbean coral species** extracted from **309 diver-operated underwater videos**:
- **Strict Video-Level Partitioning**: All frames from a single source video are confined to only one partition (train, val, or test), preventing cross-contamination from adjacent video frames.
- **Taxonomic Breadth**: Includes 22 stony corals (*Scleractinia*), 3 fire corals (*Millepora* hydrocorals), and 2 black corals (*Antipatharia*).
- **Subsets**:
  - **Subset A (10 classes, 44,514 images)**: Near-balanced benchmark of the most prevalent species.
  - **Subset B (17 classes, 65,175 images)**: Moderately expanded taxonomic diversity.
  - **Subset C (27 classes, 78,012 images)**: Full in situ distribution with up to 9.9:1 natural class imbalance.

---

## ⚡ Getting Started

### 1. Installation

```bash
git clone https://github.com/imtiendat0311/coral-dinov3.git
cd coral-dinov3
pip install -r requirements.txt
pip install -e .
```

### 2. DINOv3 Foundation Weights Authentication

The foundation backbones are loaded from [Meta's official DINOv3 repository](https://github.com/facebookresearch/dinov3). Ensure you are authenticated with Hugging Face:

```bash
huggingface-cli login
```

*(Note: On high-performance computing clusters without public internet, set `export DINOV3_REPO_DIR=/path/to/local/dinov3`).*

---

## 🚀 Usage

### Single-Image & Patch Classification

Classify a single coral photo or an entire folder of cropped patches:

```bash
# Using local checkpoint:
python infer.py --image samples/coral.jpg --checkpoint checkpoints/model-9.pth

# Using Hugging Face Hub (automatic weight download):
python infer.py --image samples/coral.jpg --hf-repo imtiendat0311/coral-dinov3-vit-convnext

# Dry-run test with mock backbones:
python infer.py --image samples/coral.jpg --mock --topk 5
```

### Large Orthomosaic Processing & TagLab Export (SAM 3 + DINOv3)

Process large underwater survey mosaics, segment colonies with SAM 3, classify them with the hybrid model, and export directly to **TagLab JSON format**:

```bash
python mosaic_pipeline.py \
  --image /path/to/orthomosaic.png \
  --output ./mosaic_results \
  --checkpoint checkpoints/model-9.pth \
  --crop-size 3256 3256 \
  --crop-overlap 0 \
  --prompt "coral"
```

Generated outputs in `./mosaic_results`:
- `taglab_annotations.json`: TagLab differential contour annotations with species names and centroids.
- `tile_masks/`: Individual colony binary masks and masked color crops.

### Training & Reproducing Results

Train the hybrid model on custom datasets or GV3 partitions using [Hugging Face Accelerate](https://huggingface.co/docs/accelerate):

```bash
# Single GPU / CPU:
python -m training.train --config training/train_config.json

# Multi-GPU / Cluster via Accelerate:
accelerate launch -m training.train --config training/train_config.json
```

---

## 📦 Model Weights & Hugging Face Hub

Trained model checkpoints are hosted on the Hugging Face Hub:

| Model Variant | Backbone Size | Top-1 (27-Class) | Hugging Face Link |
| :--- | :---: | :---: | :---: |
| **DINOv3 ViT-B/16** | 86M | 87.43% | [Download Weights](https://huggingface.co/) |
| **DINOv3 ViT-H+/16** | 840M | 89.18% | [Download Weights](https://huggingface.co/) |
| **DINOv3 ViT+ConvNeXt (SOTA)** | 6,914M | **91.04%** | [Download Weights](https://huggingface.co/) |
---

## 📖 Citation

If you use this model, code, or the GV3 benchmark in your research, please cite our paper:

```bibtex
@article{nguyen2026robust,
  title={Robust In-Situ Coral Image Classification with a DINOv3 Hybrid Vision Architecture},
  author={Nguyen, Dat and Tran, Minh and Bobeldyk, Denton and Leidig, Jonathan P.},
  journal={arXiv preprint},
  year={2026},
  institution={Grand Valley State University}
}
```

---

## 📄 License & Copyright

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  
Copyright (c) 2026 Dat Nguyen, Minh Tran, Denton Bobeldyk, Jonathan P. Leidig (Grand Valley State University).
