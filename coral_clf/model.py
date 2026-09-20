import os
import json
import torch
import torch.nn as nn
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from .backbones import load_dinov3_backbones
from .helper import activation

DEFAULT_MLP_CONFIG = [
    [1024, 512, "BATCHNORM", "RELU", "DROP 0.5"],
    [512, 256, "BATCHNORM", "RELU", "DROP 0.5"],
    [256, 128, "BATCHNORM", "RELU", "DROP 0.5"],
    [128, 27],
]

class ConvNextVIT(nn.Module):
    """
    Hybrid SOTA Coral Classifier fusing Vision Transformer (ViT)
    and ConvNeXt backbones with a multi-stage classification head.
    """
    def __init__(
        self,
        model_vit: nn.Module,
        latent_dim: int,
        model_convnext: nn.Module,
        fc: nn.Module,
        dp_rate: float = 0.5,
    ):
        super().__init__()
        self.model_vit = model_vit
        self.vit_process = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dp_rate),
            nn.Linear(4096, latent_dim),
        )
        self.model_conv = model_convnext
        self.conv_process = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dp_rate),
            nn.Linear(1536, latent_dim),
        )
        self.fc = fc

    def forward(self, vit_img: torch.Tensor, conv_img: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for ConvNextVIT.
        If only one image tensor is passed, it is forwarded through both branches.
        """
        if conv_img is None:
            conv_img = vit_img

        out_vit = self.model_vit(vit_img)
        out_vit = self.vit_process(out_vit)

        out_conv = self.model_conv(conv_img)
        out_conv = self.conv_process(out_conv)

        output_combined = torch.cat((out_vit, out_conv), dim=1)
        return self.fc(output_combined)


def build_head(mlp_config=None, num_classes: int = 27) -> nn.Sequential:
    """Build the multi-layer classification head."""
    layers = mlp_config or DEFAULT_MLP_CONFIG
    fc = nn.Sequential()

    for layer in layers:
        out_features = num_classes if layer == layers[-1] else layer[1]
        fc.append(nn.Linear(layer[0], out_features))

        for token in layer[2:]:
            token_str = str(token).lower()
            if token_str == "batchnorm":
                fc.append(nn.BatchNorm1d(out_features))
            elif token_str in activation:
                fc.append(activation[token_str])
            elif token_str.startswith("drop"):
                dp_rate = float(token_str.split("drop")[1].strip())
                fc.append(nn.Dropout(dp_rate))

    return fc


def build_coral_classifier(
    config: dict = None,
    checkpoint_path: str = None,
    hf_repo_id: str = None,
    device: str = "cpu",
    use_mock_backbones: bool = False,
    repo_dir: str = None,
) -> tuple[ConvNextVIT, dict]:
    """
    Factory function to construct the full ConvNextVIT model and load weights.
    
    Args:
        config: Model configuration dict (defaults to standard 27-class config)
        checkpoint_path: Local path to .pth or .safetensors weights
        hf_repo_id: Hugging Face model repository ID (e.g. 'username/coral-convnext-vit')
        device: Device to place the model on
        use_mock_backbones: Dry-run mode using dummy backbones
        repo_dir: Local path to DINOv3 repo
    """
    cfg = {
        "num_class": 27,
        "latent_dim": 512,
        "dropout_rate": 0.5,
        "mlp": DEFAULT_MLP_CONFIG,
    }
    if config:
        cfg.update(config)

    # 1. Load backbones
    model_vit, model_cnn = load_dinov3_backbones(
        repo_dir=repo_dir,
        use_mock=use_mock_backbones,
    )

    # 2. Build head
    fc = build_head(cfg.get("mlp"), num_classes=cfg.get("num_class", 27))

    # 3. Assemble ConvNextVIT
    model = ConvNextVIT(
        model_vit=model_vit,
        latent_dim=cfg["latent_dim"],
        model_convnext=model_cnn,
        fc=fc,
        dp_rate=cfg["dropout_rate"],
    )

    # 4. Load weights if provided
    weights_path = checkpoint_path
    if not weights_path and hf_repo_id:
        print(f"[INFO] Downloading model weights from Hugging Face Hub: {hf_repo_id}...")
        weights_path = hf_hub_download(repo_id=hf_repo_id, filename="model.safetensors")

    if weights_path and os.path.exists(weights_path):
        print(f"[INFO] Loading checkpoint from: {weights_path}")
        if weights_path.endswith(".safetensors"):
            state_dict = load_file(weights_path, device=str(device))
        else:
            loaded = torch.load(weights_path, map_location=device)
            state_dict = loaded.get("state_dict", loaded.get("model", loaded))

        clean_dict = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(clean_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys during load: {len(missing)}")
        if unexpected:
            print(f"[WARN] Unexpected keys during load: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model, cfg
