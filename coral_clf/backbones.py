import os
import torch
import torch.nn as nn

DEFAULT_GITHUB_REPO = "facebookresearch/dinov3"

def load_dinov3_backbones(
    vit_name: str = "dinov3_vit7b16",
    cnn_name: str = "dinov3_convnext_large",
    repo_dir: str = None,
    pretrained: bool = True,
    use_mock: bool = False,
):
    """
    Load DINOv3 Vision Transformer and ConvNeXt backbones.
    
    Loads from local repo if repo_dir or DINOV3_REPO_DIR env var is set,
    otherwise loads directly from facebookresearch/dinov3 on GitHub.
    
    Args:
        vit_name: Name of the ViT model in DINOv3 (default: dinov3_vit7b16)
        cnn_name: Name of the ConvNeXt model in DINOv3 (default: dinov3_convnext_large)
        repo_dir: Optional local path to cloned dinov3 repository
        pretrained: Whether to load pretrained weights from Hugging Face / Meta
        use_mock: If True, returns lightweight dummy backbones for offline testing/dry-runs
    """
    if use_mock:
        print("[INFO] Instantiating mock backbones for dry-run/testing...")
        class MockViT(nn.Module):
            def forward(self, x):
                return torch.zeros((x.size(0), 4096), device=x.device)

        class MockConv(nn.Module):
            def forward(self, x):
                return torch.zeros((x.size(0), 1536), device=x.device)

        return MockViT(), MockConv()

    target_repo = repo_dir or os.environ.get("DINOV3_REPO_DIR")

    try:
        if target_repo and os.path.exists(target_repo):
            print(f"[INFO] Loading DINOv3 backbones from local path: {target_repo}")
            model_vit = torch.hub.load(target_repo, vit_name, source="local", pretrained=pretrained)
            model_cnn = torch.hub.load(target_repo, cnn_name, source="local", pretrained=pretrained)
        else:
            print(f"[INFO] Loading DINOv3 backbones from GitHub: {DEFAULT_GITHUB_REPO}")
            model_vit = torch.hub.load(DEFAULT_GITHUB_REPO, vit_name, source="github", pretrained=pretrained)
            model_cnn = torch.hub.load(DEFAULT_GITHUB_REPO, cnn_name, source="github", pretrained=pretrained)

        # Freeze backbones as feature extractors
        for p in model_vit.parameters():
            p.requires_grad = False
        for p in model_cnn.parameters():
            p.requires_grad = False

        return model_vit, model_cnn

    except Exception as e:
        raise RuntimeError(
            f"Failed to load DINOv3 backbones ({vit_name}, {cnn_name}).\n"
            f"Error details: {e}\n\n"
            f"Troubleshooting:\n"
            f"1. Make sure you are authenticated with Hugging Face for Meta DINOv3: run `huggingface-cli login`.\n"
            f"2. If working in an offline environment or cluster, clone https://github.com/facebookresearch/dinov3\n"
            f"   and set `export DINOV3_REPO_DIR=/path/to/dinov3`.\n"
            f"3. For offline code verification, pass `use_mock=True`."
        ) from e
