"""
Coral Classification Package (coral_clf)
Hybrid Vision Transformer + ConvNeXt SOTA Coral Classifier.
"""

from .model import ConvNextVIT, build_coral_classifier
from .transforms import make_transform
from .backbones import load_dinov3_backbones

__version__ = "0.1.0"
__all__ = [
    "ConvNextVIT",
    "build_coral_classifier",
    "make_transform",
    "load_dinov3_backbones",
]
