from .basic import L1LabLoss
from .perceptual import PerceptualLoss
from .gan import GANLoss
from .patchnce import PatchNCELoss
from .advanced import (
    PhotometricSmoothnessLoss,
    DepthSmoothnessLoss,
    ColorConsistencyPyrLoss,
    EntropyLoss,
    OMMClusterLoss,
)

__all__ = [
    "L1LabLoss",
    "PerceptualLoss",
    "GANLoss",
    "PatchNCELoss",
    "PhotometricSmoothnessLoss",
    "DepthSmoothnessLoss",
    "ColorConsistencyPyrLoss",
    "EntropyLoss",
    "OMMClusterLoss",
]
