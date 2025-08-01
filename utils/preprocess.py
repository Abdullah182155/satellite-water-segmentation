import rasterio
import numpy as np
import torch
from typing import List, Optional, Tuple


def preprocess_tif(
    path: str,
    vis_channels: List[int] = [4, 3, 2],
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocesses a multispectral .tif image for segmentation models.

    Args:
        path (str): Path to the .tif file.
        vis_channels (List[int]): 1-based indices used for visualization (e.g., [4, 3, 2]).
        device (Optional[torch.device]): Torch device to move tensors to (e.g., 'cuda').

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - Full normalized image tensor of shape (1, C, H, W)
            - Visualization image tensor of shape (1, 3, H, W)
    """
    with rasterio.open(path) as src:
        img = src.read()  # (C, H, W)

    if img.ndim != 3:
        raise ValueError(f"Expected 3D image but got shape {img.shape}")

    img = img.astype(np.float32)

    # Normalize each channel independently using min-max
    img_hwc = np.transpose(img, (1, 2, 0))  # (H, W, C)
    min_vals = img_hwc.min(axis=(0, 1), keepdims=True)
    max_vals = img_hwc.max(axis=(0, 1), keepdims=True)
    range_vals = np.maximum(max_vals - min_vals, 1e-6)  # avoid division by zero

    normalized = (img_hwc - min_vals) / range_vals  # (H, W, C)

    # Extract and normalize visualization channels
    vis_idxs = [c - 1 for c in vis_channels]  # convert to 0-based
    vis_hwc = np.transpose(img[vis_idxs], (1, 2, 0)).astype(np.float32)
    vis_min, vis_max = vis_hwc.min(), vis_hwc.max()
    vis_norm = (vis_hwc - vis_min) / (vis_max - vis_min + 1e-6)

    # Convert to torch tensors
    full_tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    vis_tensor = torch.from_numpy(vis_norm).permute(2, 0, 1).unsqueeze(0)     # (1, 3, H, W)

    if device:
        full_tensor = full_tensor.to(device)
        vis_tensor = vis_tensor.to(device)

    return full_tensor, vis_tensor
