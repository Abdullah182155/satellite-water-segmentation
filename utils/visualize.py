import io
import base64
import numpy as np
import matplotlib
matplotlib.use('agg')  # Use non-GUI backend for safe server-side plotting
import matplotlib.pyplot as plt
import torch
import cv2
from typing import Union


def plot_to_base64(
    input_tensor_vis: Union[torch.Tensor, np.ndarray],
    pred_mask: np.ndarray,
    true_mask: np.ndarray
) -> str:
    """
    Creates a side-by-side visualization of RGB image, true mask, and predicted mask,
    and returns the result as a base64-encoded PNG image.

    Args:
        input_tensor_vis (torch.Tensor | np.ndarray): (C, H, W) normalized RGB tensor or array.
        pred_mask (np.ndarray): Predicted mask (2D array).
        true_mask (np.ndarray): Ground truth mask (2D array).

    Returns:
        str: Base64-encoded string of the PNG image.
    """
    # Convert to numpy and (H, W, C) format
    if isinstance(input_tensor_vis, torch.Tensor):
        rgb = input_tensor_vis.cpu().numpy().transpose(1, 2, 0)
    else:
        rgb = np.transpose(input_tensor_vis, (1, 2, 0))

    rgb = np.clip(rgb, 0.0, 1.0)  # Ensure in [0,1]
    H, W, _ = rgb.shape

    def resize_mask(mask: np.ndarray, target_size: tuple) -> np.ndarray:
        if mask.shape != target_size:
            return cv2.resize(mask.astype(np.uint8), (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
        return mask

    pred_mask_resized = resize_mask(pred_mask, (H, W))
    true_mask_resized = resize_mask(true_mask, (H, W))

    # Plot image and masks
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    titles = ["Image (RGB)", "True Mask", "Predicted Mask"]
    images = [rgb, true_mask_resized, pred_mask_resized]
    cmaps = [None, 'gray', 'gray']

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    # Save plot to memory and encode
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode('utf-8')
