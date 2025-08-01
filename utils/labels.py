import os
import cv2
import numpy as np
from config import LABEL_DIR


def load_label_image(filename: str) -> np.ndarray:
    """
    Loads a binary label mask image from the LABEL_DIR using a .png extension.

    Args:
        filename (str): Filename (with or without extension) of the label image.

    Returns:
        np.ndarray: Binary mask (uint8), where 1 indicates mask and 0 is background.
    """
    base = os.path.splitext(filename)[0]
    label_path = os.path.join(LABEL_DIR, f"{base}.png")

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label image not found at: {label_path}")

    mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Failed to read mask image at: {label_path}")

    binary_mask = (mask > 0).astype(np.uint8)
    return binary_mask
