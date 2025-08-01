import torch
from config import DEVICE, MODEL_PATH
from UNet import UNet
from typing import Optional


def load_model(in_channels: int = 12, out_channels: int = 1, model_path: Optional[str] = None) -> torch.nn.Module:
    """
    Loads a trained UNet model from disk.

    Args:
        in_channels (int): Number of input channels (default is 12).
        out_channels (int): Number of output channels (default is 1).
        model_path (str, optional): Path to the model weights. If None, uses default MODEL_PATH.

    Returns:
        torch.nn.Module: Loaded UNet model in evaluation mode.
    """
    model = UNet(in_channels=in_channels, out_channels=out_channels).to(DEVICE)

    path = model_path or MODEL_PATH
    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def predict(model: torch.nn.Module, image_tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Performs forward pass and generates binary segmentation mask.

    Args:
        model (torch.nn.Module): Trained segmentation model.
        image_tensor (torch.Tensor): Input tensor of shape (1, C, H, W).
        threshold (float): Threshold to binarize output probabilities.

    Returns:
        np.ndarray: Binary mask as 2D NumPy array (H, W).
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs)
        mask = (probs > threshold).float().squeeze().cpu().numpy()
    
    return mask
