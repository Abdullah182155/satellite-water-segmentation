import os
import torch

BASE_UPLOAD = '/tmp/uploads'
os.makedirs(BASE_UPLOAD, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "unet_model.pth"
LABEL_DIR = os.path.join("data", "labels")
VIS_CHANNELS = [4, 3, 2]
THRESHOLD = 0.5
