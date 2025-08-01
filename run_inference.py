import os
import numpy as np
from model import load_model, predict
from utils.preprocess import preprocess_tif
from utils.labels import load_label_image
from utils.vis import combine_and_save_images  # we'll add this function below

# Path to your test TIFF file
test_file = "data/images/205.tif"

model = load_model()

# Preprocess input
input_tensor, input_tensor_vis = preprocess_tif(test_file)

# Predict mask
pred_mask = predict(model, input_tensor)

# Load true mask
true_mask = load_label_image("205.tif")

# Get input image visualization (remove batch dimension)
input_img = input_tensor_vis[0]

# Combine and save all three images into one file
combine_and_save_images(input_img, pred_mask, true_mask, "combined_result.png")

# Save metrics example: pixel-wise accuracy
accuracy = (pred_mask == true_mask).mean()
with open("metrics.txt", "w") as f:
    f.write(f"Pixel-wise accuracy: {accuracy:.4f}\n")
