import os
from model import load_model, predict
from utils.preprocess import preprocess_tif
from utils.labels import load_label_image
from utils.vis import save_plot_as_png  # You'll need to implement this function

# Use your file path:
test_file = r"data/images/205.tif"  # use forward slashes for cross-platform compatibility

model = load_model()

# Preprocess input
input_tensor, input_tensor_vis = preprocess_tif(test_file)

# Predict mask
pred_mask = predict(model, input_tensor)

# Load true mask (assuming your load_label_image works with the filename)
true_mask = load_label_image("205.tif")

# Save input visualization image
input_img = input_tensor_vis[0]  # remove batch dimension

# Save the images as PNG files for the CML report
save_plot_as_png(input_img, "input_image.png")
save_plot_as_png(pred_mask, "pred_mask.png")
save_plot_as_png(true_mask, "true_mask.png")

# Save some example metric to metrics.txt
with open("metrics.txt", "w") as f:
    # For example, compute simple pixel-wise accuracy (you can improve this)
    accuracy = (pred_mask == true_mask).mean()
    f.write(f"Pixel-wise accuracy: {accuracy:.4f}\n")
