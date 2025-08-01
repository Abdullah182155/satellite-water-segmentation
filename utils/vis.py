import matplotlib.pyplot as plt
import numpy as np

def save_plot_as_png(img_array, filename):
    plt.figure(figsize=(6,6))
    if len(img_array.shape) == 3 and img_array.shape[0] in [1,3]:  # e.g. (C,H,W)
        # Convert CHW to HWC for plotting
        img = np.transpose(img_array, (1, 2, 0))
        if img.shape[2] == 1:
            img = img.squeeze(axis=2)
    else:
        img = img_array

    plt.axis('off')
    plt.imshow(img, cmap='gray' if img.ndim==2 else None)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
