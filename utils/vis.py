import matplotlib.pyplot as plt
import numpy as np

def to_hwc(img):
    if img.ndim == 3 and img.shape[0] in [1,3]:
        return np.transpose(img, (1, 2, 0))
    return img

def combine_and_save_images(input_img, pred_mask, true_mask, output_filename="combined_result.png"):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(to_hwc(input_img))
    axs[0].set_title("Input Image")
    axs[0].axis('off')

    axs[1].imshow(pred_mask, cmap='gray')
    axs[1].set_title("Predicted Mask")
    axs[1].axis('off')

    axs[2].imshow(true_mask, cmap='gray')
    axs[2].set_title("Ground Truth Mask")
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
