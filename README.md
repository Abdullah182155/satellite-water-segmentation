# Multispectral Satellite Image Segmentation Projects

This repository contains experiments and implementations of various semantic segmentation models designed for satellite and multispectral imagery analysis. The main focus is on water body segmentation and evaluating different architectures for accurate land cover classification.

---

## 1. Models and Encoders Overview

This project explores multiple semantic segmentation architectures using the [`segmentation_models_pytorch (smp)`](https://github.com/qubvel/segmentation_models.pytorch) library. The goal is to compare model performance on multispectral satellite images with selected bands as input.

### Models Used

- **UNet-baseline**  
  A classic U-Net model used as a baseline for multispectral segmentation with standard encoder-decoder architecture and BCE loss.

- **DeepLabV3**  
  Leverages Atrous Spatial Pyramid Pooling (ASPP) for capturing multi-scale context. Ideal for segmenting large-scale objects in remote sensing images.

- **UNet++ (Nested UNet)**  
  Features dense skip connections and deep supervision. Excels at fine-grained segmentation and boundary refinement.

- **PSPNet (Pyramid Scene Parsing Network)**  
  Utilizes pyramid pooling for global and local context, effective for objects at various scales.

- **FPN (Feature Pyramid Network)**  
  Builds a top-down feature pyramid with lateral connections, great for multi-scale segmentation, especially in lightweight scenarios.

### Encoders (Backbones)

- **ResNet34** — Balanced depth and performance.  
- **ResNet50** — Deeper network with richer features.  
- **MobileNetV2** — Lightweight and fast, suitable for resource-constrained environments.

### Common Settings

- **Input channels**: 8 selected multispectral bands `[5, 6, 12, 7, 9, 10, 11, 8]`  
- **Image size**: Resized to `256x256`  
- **Loss function**: `BCEWithLogitsLoss`  
- **Optimizer**: Adam (`lr=1e-4`)  
- **Augmentations**: Horizontal/vertical flips, rotations, brightness/contrast (using Albumentations)  
- **Logging**: Training runs tracked with MLflow  

### Outputs

- Training and validation loss curves  
- Dice and IoU metrics progression  
- Saved best model weights (`best_model.pth`)

### Visualization

- Sample predictions compared to ground truth masks  
- Visualization of individual spectral bands supported 

---

## 4. Dataset Class: `WaterSegmentationDataset`

This custom PyTorch Dataset handles loading multispectral satellite images and their corresponding binary water masks.

- **Inputs**:  
  - `image_paths`: List of file paths to multispectral GeoTIFF images.  
  - `mask_paths`: List of file paths to grayscale binary mask images.  
  - `transform`: Optional Albumentations transformations applied jointly to image and mask.  
  - `selected_bands`: Optional list of 1-based band indices to select specific spectral bands from the images. If not provided, all bands are loaded.

- **Functionality**:  
  - Loads multispectral images with rasterio, selecting only the requested bands.  
  - Loads corresponding binary masks using OpenCV, converting them to binary format.  
  - Performs per-band min-max normalization on the images to scale values between 0 and 1.  
  - Applies any specified augmentations consistently to both images and masks.  
  - Returns tensors in `(channels, height, width)` format, suitable for model input.

This class ensures that the multispectral inputs and masks are properly aligned, normalized, and augmented during training.

---

## 2. Water Segmentation Using U-Net Transformer

This project aims at detecting water bodies in satellite imagery using a U-Net architecture with a Transformer-based MiT-B2 encoder.

### Dataset Structure

```
Water Segmentation/
├── data/
│   ├── images/      # 12-band multispectral GeoTIFF images
│   └── labels/      # Corresponding binary water masks (.png)
```

- Each image should have a matching label with the same base filename.  
- Images are 12-channel GeoTIFF files.  
- Labels are single-channel grayscale binary masks.

### Model Architecture

- Backbone: MiT-B2 Transformer encoder  
- Input channels: 12 spectral bands  
- Output: Binary segmentation mask (water vs. non-water)

### Training Pipeline

- Resize images to 128x128  
- Apply Albumentations augmentations (flip, rotate)  
- Normalize pixel values to [0, 1]  
- Dataset split: 70% train / 15% validation / 15% test  
- Loss: Binary Cross Entropy (BCE)  
- Metrics: Dice Score and Intersection over Union (IoU)  
- Batch size: 4  
- Optimizer: Adam  
- Training epochs: 20  
- Model checkpointing on best validation loss

### Outputs

- Training and validation loss curves  
- Dice and IoU metrics progression  
- Saved best model weights (`best_model.pth`)

### Visualization

- Sample predictions compared to ground truth masks  
- Visualization of individual spectral bands supported  

---

## Requirements

- torch==2.0.1+cu117
- torchvision==0.15.2+cu117
- segmentation-models-pytorch==0.3.3
- albumentations==1.3.0
- opencv-python==4.7.0.72
- rasterio==1.3.6
- tifffile==2023.7.26
- scipy==1.11.1
* scikit-learn==1.3.0
- matplotlib==3.7.1
- tqdm==4.66.1
- mlflow==2.7.0
 

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Notes

- Ensure that input images and masks have matching dimensions before training.  
- The MiT-B2 encoder does **not** use pretrained weights due to the 12-band input format.  
- The multispectral models expect specific input channel configurations as noted.

---

Made with ❤️ for geospatial deep learning.
