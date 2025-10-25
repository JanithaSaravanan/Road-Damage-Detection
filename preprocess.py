import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import albumentations as A
import os

# --- Photometric Normalization (MSRCR or CLAHE) ---
def apply_clahe(img):
    """Applies Contrast Limited Adaptive Histogram Equalization."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

# --- Color Constancy (Shades-of-Gray) ---
def apply_shades_of_gray(img, p=6):
    """
    Applies the Shades-of-Gray algorithm for white balancing.
    It assumes the average pixel color should be a 'shade of gray'.
    """
    img_float = img.astype(np.float32)
    # Convert BGR to RGB for processing
    img_rgb = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)
    
    # Calculate the p-norm for each color channel
    r_norm = np.mean(img_rgb[:, :, 0]**p)**(1/p)
    g_norm = np.mean(img_rgb[:, :, 1]**p)**(1/p)
    b_norm = np.mean(img_rgb[:, :, 2]**p)**(1/p)

    # Calculate the gray reference and scaling factors
    sum_norm = r_norm + g_norm + b_norm
    gray_ref = np.mean(img_rgb**p)**(1/p)
    
    # Apply scaling to each channel
    img_rgb[:, :, 0] = img_rgb[:, :, 0] * (gray_ref / r_norm)
    img_rgb[:, :, 1] = img_rgb[:, :, 1] * (gray_ref / g_norm)
    img_rgb[:, :, 2] = img_rgb[:, :, 2] * (gray_ref / b_norm)

    # Clamp and convert back to BGR
    img_corrected = np.clip(img_rgb, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_corrected, cv2.COLOR_RGB2BGR)
    return img_bgr

# --- Data Augmentation (Albumentations + RandAugment) ---
def apply_data_augmentation(img):
    """
    Applies a series of random augmentations to the image.
    RandAugment applies a random number of transformations with a random magnitude.
    This is typically used during model training, not for single-image inference.
    """
    transform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
        ], p=1.0),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.ISONoise(p=0.5),
        ], p=1.0),
        A.OneOf([
            A.CLAHE(p=0.5),
            A.Sharpen(p=0.5),
        ], p=1.0),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=1.0),
    ])
    
    augmented = transform(image=img)['image']
    return augmented

# --- Domain Adaptation (DANN + CycleGAN) ---
def explain_domain_adaptation():
    """Explains why DANN and CycleGAN cannot be applied to a single image."""
    print("\n--- Domain Adaptation ---")
    print("DANN and CycleGAN are not single-image processing techniques.")
    print("They are **training frameworks** used to adapt a model to a new dataset.")
    print("For example:")
    print("• DANN (Domain-Adversarial Neural Network) is used to train a model to be robust to changes between datasets (e.g., from a 'daytime' image dataset to a 'nighttime' dataset).")
    print("• CycleGAN is used to translate an entire collection of images from one domain to another (e.g., converting a folder of landscape photos to the style of a Monet painting).")
    print("They cannot be applied to a single image as a standalone step because they require a source domain and a target domain for training.")
    print("-" * 25)

def main():
    image_path = "img1.jpg"

    if not os.path.exists(image_path):
        print(f"Error: The image file '{image_path}' was not found.")
        return

    # --- Load Image ---
    print("1. Loading image...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read the image file at '{image_path}'.")
        return
    
    # Create an output directory
    os.makedirs("output_images", exist_ok=True)
    cv2.imwrite("output_images/original_img.png", img)

    # --- Step 1: Photometric Normalization (CLAHE) ---
    print("2. Applying CLAHE normalization...")
    clahe_img = apply_clahe(img)
    cv2.imwrite("output_images/clahe_img.png", clahe_img)

    # --- Step 2: Color Constancy (Shades-of-Gray) ---
    print("3. Applying Shades-of-Gray color constancy...")
    sog_img = apply_shades_of_gray(clahe_img)
    cv2.imwrite("output_images/sog_img.png", sog_img)

    # --- Step 3: Data Augmentation ---
    print("4. Applying data augmentation (randomly)...")
    augmented_img = apply_data_augmentation(sog_img)
    cv2.imwrite("output_images/augmented_img.png", augmented_img)
    
    # --- Step 4: Domain Adaptation Explanation ---
    explain_domain_adaptation()
    print("✅ All processing complete. Outputs saved in 'output_images/' directory.")


if __name__ == "__main__":
    main()