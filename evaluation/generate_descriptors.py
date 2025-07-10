import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm

# Settings
root_dir = "C:/Users/rober/Desktop/Aqua AI/5600-visuals/output"  # CHANGE THIS
output_csv = "image_descriptors.csv"

# Parameters for LBP
radius = 1
n_points = 8 * radius
METHOD = 'uniform'

# Output container
all_descriptors = []

# Function to compute descriptors
def compute_descriptors(image_path):
    entry = {"filename": os.path.basename(image_path)}
    
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Resize for consistency
    img = cv2.resize(img, (256, 256))

    # --- SIFT ---
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is not None:
        entry["sift_mean"] = np.mean(descriptors)
        entry["sift_std"] = np.std(descriptors)
    else:
        entry["sift_mean"] = np.nan
        entry["sift_std"] = np.nan

    # --- HOG ---
    hog_desc = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False, feature_vector=True)
    entry["hog_mean"] = np.mean(hog_desc)
    entry["hog_std"] = np.std(hog_desc)

    # --- LBP ---
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2), density=True)
    entry["lbp_uniformity"] = np.var(lbp_hist)
    
    return entry

# Iterate through all PNG files
for dirpath, _, filenames in os.walk(root_dir):
    for file in filenames:
        if file.lower().endswith(".png"):
            img_path = os.path.join(dirpath, file)
            print(f"Reading {img_path}")
            desc = compute_descriptors(img_path)
            if desc:
                all_descriptors.append(desc)

# Convert to DataFrame and save
df = pd.DataFrame(all_descriptors)
df.to_csv(output_csv, index=False)

print(f"Saved descriptors for {len(df)} images to {output_csv}")
