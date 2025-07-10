import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Settings
image_folder = "C:/Users/rober/Desktop/Aqua AI/dataset"  # CHANGE THIS
#image_folder = "C:/Users/rober/Desktop/small dataset"  # CHANGE THIS
output_image_path = "pca_visualization.png"

def extract_features(image_path):
    """Extract color histogram features from an image."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (128, 128))  # Resize for uniformity
    hist_features = []
    for i in range(3):  # For each color channel
        hist = cv2.calcHist([image], [i], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)
    return np.array(hist_features)

def load_images_and_extract_features(folder_path):
    features = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            path = os.path.join(folder_path, filename)
            print(filename)
            feat = extract_features(path)
            if feat is not None:
                features.append(feat)
                filenames.append(filename)
    return np.array(features), filenames

# Step 1: Load images and extract features
features, filenames = load_images_and_extract_features(image_folder)

# Step 2: Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 3: Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)

# Step 4: Visualize and Save
plt.figure(figsize=(10, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', alpha=0.6)
#for i, name in enumerate(filenames):
#    plt.text(pca_result[i, 0], pca_result[i, 1], name, fontsize=8, alpha=0.7)
#plt.title("PCA of Image Features")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_image_path)
plt.show()
