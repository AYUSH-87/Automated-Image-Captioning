"""
Image Feature Extraction Module.

Uses a pretrained InceptionV3 CNN (ImageNet weights) with the classification
head removed to extract a 2048-dimensional feature vector for each image.
Features are cached to disk so the CNN doesn't need to run during training.
"""

import os

import numpy as np
from PIL import Image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input  # type: ignore
from tensorflow.keras.models import Model  # type: ignore

import config
from utils import save_pickle


# ---------------------------------------------------------------------------
# 1. Build the CNN feature extractor
# ---------------------------------------------------------------------------

def build_feature_extractor():
    """
    Load InceptionV3 pretrained on ImageNet with the classification head
    removed and global average pooling applied.

    Returns
    -------
    model : keras Model
        Input: (1, 299, 299, 3)  →  Output: (1, 2048)
    """
    # include_top=False removes the final classification Dense layers.
    # pooling='avg' applies GlobalAveragePooling2D to the last conv output,
    # giving us a compact (2048,) feature vector per image.
    base_model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")

    # Freeze all layers — we only use this as a fixed feature extractor
    for layer in base_model.layers:
        layer.trainable = False

    print(f"[INFO] InceptionV3 feature extractor loaded. "
          f"Output shape: {base_model.output_shape}")
    return base_model


# ---------------------------------------------------------------------------
# 2. Load & preprocess a single image
# ---------------------------------------------------------------------------

def load_and_preprocess_image(image_path):
    """
    Load an image from disk, resize to InceptionV3 input size, and apply
    the model-specific preprocessing (scale pixels to [-1, 1]).

    Returns
    -------
    np.ndarray : shape (1, 299, 299, 3)
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img = img.resize(config.IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32)

    # Add batch dimension: (299, 299, 3) → (1, 299, 299, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # InceptionV3-specific preprocessing (scales to [-1, 1])
    img_array = preprocess_input(img_array)

    return img_array


# ---------------------------------------------------------------------------
# 3. Extract features for a single image
# ---------------------------------------------------------------------------

def extract_single_feature(model, image_path):
    """
    Extract the CNN feature vector for one image.

    Returns
    -------
    np.ndarray : shape (2048,)
    """
    img_array = load_and_preprocess_image(image_path)
    feature = model.predict(img_array, verbose=0)  # shape (1, 2048)
    return feature.flatten()  # shape (2048,)


# ---------------------------------------------------------------------------
# 4. Extract features for the entire dataset
# ---------------------------------------------------------------------------

def extract_all_features():
    """
    Iterate over every image in IMAGES_DIR, extract InceptionV3 features,
    and save the result as a pickled dict {image_filename: feature_vector}.

    Returns
    -------
    dict : {image_filename: np.ndarray of shape (2048,)}
    """
    print("\n" + "=" * 60)
    print("IMAGE FEATURE EXTRACTION")
    print("=" * 60)

    if not os.path.isdir(config.IMAGES_DIR):
        raise FileNotFoundError(
            f"Images directory not found: {config.IMAGES_DIR}\n"
            "Make sure the Flickr8k dataset is placed in the configured "
            "DATASET_DIR or set the FLICKR8K_DIR environment variable."
        )

    model = build_feature_extractor()

    features = {}
    image_files = [
        f for f in os.listdir(config.IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    total = len(image_files)
    print(f"[INFO] Found {total} images to process.")

    for idx, filename in enumerate(image_files, 1):
        image_path = os.path.join(config.IMAGES_DIR, filename)
        try:
            feature = extract_single_feature(model, image_path)
            features[filename] = feature
        except Exception as e:
            print(f"[WARN] Skipping {filename}: {e}")

        # Progress logging every 100 images
        if idx % 100 == 0 or idx == total:
            print(f"[INFO] Extracted features: {idx}/{total}")

    # Save to disk
    save_pickle(features, config.FEATURES_FILE)

    print(f"[INFO] Feature extraction complete. "
          f"Extracted {len(features)} / {total} images.\n")
    return features


# ---------------------------------------------------------------------------
# Entry point (can be run standalone for testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    extract_all_features()
