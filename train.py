"""
Training Module for the Image Captioning Pipeline.

Orchestrates:
1. Loading preprocessed captions, features, and tokenizer
2. Splitting data into train/validation sets
3. Building the LSTM decoder model
4. Training with data generators, checkpointing, and early stopping
5. Plotting and saving training/validation loss curves
"""

import os
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping  # type: ignore

import config
from utils import load_pickle, data_generator, count_samples
from model import build_caption_model
from preprocess_captions import preprocess_captions
from preprocess_images import extract_all_features


# ---------------------------------------------------------------------------
# 1. Split image IDs into train / validation sets
# ---------------------------------------------------------------------------

def split_data(captions_dict, val_fraction=None):
    """
    Split image IDs into training and validation sets.

    Parameters
    ----------
    captions_dict : dict
        {image_id: [caption_list]}
    val_fraction : float, optional
        Fraction of images for validation. Defaults to config.VALIDATION_SPLIT.

    Returns
    -------
    train_captions : dict
    val_captions : dict
    """
    if val_fraction is None:
        val_fraction = config.VALIDATION_SPLIT

    image_ids = list(captions_dict.keys())
    np.random.seed(42)
    np.random.shuffle(image_ids)

    split_idx = int(len(image_ids) * (1 - val_fraction))
    train_ids = set(image_ids[:split_idx])
    val_ids = set(image_ids[split_idx:])

    train_captions = {k: v for k, v in captions_dict.items() if k in train_ids}
    val_captions = {k: v for k, v in captions_dict.items() if k in val_ids}

    print(f"[INFO] Train: {len(train_captions)} images, "
          f"Val: {len(val_captions)} images")
    return train_captions, val_captions


# ---------------------------------------------------------------------------
# 2. Plot and save training history
# ---------------------------------------------------------------------------

def plot_training_history(history):
    """Save training & validation loss curves to disk."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss", linewidth=2)
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    plt.title("Image Captioning Model - Training Loss", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Categorical Cross-Entropy)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.HISTORY_PLOT, dpi=150)
    plt.close()
    print(f"[INFO] Loss plot saved -> {config.HISTORY_PLOT}")


# ---------------------------------------------------------------------------
# 3. Main training function
# ---------------------------------------------------------------------------

def train():
    """
    Run the full training pipeline:
    1. Preprocess captions (or load from cache)
    2. Extract image features (or load from cache)
    3. Build model
    4. Train with generators
    5. Save loss plot
    """
    print("\n" + "=" * 60)
    print("IMAGE CAPTIONING - TRAINING PIPELINE")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Caption preprocessing
    # ------------------------------------------------------------------
    captions_clean_path = os.path.join(config.OUTPUT_DIR, "captions_clean.pkl")

    if os.path.isfile(config.TOKENIZER_FILE) and os.path.isfile(captions_clean_path):
        print("[INFO] Found cached caption artifacts. Loading...")
        captions_dict = load_pickle(captions_clean_path)
        tokenizer = load_pickle(config.TOKENIZER_FILE)
        vocab_size = len(tokenizer.word_index) + 1
        max_len = max(
            len(cap.split()) for caps in captions_dict.values() for cap in caps
        )
        print(f"[INFO] Vocabulary size: {vocab_size}, Max caption length: {max_len}")
    else:
        captions_dict, tokenizer, vocab_size, max_len = preprocess_captions()

    # ------------------------------------------------------------------
    # Step 2: Image feature extraction
    # ------------------------------------------------------------------
    if os.path.isfile(config.FEATURES_FILE):
        print("[INFO] Found cached image features. Loading...")
        features = load_pickle(config.FEATURES_FILE)
    else:
        features = extract_all_features()

    # ------------------------------------------------------------------
    # Step 3: Split data
    # ------------------------------------------------------------------
    train_captions, val_captions = split_data(captions_dict)

    # ------------------------------------------------------------------
    # Step 4: Build model
    # ------------------------------------------------------------------
    caption_model = build_caption_model(vocab_size, max_len)

    # ------------------------------------------------------------------
    # Step 5: Compute steps per epoch
    # ------------------------------------------------------------------
    train_samples = count_samples(train_captions, features, tokenizer)
    val_samples = count_samples(val_captions, features, tokenizer)

    steps_per_epoch = math.ceil(train_samples / config.BATCH_SIZE)
    val_steps = math.ceil(val_samples / config.BATCH_SIZE)

    print(f"[INFO] Training samples: {train_samples} -> {steps_per_epoch} steps/epoch")
    print(f"[INFO] Validation samples: {val_samples} -> {val_steps} steps/epoch")

    # ------------------------------------------------------------------
    # Step 6: Create data generators
    # ------------------------------------------------------------------
    train_gen = data_generator(
        train_captions, features, tokenizer, max_len, vocab_size, config.BATCH_SIZE
    )
    val_gen = data_generator(
        val_captions, features, tokenizer, max_len, vocab_size, config.BATCH_SIZE
    )

    # ------------------------------------------------------------------
    # Step 7: Callbacks
    # ------------------------------------------------------------------
    callbacks = [
        # Save the best model based on validation loss
        ModelCheckpoint(
            filepath=config.MODEL_FILE,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        # Stop early if validation loss doesn't improve
        EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # ------------------------------------------------------------------
    # Step 8: Train!
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("STARTING TRAINING")
    print("-" * 60)

    history = caption_model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=config.EPOCHS,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # Step 9: Save final model (in case checkpoint didn't trigger)
    # ------------------------------------------------------------------
    final_model_path = os.path.join(config.OUTPUT_DIR, "final_model.keras")
    caption_model.save(final_model_path)
    print(f"[INFO] Final model saved -> {final_model_path}")

    # ------------------------------------------------------------------
    # Step 10: Plot loss curves
    # ------------------------------------------------------------------
    plot_training_history(history)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
