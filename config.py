"""
Configuration & Constants for the Image Captioning Pipeline.

All tuneable hyperparameters, file paths, and model settings are defined here.
Dataset path can be overridden via the FLICKR8K_DIR environment variable.
"""

import os

# ---------------------------------------------------------------------------
# Dataset Paths
# ---------------------------------------------------------------------------
# Override with:  export FLICKR8K_DIR=/path/to/flickr8k   (Linux/Mac)
#            or:  set FLICKR8K_DIR=D:\datasets\flickr8k   (Windows)
DATASET_DIR = os.environ.get("FLICKR8K_DIR", os.path.join(".", "data", "flickr8k"))

# Flickr8k ships images in an "Images" folder and captions in "captions.txt"
IMAGES_DIR = os.path.join(DATASET_DIR, "Images")
CAPTIONS_FILE = os.path.join(DATASET_DIR, "captions.txt")

# ---------------------------------------------------------------------------
# Output / Artifact Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(".", "output")
FEATURES_FILE = os.path.join(OUTPUT_DIR, "features.pkl")
TOKENIZER_FILE = os.path.join(OUTPUT_DIR, "tokenizer.pkl")
MODEL_FILE = os.path.join(OUTPUT_DIR, "best_model.keras")
HISTORY_PLOT = os.path.join(OUTPUT_DIR, "training_loss.png")

# ---------------------------------------------------------------------------
# CNN Feature Extractor Settings
# ---------------------------------------------------------------------------
CNN_MODEL = "inceptionv3"           # Backbone identifier
IMAGE_SIZE = (299, 299)             # InceptionV3 native input resolution
CNN_FEATURE_DIM = 2048              # Output dimension of InceptionV3 avg-pool

# ---------------------------------------------------------------------------
# Caption / Text Settings
# ---------------------------------------------------------------------------
START_TOKEN = "startseq"
END_TOKEN = "endseq"
MAX_CAPTION_LENGTH = 34             # Max padded sequence length (tokens)
# VOCAB_SIZE is computed at runtime after tokenizer fitting

# ---------------------------------------------------------------------------
# LSTM Decoder Hyperparameters
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 256                 # Word embedding dimension
LSTM_UNITS = 256                    # LSTM hidden state size
DROPOUT_RATE = 0.5                  # Dropout probability

# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.15            # Fraction of images held out for validation
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 5

# ---------------------------------------------------------------------------
# Ensure output directory exists
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
