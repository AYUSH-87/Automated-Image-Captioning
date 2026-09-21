"""
Caption Preprocessing Module.

Handles loading, cleaning, tokenizing, and saving captions from the Flickr8k
dataset. Produces a fitted Keras Tokenizer and a cleaned captions dictionary.

Flickr8k captions.txt format (header row + CSV-like):
    image,caption
    1000268201_693b08cb0e.jpg,A child in a pink dress ...
"""

import os
import re
import string

from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore

import config
from utils import save_pickle


# ---------------------------------------------------------------------------
# 1. Load raw captions from disk
# ---------------------------------------------------------------------------

def load_captions(captions_file):
    """
    Parse the Flickr8k captions.txt file.

    Returns
    -------
    dict : {image_filename: [caption_1, caption_2, ...]}
    """
    if not os.path.isfile(captions_file):
        raise FileNotFoundError(
            f"Captions file not found: {captions_file}\n"
            "Make sure the Flickr8k dataset is placed in the configured "
            "DATASET_DIR or set the FLICKR8K_DIR environment variable."
        )

    captions_dict = {}

    with open(captions_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip the header line ("image,caption")
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        # Split on the FIRST comma only (captions may contain commas)
        parts = line.split(",", 1)
        if len(parts) < 2:
            continue

        image_id = parts[0].strip()
        caption = parts[1].strip()

        if image_id not in captions_dict:
            captions_dict[image_id] = []
        captions_dict[image_id].append(caption)

    print(f"[INFO] Loaded captions for {len(captions_dict)} images.")
    return captions_dict


# ---------------------------------------------------------------------------
# 2. Clean individual captions
# ---------------------------------------------------------------------------

def clean_caption(caption):
    """
    Apply text-cleaning steps to a single caption string.

    Steps:
    - Lowercase
    - Remove digits
    - Remove punctuation
    - Remove single-character words (except 'a' and 'i')
    - Collapse multiple spaces
    """
    caption = caption.lower()
    # Remove digits
    caption = re.sub(r"\d+", "", caption)
    # Remove punctuation
    caption = caption.translate(str.maketrans("", "", string.punctuation))
    # Remove single-character words except 'a' and 'i'
    caption = " ".join(
        word for word in caption.split() if len(word) > 1 or word in ("a", "i")
    )
    # Collapse whitespace
    caption = caption.strip()
    return caption


def clean_all_captions(captions_dict):
    """
    Clean every caption in the dict and wrap with start/end tokens.

    Returns
    -------
    dict : {image_id: [cleaned_caption_with_tokens, ...]}
    """
    cleaned = {}
    for image_id, caption_list in captions_dict.items():
        cleaned[image_id] = []
        for cap in caption_list:
            cap = clean_caption(cap)
            # Wrap with start/end sequence tokens
            cap = f"{config.START_TOKEN} {cap} {config.END_TOKEN}"
            cleaned[image_id].append(cap)
    print(f"[INFO] Cleaned captions for {len(cleaned)} images.")
    return cleaned


# ---------------------------------------------------------------------------
# 3. Build vocabulary & fit tokenizer
# ---------------------------------------------------------------------------

def build_tokenizer(captions_dict):
    """
    Fit a Keras Tokenizer on all cleaned captions.

    Returns
    -------
    tokenizer : keras Tokenizer
    vocab_size : int   (word_index size + 1 for zero-padding)
    """
    all_captions = []
    for caption_list in captions_dict.values():
        all_captions.extend(caption_list)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)

    vocab_size = len(tokenizer.word_index) + 1  # +1 because index 0 is reserved

    print(f"[INFO] Vocabulary size: {vocab_size}")
    return tokenizer, vocab_size


# ---------------------------------------------------------------------------
# 4. Compute maximum caption length (in tokens)
# ---------------------------------------------------------------------------

def compute_max_length(captions_dict):
    """Return the length (in tokens) of the longest caption."""
    max_len = 0
    for caption_list in captions_dict.values():
        for cap in caption_list:
            length = len(cap.split())
            if length > max_len:
                max_len = length
    print(f"[INFO] Maximum caption length: {max_len} tokens")
    return max_len


# ---------------------------------------------------------------------------
# 5. Run full caption preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_captions():
    """
    End-to-end caption preprocessing:
    1. Load raw captions
    2. Clean & add start/end tokens
    3. Fit tokenizer & compute vocab size
    4. Compute max caption length
    5. Save tokenizer and cleaned captions to disk

    Returns
    -------
    captions_dict : dict of cleaned captions
    tokenizer : fitted Keras Tokenizer
    vocab_size : int
    max_len : int
    """
    print("\n" + "=" * 60)
    print("CAPTION PREPROCESSING")
    print("=" * 60)

    # Load
    raw_captions = load_captions(config.CAPTIONS_FILE)

    # Clean
    captions_dict = clean_all_captions(raw_captions)

    # Tokenizer
    tokenizer, vocab_size = build_tokenizer(captions_dict)

    # Max length
    max_len = compute_max_length(captions_dict)

    # Save tokenizer
    save_pickle(tokenizer, config.TOKENIZER_FILE)

    # Save cleaned captions
    captions_path = os.path.join(config.OUTPUT_DIR, "captions_clean.pkl")
    save_pickle(captions_dict, captions_path)

    print(f"[INFO] Caption preprocessing complete.\n")
    return captions_dict, tokenizer, vocab_size, max_len


# ---------------------------------------------------------------------------
# Entry point (can be run standalone for testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    preprocess_captions()
