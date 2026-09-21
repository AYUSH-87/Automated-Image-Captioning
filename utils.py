"""
Utility functions for the Image Captioning Pipeline.

Includes:
- Pickle save/load helpers
- Custom data generator for memory-efficient training
- Miscellaneous helpers
"""

import os
import pickle

import numpy as np
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore


# ---------------------------------------------------------------------------
# Pickle helpers
# ---------------------------------------------------------------------------

def save_pickle(obj, filepath):
    """Serialize an object to a pickle file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"[INFO] Saved -> {filepath}")


def load_pickle(filepath):
    """Deserialize an object from a pickle file."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    print(f"[INFO] Loaded <- {filepath}")
    return obj


# ---------------------------------------------------------------------------
# Data Generator
# ---------------------------------------------------------------------------

def data_generator(captions_dict, features, tokenizer, max_len, vocab_size, batch_size):
    """
    Yield batches of ([image_features, partial_sequence], next_word_one_hot).

    This generator avoids expanding all (image, partial_seq, next_word) triples
    into memory at once, which is critical for running on a normal laptop.

    Parameters
    ----------
    captions_dict : dict
        {image_id: [list of cleaned caption strings]} for the current split.
    features : dict
        {image_id: np.ndarray of shape (CNN_FEATURE_DIM,)} pre-extracted CNN features.
    tokenizer : keras Tokenizer
        Fitted tokenizer mapping words → integer indices.
    max_len : int
        Maximum padded sequence length.
    vocab_size : int
        Size of the vocabulary (for one-hot encoding the target word).
    batch_size : int
        Number of samples per batch.

    Yields
    ------
    ([X_image, X_seq], y) where:
        X_image : np.ndarray, shape (batch_size, CNN_FEATURE_DIM)
        X_seq   : np.ndarray, shape (batch_size, max_len)
        y       : np.ndarray, shape (batch_size, vocab_size)
    """
    while True:  # Loop indefinitely for Keras model.fit
        # Collect samples
        X_image, X_seq, y = [], [], []

        for image_id, caption_list in captions_dict.items():
            # Skip images whose features were not extracted
            if image_id not in features:
                continue

            feature = features[image_id]

            for caption in caption_list:
                # Convert the full caption to a sequence of integers
                seq = tokenizer.texts_to_sequences([caption])[0]

                # Create input-output pairs using teacher forcing:
                # For caption [1, 5, 8, 3, 9], generate pairs:
                #   input=[1, 5]       target=8
                #   input=[1, 5, 8]    target=3
                #   input=[1, 5, 8, 3] target=9
                for i in range(1, len(seq)):
                    in_seq = seq[:i]
                    out_word = seq[i]

                    # Pad input sequence to max_len
                    in_seq = pad_sequences([in_seq], maxlen=max_len, padding="post")[0]

                    # One-hot encode target word
                    out_word_vec = to_categorical([out_word], num_classes=vocab_size)[0]

                    X_image.append(feature)
                    X_seq.append(in_seq)
                    y.append(out_word_vec)

                    # Yield a batch when we have enough samples
                    if len(X_image) >= batch_size:
                        yield (
                            [np.array(X_image[:batch_size]), np.array(X_seq[:batch_size])],
                            np.array(y[:batch_size]),
                        )
                        # Keep any overflow for the next batch
                        X_image = X_image[batch_size:]
                        X_seq = X_seq[batch_size:]
                        y = y[batch_size:]

        # Yield remaining samples (partial last batch)
        if X_image:
            yield (
                [np.array(X_image), np.array(X_seq)],
                np.array(y),
            )


def count_samples(captions_dict, features, tokenizer):
    """
    Count the total number of (partial_seq → next_word) training samples.

    Used to compute steps_per_epoch for model.fit().
    """
    total = 0
    for image_id, caption_list in captions_dict.items():
        if image_id not in features:
            continue
        for caption in caption_list:
            seq = tokenizer.texts_to_sequences([caption])[0]
            # Each caption of length L produces (L - 1) training pairs
            total += max(0, len(seq) - 1)
    return total
