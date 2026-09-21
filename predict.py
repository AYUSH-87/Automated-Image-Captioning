"""
Inference / Prediction Module.

Loads the trained LSTM model, tokenizer, and CNN feature extractor, then
generates a natural-language caption for a given image using greedy decoding.
"""

import os

import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore

import config
from preprocess_images import build_feature_extractor, load_and_preprocess_image
from utils import load_pickle


# ---------------------------------------------------------------------------
# 1. Greedy Decoding
# ---------------------------------------------------------------------------

def generate_caption(model, feature_extractor, tokenizer, image_path, max_len):
    """
    Generate a caption for a single image using greedy decoding.

    Steps:
    1. Extract CNN features from the image.
    2. Start with the START_TOKEN.
    3. At each step, predict the next word (highest probability).
    4. Append the predicted word and repeat until END_TOKEN or max_len.

    Parameters
    ----------
    model : keras Model
        Trained captioning model (image_features + text_seq → next_word).
    feature_extractor : keras Model
        CNN model that outputs image feature vectors.
    tokenizer : keras Tokenizer
        Fitted tokenizer for word ↔ index mapping.
    image_path : str
        Path to the input image file.
    max_len : int
        Maximum caption length (for padding).

    Returns
    -------
    str : Generated caption (without start/end tokens).
    """
    # Extract image features: (1, 2048)
    img_array = load_and_preprocess_image(image_path)
    feature = feature_extractor.predict(img_array, verbose=0)  # (1, 2048)

    # Build reverse word index: {index: word}
    index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}

    # Start decoding with the start token
    caption_tokens = [config.START_TOKEN]

    for _ in range(max_len):
        # Convert current partial caption to integer sequence
        seq = tokenizer.texts_to_sequences([" ".join(caption_tokens)])[0]

        # Pad to max_len
        seq = pad_sequences([seq], maxlen=max_len, padding="post")

        # Predict next word probabilities
        predictions = model.predict([feature, seq], verbose=0)  # (1, vocab_size)

        # Greedy: pick the word with highest probability
        predicted_index = np.argmax(predictions[0])

        # Map index back to word
        predicted_word = index_to_word.get(predicted_index, None)

        # Stop if we predict the end token or an unknown index
        if predicted_word is None or predicted_word == config.END_TOKEN:
            break

        caption_tokens.append(predicted_word)

    # Remove the start token from the final caption
    final_caption = " ".join(caption_tokens[1:])
    return final_caption


# ---------------------------------------------------------------------------
# 2. Main prediction entry point
# ---------------------------------------------------------------------------

def predict(image_path):
    """
    Full prediction pipeline for a single image.

    Loads the trained model, tokenizer, and CNN feature extractor, then
    generates and prints the caption.

    Parameters
    ----------
    image_path : str
        Path to the image file to caption.
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not os.path.isfile(config.MODEL_FILE):
        raise FileNotFoundError(
            f"Trained model not found: {config.MODEL_FILE}\n"
            "Please run 'python main.py train' first."
        )

    if not os.path.isfile(config.TOKENIZER_FILE):
        raise FileNotFoundError(
            f"Tokenizer not found: {config.TOKENIZER_FILE}\n"
            "Please run 'python main.py train' first."
        )

    # ------------------------------------------------------------------
    # Load artifacts
    # ------------------------------------------------------------------
    print("\n[INFO] Loading trained model...")
    caption_model = load_model(config.MODEL_FILE)

    print("[INFO] Loading tokenizer...")
    tokenizer = load_pickle(config.TOKENIZER_FILE)

    print("[INFO] Loading CNN feature extractor...")
    feature_extractor = build_feature_extractor()

    # Determine max_len from the model's text input shape
    # The text_input layer shape is (None, max_len)
    max_len = caption_model.input_shape[1][1]
    print(f"[INFO] Max caption length: {max_len}")

    # ------------------------------------------------------------------
    # Generate caption
    # ------------------------------------------------------------------
    print(f"\n[INFO] Generating caption for: {image_path}")
    caption = generate_caption(
        caption_model, feature_extractor, tokenizer, image_path, max_len
    )

    print("\n" + "=" * 60)
    print("GENERATED CAPTION")
    print("=" * 60)
    print(f"\n  >> {caption}\n")
    print("=" * 60)

    return caption


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    predict(sys.argv[1])
