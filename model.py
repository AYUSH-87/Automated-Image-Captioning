"""
LSTM Decoder Model Definition.

Implements the "merge" architecture where image features and text sequences
are processed in separate branches and then combined (added) to predict the
next word in the caption.

Architecture
------------
Image branch:
    Input(2048) → Dense(256, relu) → Dropout(0.5) → image_embedding

Text branch:
    Input(max_len) → Embedding(vocab_size, 256) → Dropout(0.5) → LSTM(256) → text_embedding

Merge:
    image_embedding + text_embedding → Dense(256, relu) → Dense(vocab_size, softmax)
"""

from tensorflow.keras.layers import (  # type: ignore
    Add,
    Dense,
    Dropout,
    Embedding,
    Input,
    LSTM,
)
from tensorflow.keras.models import Model  # type: ignore

import config


def build_caption_model(vocab_size, max_len):
    """
    Build and compile the image captioning model.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary (number of unique words + 1 for padding).
    max_len : int
        Maximum padded sequence length for caption inputs.

    Returns
    -------
    model : keras Model
        Compiled model ready for training.
    """

    # -----------------------------------------------------------------------
    # Image feature branch
    # -----------------------------------------------------------------------
    # Input: pre-extracted CNN feature vector (2048-dim for InceptionV3)
    image_input = Input(shape=(config.CNN_FEATURE_DIM,), name="image_input")

    # Project image features down to the same dimensionality as the text branch
    image_dense = Dense(config.EMBEDDING_DIM, activation="relu",
                        name="image_dense")(image_input)
    image_drop = Dropout(config.DROPOUT_RATE, name="image_dropout")(image_dense)

    # -----------------------------------------------------------------------
    # Text sequence branch
    # -----------------------------------------------------------------------
    # Input: padded integer sequence of partial caption
    text_input = Input(shape=(max_len,), name="text_input")

    # Embedding layer learns dense word representations
    text_embed = Embedding(
        input_dim=vocab_size,
        output_dim=config.EMBEDDING_DIM,
        mask_zero=False,  # We use post-padding, masking not needed with Add
        name="text_embedding",
    )(text_input)

    text_drop = Dropout(config.DROPOUT_RATE, name="text_dropout")(text_embed)

    # LSTM processes the embedded sequence and returns the final hidden state
    text_lstm = LSTM(config.LSTM_UNITS, name="text_lstm")(text_drop)

    # -----------------------------------------------------------------------
    # Merge branches
    # -----------------------------------------------------------------------
    # Element-wise addition combines image and text representations.
    # Both branches output vectors of size EMBEDDING_DIM (= LSTM_UNITS = 256).
    merged = Add(name="merge_add")([image_drop, text_lstm])

    # -----------------------------------------------------------------------
    # Output layers
    # -----------------------------------------------------------------------
    output_dense = Dense(config.EMBEDDING_DIM, activation="relu",
                         name="output_dense")(merged)
    output = Dense(vocab_size, activation="softmax", name="output")(output_dense)

    # -----------------------------------------------------------------------
    # Build & compile
    # -----------------------------------------------------------------------
    model = Model(inputs=[image_input, text_input], outputs=output,
                  name="image_caption_model")

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
    )

    print("\n[INFO] Model architecture:")
    model.summary()

    return model


# ---------------------------------------------------------------------------
# Entry point (quick shape verification)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick sanity check with dummy values
    dummy_vocab = 5000
    dummy_max_len = 34
    m = build_caption_model(dummy_vocab, dummy_max_len)
    print(f"\n[OK] Model built successfully. "
          f"Inputs: {[i.shape for i in m.inputs]}, "
          f"Output: {m.output_shape}")
