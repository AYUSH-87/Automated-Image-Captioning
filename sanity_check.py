"""
End-to-end sanity check for the Image Captioning Pipeline.

Tests:
1. Caption preprocessing (with synthetic data)
2. CNN feature extraction (single image)
3. Model build & shape verification
4. Data generator output shapes
5. Single training step
6. Greedy decoding (with random weights)

All tests use synthetic/dummy data — no Flickr8k dataset needed.
"""

import os
import sys
import tempfile
import numpy as np

# Patch config to use temp dirs before importing other modules
import config
config.OUTPUT_DIR = os.path.join(".", "output_test")
config.TOKENIZER_FILE = os.path.join(config.OUTPUT_DIR, "tokenizer.pkl")
config.FEATURES_FILE = os.path.join(config.OUTPUT_DIR, "features.pkl")
config.MODEL_FILE = os.path.join(config.OUTPUT_DIR, "test_model.keras")
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  [PASS] {name}")
        PASS += 1
    else:
        print(f"  [FAIL] {name} -- {detail}")
        FAIL += 1

# ===================================================================
print("\n1. CAPTION PREPROCESSING")
print("-" * 40)
# ===================================================================

from preprocess_captions import clean_caption, clean_all_captions, build_tokenizer

cap = clean_caption("A Dog is Running 123 in the Park!!!")
check("clean_caption lowercases", cap == cap.lower())
check("clean_caption removes digits", "123" not in cap)
check("clean_caption removes punctuation", "!" not in cap)

captions_dict = {
    "img1.jpg": ["A dog runs in the park", "The dog is playing outside"],
    "img2.jpg": ["A cat sits on a mat", "The cat is sleeping"],
    "img3.jpg": ["Mountains and blue sky", "Beautiful travel landscape"],
}
cleaned = clean_all_captions(captions_dict)
check("clean_all adds start token", all(
    c.startswith(config.START_TOKEN) for caps in cleaned.values() for c in caps
))
check("clean_all adds end token", all(
    c.endswith(config.END_TOKEN) for caps in cleaned.values() for c in caps
))

tokenizer, vocab_size = build_tokenizer(cleaned)
check("vocab_size > 0", vocab_size > 0, f"got {vocab_size}")
check("tokenizer encodes 'startseq'", config.START_TOKEN in tokenizer.word_index)
check("tokenizer encodes 'endseq'", config.END_TOKEN in tokenizer.word_index)

# ===================================================================
print("\n2. CNN FEATURE EXTRACTOR")
print("-" * 40)
# ===================================================================

from preprocess_images import build_feature_extractor, load_and_preprocess_image, extract_single_feature
from PIL import Image

fe = build_feature_extractor()
check("feature extractor output dim", fe.output_shape[-1] == 2048,
      f"got {fe.output_shape}")

# Create a temporary test image
test_img_path = os.path.join(config.OUTPUT_DIR, "test_image.jpg")
img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
img.save(test_img_path)

preprocessed = load_and_preprocess_image(test_img_path)
check("preprocessed shape", preprocessed.shape == (1, 299, 299, 3),
      f"got {preprocessed.shape}")

feature = extract_single_feature(fe, test_img_path)
check("feature vector shape", feature.shape == (2048,), f"got {feature.shape}")

# ===================================================================
print("\n3. MODEL BUILD & SHAPES")
print("-" * 40)
# ===================================================================

from model import build_caption_model

max_len = 20
caption_model = build_caption_model(vocab_size, max_len)

check("model has 2 inputs", len(caption_model.inputs) == 2)
check("image input shape", caption_model.inputs[0].shape[-1] == 2048,
      f"got {caption_model.inputs[0].shape}")
check("text input shape", caption_model.inputs[1].shape[-1] == max_len,
      f"got {caption_model.inputs[1].shape}")
check("output shape matches vocab", caption_model.output_shape[-1] == vocab_size,
      f"got {caption_model.output_shape}")

# ===================================================================
print("\n4. DATA GENERATOR")
print("-" * 40)
# ===================================================================

from utils import data_generator, count_samples

features = {
    "img1.jpg": np.random.randn(2048).astype(np.float32),
    "img2.jpg": np.random.randn(2048).astype(np.float32),
    "img3.jpg": np.random.randn(2048).astype(np.float32),
}

n_samples = count_samples(cleaned, features, tokenizer)
check("count_samples > 0", n_samples > 0, f"got {n_samples}")

gen = data_generator(cleaned, features, tokenizer, max_len, vocab_size, batch_size=4)
batch = next(gen)
X_batch, y_batch = batch
X_img, X_seq = X_batch

check("X_img batch shape", X_img.shape[1] == 2048, f"got {X_img.shape}")
check("X_seq batch shape", X_seq.shape[1] == max_len, f"got {X_seq.shape}")
check("y batch shape", y_batch.shape[1] == vocab_size, f"got {y_batch.shape}")
check("batch sizes match", X_img.shape[0] == X_seq.shape[0] == y_batch.shape[0])

# ===================================================================
print("\n5. SINGLE TRAINING STEP")
print("-" * 40)
# ===================================================================

loss = caption_model.train_on_batch([X_img, X_seq], y_batch)
check("training step runs", True)
check("loss is finite", np.isfinite(loss), f"got {loss}")

# ===================================================================
print("\n6. GREEDY DECODING")
print("-" * 40)
# ===================================================================

from predict import generate_caption

# Save model + tokenizer so predict can work
caption_model.save(config.MODEL_FILE)
from utils import save_pickle
save_pickle(tokenizer, config.TOKENIZER_FILE)

caption = generate_caption(caption_model, fe, tokenizer, test_img_path, max_len)
check("caption is a string", isinstance(caption, str))
check("caption is non-empty", len(caption) > 0, f"got: '{caption}'")
check("caption has no start token", config.START_TOKEN not in caption)
check("caption has no end token", config.END_TOKEN not in caption)

# ===================================================================
# CLEANUP & SUMMARY
# ===================================================================
import shutil
try:
    shutil.rmtree(config.OUTPUT_DIR)
except:
    pass

print("\n" + "=" * 50)
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} checks")
print("=" * 50)

if FAIL > 0:
    sys.exit(1)
else:
    print("[ALL CHECKS PASSED]")
    sys.exit(0)
