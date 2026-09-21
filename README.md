# Automated Image Captioning — CNN-LSTM Pipeline

A backend-only Python project that combines a pretrained **InceptionV3** CNN feature extractor with an **LSTM decoder** to generate descriptive, context-aware captions for travel photographs. Runs entirely from the terminal.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Merge Architecture                      │
│                                                             │
│  Image ──► InceptionV3 ──► Dense(256) ──► Dropout ──┐      │
│                                                      │ Add  │
│  Text  ──► Embedding ──► Dropout ──► LSTM(256) ─────┘      │
│                                          │                  │
│                                    Dense(256, relu)         │
│                                          │                  │
│                                Dense(vocab_size, softmax)   │
│                                          │                  │
│                                    Next Word                │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
├── config.py              # All configurable constants (paths, hyperparams)
├── main.py                # CLI entry point (train / predict)
├── preprocess_captions.py # Caption cleaning, tokenization, vocabulary
├── preprocess_images.py   # CNN feature extraction from images
├── model.py               # LSTM decoder model definition
├── train.py               # Training loop, validation, checkpointing
├── predict.py             # Inference: feature extraction + greedy decoding
├── utils.py               # Shared utilities (data generator, helpers)
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## Prerequisites

- Python 3.8+
- ~2 GB disk space for InceptionV3 weights + dataset
- GPU recommended but not required (CPU training is slower but works)

## Setup

### 1. Create a Virtual Environment

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Flickr8k Dataset

Download the Flickr8k dataset from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) and extract it. The expected structure is:

```
flickr8k/
├── Images/
│   ├── 1000268201_693b08cb0e.jpg
│   ├── 1001773457_577c3a7d70.jpg
│   └── ... (8,091 images)
└── captions.txt
```

### 4. Configure the Dataset Path

By default, the project looks for the dataset at `./data/flickr8k/`. You can change this by setting an environment variable:

```bash
# Linux / macOS
export FLICKR8K_DIR=/path/to/flickr8k

# Windows (Command Prompt)
set FLICKR8K_DIR=D:\datasets\flickr8k

# Windows (PowerShell)
$env:FLICKR8K_DIR = "D:\datasets\flickr8k"
```

Or place the dataset in the default location:
```bash
mkdir -p data
# Move/copy your flickr8k folder into ./data/
```

## Usage

### Train the Model

```bash
python main.py train
```

This runs the full pipeline:
1. **Caption preprocessing** — cleans text, builds vocabulary, fits tokenizer
2. **Feature extraction** — extracts InceptionV3 features for all images (~10-20 min on CPU)
3. **Model training** — trains the LSTM decoder with early stopping and checkpointing

Outputs are saved to the `output/` directory:
- `features.pkl` — cached CNN feature vectors
- `tokenizer.pkl` — fitted Keras Tokenizer
- `best_model.keras` — best model checkpoint (by validation loss)
- `final_model.keras` — model at end of training
- `training_loss.png` — loss curve plot

### Generate a Caption

```bash
python main.py predict --image path/to/your/image.jpg
```

Example:
```bash
python main.py predict --image ./data/flickr8k/Images/1000268201_693b08cb0e.jpg
```

Output:
```
============================================================
GENERATED CAPTION
============================================================

  📷  a dog is running through the grass

============================================================
```

## Configuration

All hyperparameters are defined in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `IMAGE_SIZE` | `(299, 299)` | Input resolution for InceptionV3 |
| `MAX_CAPTION_LENGTH` | `34` | Max tokens per caption |
| `EMBEDDING_DIM` | `256` | Word embedding size |
| `LSTM_UNITS` | `256` | LSTM hidden units |
| `DROPOUT_RATE` | `0.5` | Dropout probability |
| `BATCH_SIZE` | `32` | Training batch size |
| `EPOCHS` | `20` | Maximum training epochs |
| `VALIDATION_SPLIT` | `0.15` | Fraction held for validation |
| `EARLY_STOPPING_PATIENCE` | `5` | Epochs before early stop |

## Tech Stack

- **TensorFlow / Keras** — Model building, training, inference
- **InceptionV3** — Pretrained CNN feature extractor (ImageNet)
- **NumPy** — Numerical operations
- **Pillow** — Image loading and preprocessing
- **Matplotlib** — Loss curve visualization

## License

This project is for educational purposes.
