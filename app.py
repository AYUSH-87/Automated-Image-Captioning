"""
FastAPI Server for Automated Image Captioning.

Serves a clean REST API and the minimalist web frontend.
Pre-loads InceptionV3 and the trained LSTM decoder into memory for rapid inference.
"""

import io
import os
import sys
import time
from typing import Optional

import numpy as np
from PIL import Image
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore

import config
import utils
from contextlib import asynccontextmanager


# -----------------------------------------------------------------------------
# Global Model Registry
# -----------------------------------------------------------------------------
class ModelManager:
    def __init__(self):
        self.caption_model = None
        self.feature_extractor = None
        self.tokenizer = None
        self.index_to_word = {}
        self.max_len = config.MAX_CAPTION_LENGTH
        self.vocab_size = 0
        self.is_loaded = False

    def load_artifacts(self):
        if self.is_loaded:
            return

        print("[STARTUP] Loading trained caption model...")
        if not os.path.isfile(config.MODEL_FILE):
            raise FileNotFoundError(f"Model file not found: {config.MODEL_FILE}")
        self.caption_model = load_model(config.MODEL_FILE)

        print("[STARTUP] Loading tokenizer...")
        if not os.path.isfile(config.TOKENIZER_FILE):
            raise FileNotFoundError(f"Tokenizer file not found: {config.TOKENIZER_FILE}")
        self.tokenizer = utils.load_pickle(config.TOKENIZER_FILE)
        self.index_to_word = {
            idx: word for word, idx in self.tokenizer.word_index.items()
        }
        self.vocab_size = len(self.tokenizer.word_index) + 1

        print("[STARTUP] Loading InceptionV3 feature extractor...")
        self.feature_extractor = InceptionV3(
            weights="imagenet", include_top=False, pooling="avg"
        )
        for layer in self.feature_extractor.layers:
            layer.trainable = False

        self.max_len = self.caption_model.input_shape[1][1]
        self.is_loaded = True
        print(f"[STARTUP] Models ready. Max sequence length: {self.max_len}, Vocab: {self.vocab_size}")

        # Warm-up pass to eliminate initial JIT/graph compilation latency
        try:
            print("[STARTUP] Running warm-up inference pass...")
            dummy_img = np.zeros((1, 299, 299, 3), dtype=np.float32)
            dummy_feat = self.feature_extractor.predict(dummy_img, verbose=0)
            dummy_seq = np.zeros((1, self.max_len), dtype=np.int32)
            self.caption_model.predict([dummy_feat, dummy_seq], verbose=0)
            print("[STARTUP] Warm-up complete.")
        except Exception as e:
            print(f"[WARN] Warm-up pass failed (non-critical): {e}")

    def generate_caption(self, pil_image: Image.Image):
        """Generates caption and per-token confidence scores."""
        t0 = time.perf_counter()

        # Preprocess PIL image
        img = pil_image.convert("RGB").resize(config.IMAGE_SIZE)
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # CNN Feature extraction
        features = self.feature_extractor.predict(img_array, verbose=0)  # (1, 2048)

        # Greedy decoding
        tokens = [config.START_TOKEN]
        token_breakdown = []

        for _ in range(self.max_len):
            seq = self.tokenizer.texts_to_sequences([" ".join(tokens)])[0]
            seq = pad_sequences([seq], maxlen=self.max_len, padding="post")

            predictions = self.caption_model.predict([features, seq], verbose=0)
            predicted_index = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_index])
            predicted_word = self.index_to_word.get(predicted_index, None)

            if predicted_word is None or predicted_word == config.END_TOKEN:
                break

            tokens.append(predicted_word)
            token_breakdown.append({
                "token": predicted_word,
                "confidence": round(confidence * 100, 1),
                "step": len(token_breakdown) + 1
            })

        raw_caption = " ".join(tokens[1:])
        # Format caption: capitalize first letter, add punctuation
        formatted_caption = raw_caption.strip()
        if formatted_caption:
            formatted_caption = formatted_caption[0].upper() + formatted_caption[1:]
            if not formatted_caption.endswith((".", "!", "?")):
                formatted_caption += "."

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        avg_confidence = (
            round(sum(t["confidence"] for t in token_breakdown) / len(token_breakdown), 1)
            if token_breakdown
            else 0.0
        )

        return {
            "caption": formatted_caption,
            "raw_caption": raw_caption,
            "tokens": token_breakdown,
            "token_count": len(token_breakdown),
            "avg_confidence": avg_confidence,
            "latency_ms": latency_ms,
        }
model_mgr = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: pre-load trained models into RAM
    model_mgr.load_artifacts()
    yield

# -----------------------------------------------------------------------------
# App Initialization & CORS
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Neural Image Captioning API",
    description="Minimalist inference backend combining InceptionV3 and an LSTM decoder.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_mgr.is_loaded,
        "max_length": model_mgr.max_len,
        "vocab_size": model_mgr.vocab_size,
        "backbone": "InceptionV3 (2048-dim)",
        "decoder": "LSTM (256-units)",
    }


@app.get("/api/samples")
def get_sample_images():
    samples_dir = os.path.join(os.path.dirname(__file__), "frontend", "samples")
    if not os.path.exists(samples_dir):
        return {"samples": []}

    sample_meta = {
        "my_image.jpg": {
            "title": "Baseball Player",
            "category": "Sports",
            "description": "Athlete in action on field",
        },
        "1000268201_693b08cb0e.jpg": {
            "title": "Child on Stairs",
            "category": "People",
            "description": "Toddler exploring wooden staircase",
        },
        "1001773457_577c3a7d70.jpg": {
            "title": "Playing Dogs",
            "category": "Animals",
            "description": "Two dogs interacting outdoors",
        },
        "1002674143_1b742ab4b8.jpg": {
            "title": "Girl Painting",
            "category": "Art & Play",
            "description": "Child playing with finger paints",
        },
    }

    files = [f for f in os.listdir(samples_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    result = []
    for f in sorted(files):
        meta = sample_meta.get(f, {
            "title": f.split(".")[0].replace("_", " ").title(),
            "category": "Sample",
            "description": "Sample test photograph",
        })
        result.append({
            "id": f,
            "filename": f,
            "url": f"/frontend/samples/{f}",
            "title": meta["title"],
            "category": meta["category"],
            "description": meta["description"],
        })
    return {"samples": result}


class SampleRequest(BaseModel):
    filename: str


@app.post("/api/caption/sample")
def caption_sample(request: SampleRequest):
    samples_dir = os.path.join(os.path.dirname(__file__), "frontend", "samples")
    file_path = os.path.join(samples_dir, request.filename)

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"Sample file {request.filename} not found.")

    try:
        pil_img = Image.open(file_path)
        width, height = pil_img.size
        result = model_mgr.generate_caption(pil_img)
        result["image_info"] = {
            "source": request.filename,
            "width": width,
            "height": height,
            "format": pil_img.format or "JPEG",
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/caption")
async def caption_uploaded_file(file: UploadFile = File(...)):
    """Receives an uploaded image file, processes it, and returns the generated caption."""
    content_type = file.content_type or ""
    filename = (file.filename or "").lower()
    valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    if not (content_type.startswith("image/") or any(filename.endswith(ext) for ext in valid_exts)):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Please upload a valid image file (PNG, JPEG, WebP)."
        )

    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
        width, height = pil_img.size
        result = model_mgr.generate_caption(pil_img)
        result["image_info"] = {
            "source": file.filename or "upload.jpg",
            "width": width,
            "height": height,
            "format": pil_img.format or "JPEG",
            "size_kb": round(len(contents) / 1024, 1),
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image captioning error: {str(e)}")


# -----------------------------------------------------------------------------
# Static Files & Frontend Routing
# -----------------------------------------------------------------------------
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if not os.path.exists(frontend_dir):
    os.makedirs(frontend_dir, exist_ok=True)

app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")


@app.get("/")
def serve_index():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"status": "Frontend not found. Please create index.html in frontend/"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print("==================================================")
    print("  Starting Automated Image Captioning Web Server  ")
    print(f"  URL: http://localhost:{port}                   ")
    print("==================================================")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
