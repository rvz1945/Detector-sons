
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlparse
from contextlib import asynccontextmanager

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import yt_dlp

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from pydub import AudioSegment


MODEL_URL = "https://tfhub.dev/google/yamnet/1"
THRESHOLD = 0.70

TARGET_WORDS = (
    "burp",
    "burping",
    "belch",
    "belching",
    "cough",
    "throat clearing",
    "throat clear",
)

model = None
labels = []
analysis_lock = threading.Lock()


def load_labels():
    path = model.class_map_path().numpy().decode("utf-8")

    with open(path, newline="", encoding="utf-8") as file:
        return [
            row["display_name"]
            for row in csv.DictReader(file)
        ]


def check_ytdlp_tools():
    version = yt_dlp.version.__version__
    deno_path = shutil.which("deno")

    deno_version = "Deno não encontrado"

    if deno_path:
        try:
            result = subprocess.run(
                [deno_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            deno_version = (
                result.stdout.strip()
                or result.stderr.strip()
                or f"Comando terminou com código {result.returncode}"
            )
        except Exception as exc:
            deno_version = f"Erro ao verificar Deno: {exc}"

    return {
        "yt_dlp_version": version,
        "deno_path": deno_path,
        "deno_version": deno_version,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, labels

    model = hub.load(MODEL_URL)
    labels = load_labels()

    yield


app = FastAPI(
    title="Detector de Sons API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    url: HttpUrl


def validate_youtube_url(url):
    host = (urlparse(str(url)).hostname or "").lower()

    allowed = {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
        "www.youtu.be",
    }

    if host not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Envie um link válido do YouTube.",
        )


def download_audio(url, folder):
    output = str(Path(folder) / "audio.%(ext)s")

    options = {
        "format": "bestaudio/best",
        "outtmpl": output,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "max_filesize": 100 * 1024 * 1024,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "js_runtimes": {
            "deno": {},
        },
    }

    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            downloader.download([str(url)])

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Erro do yt-dlp: {type(exc).__name__}: {exc}"
            ),
        ) from exc

    audio_path = Path(folder) / "audio.wav"

    if not audio_path.exists():
        raise HTTPException(
            status_code=500,
            detail="O arquivo de áudio não foi gerado.",
        )

    return audio_path


def analyze_audio(audio_path):
    audio = (
        AudioSegment.from_file(str(audio_path))
        .set_channels(1)
        .set_frame_rate(16000)
        .set_sample_width(2)
    )

    samples = np.array(
        audio.get_array_of_samples(),
        dtype=np.float32,
    )

    samples = samples / 32768.0

    waveform = tf.convert_to_tensor(
        samples,
        dtype=tf.float32,
    )

    scores, embeddings, spectrogram = model(waveform)
    scores = scores.numpy()

    detections = []

    for frame_index, frame_scores in enumerate(scores):
        class_index = int(np.argmax(frame_scores))
        confidence = float(frame_scores[class_index])
        label = labels[class_index]

        if confidence < THRESHOLD:
            continue

        normalized_label = label.lower()

        if not any(
            word in normalized_label
            for word in TARGET_WORDS
        ):
            continue

        detections.append({
            "label": label,
            "timestamp": round(frame_index * 0.48, 2),
            "score": round(confidence, 4),
        })

    return detections


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Detector de Sons API funcionando",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "labels_loaded": len(labels),
    }


@app.get("/debug-tools")
def debug_tools():
    return check_ytdlp_tools()


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    validate_youtube_url(request.url)

    if model is None or not labels:
        raise HTTPException(
            status_code=503,
            detail="O modelo de áudio ainda está carregando.",
        )

    if not analysis_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail=(
                "Já existe uma análise em andamento. "
                "Tente novamente."
            ),
        )

    try:
        with tempfile.TemporaryDirectory() as folder:
            audio_path = download_audio(request.url, folder)
            return analyze_audio(audio_path)

    finally:
        analysis_lock.release()
