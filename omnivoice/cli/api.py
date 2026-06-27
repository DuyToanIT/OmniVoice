#!/usr/bin/env python3
# Copyright    2026  OmniVoice Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FastAPI REST API server for OmniVoice with Swagger UI.

Provides REST endpoints for text-to-speech synthesis, supporting
voice cloning, voice design, and auto voice modes.

Swagger UI is available at /docs.
ReDoc is available at /redoc.

Usage:
    omnivoice-api --model k2-fsa/OmniVoice --port 8000
"""

import argparse
import io
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_IDS, LANG_NAMES, lang_display_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state — model is loaded once at startup via lifespan
# ---------------------------------------------------------------------------
_model: Optional[OmniVoice] = None
_model_name: str = "k2-fsa/OmniVoice"
_device: str = "cpu"
_load_asr: bool = True
_asr_model: str = "openai/whisper-large-v3-turbo"


# ---------------------------------------------------------------------------
# Pydantic schemas for request/response
# ---------------------------------------------------------------------------


class AudioFormat(str, Enum):
    wav = "wav"
    flac = "flac"
    mp3 = "mp3"


class TTSRequest(BaseModel):
    """Request body for the text-to-speech synthesis endpoint."""

    text: str = Field(
        ...,
        description="Text to synthesize into speech.",
        min_length=1,
        max_length=5000,
        examples=["Hello, this is a test of OmniVoice text-to-speech."],
    )
    language: Optional[str] = Field(
        None,
        description=(
            "Language name (e.g. 'English', 'Chinese') or ISO code (e.g. 'en', 'zh'). "
            "Leave empty for auto-detection."
        ),
        examples=["English", "en", "Chinese"],
    )
    instruct: Optional[str] = Field(
        None,
        description=(
            "Voice design instruction. Comma-separated attributes: "
            "gender (male/female), age (child/teenager/young adult/middle-aged/elderly), "
            "pitch (very low/low/moderate/high/very high), style (whisper), "
            "accent (American/British/etc.)."
        ),
        examples=["female, low pitch, british accent"],
    )
    num_step: int = Field(
        32,
        ge=4,
        le=64,
        description="Number of diffusion steps. Lower = faster, higher = better quality.",
    )
    guidance_scale: float = Field(
        2.0,
        ge=0.0,
        le=4.0,
        description="Classifier-free guidance scale.",
    )
    speed: Optional[float] = Field(
        None,
        ge=0.5,
        le=2.0,
        description="Speaking speed factor. >1.0 faster, <1.0 slower. Ignored if duration is set.",
    )
    duration: Optional[float] = Field(
        None,
        ge=0.1,
        le=120.0,
        description="Fixed output duration in seconds. Overrides speed when set.",
    )
    denoise: bool = Field(
        True,
        description="Enable denoising for cleaner output.",
    )
    postprocess_output: bool = Field(
        True,
        description="Post-process output (remove long silences, apply fade).",
    )
    format: AudioFormat = Field(
        AudioFormat.wav,
        description="Output audio format.",
    )


class TTSResponse(BaseModel):
    """Response metadata for TTS generation (audio is returned as binary)."""

    message: str = Field(..., description="Status message.")
    sampling_rate: int = Field(..., description="Audio sampling rate in Hz.")
    duration_seconds: float = Field(..., description="Generated audio duration in seconds.")
    format: str = Field(..., description="Audio format.")


class ModelInfoResponse(BaseModel):
    """Model information and server status."""

    model_name: str = Field(..., description="Loaded model name or path.")
    version: str = Field(..., description="OmniVoice package version.")
    device: str = Field(..., description="Compute device (cuda, cpu, mps, xpu).")
    sampling_rate: int = Field(..., description="Audio output sampling rate in Hz.")
    asr_loaded: bool = Field(..., description="Whether the ASR model is loaded.")
    num_languages: int = Field(..., description="Number of supported languages.")


class LanguageInfo(BaseModel):
    """Information about a supported language."""

    name: str = Field(..., description="Language display name.")
    code: str = Field(..., description="Language ISO code.")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Server health status.")
    model_loaded: bool = Field(..., description="Whether the model is loaded.")


class VoiceDesignAttributes(BaseModel):
    """Available voice design attributes."""

    gender: list[str] = Field(..., description="Available gender options.")
    age: list[str] = Field(..., description="Available age options.")
    pitch: list[str] = Field(..., description="Available pitch options.")
    style: list[str] = Field(..., description="Available style options.")
    english_accent: list[str] = Field(..., description="Available English accent options.")
    chinese_dialect: list[str] = Field(..., description="Available Chinese dialect options.")


# ---------------------------------------------------------------------------
# FastAPI lifespan — load model on startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the OmniVoice model on startup, clean up on shutdown."""
    global _model

    logger.info("Loading OmniVoice model: %s on %s ...", _model_name, _device)
    _model = OmniVoice.from_pretrained(
        _model_name,
        device_map=_device,
        dtype=torch.float16,
        load_asr=_load_asr,
        asr_model_name=_asr_model,
    )
    logger.info("Model loaded successfully. Sampling rate: %d Hz", _model.sampling_rate)

    yield

    # Cleanup
    logger.info("Shutting down, releasing model...")
    _model = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OmniVoice API",
    description=(
        "🌍 **OmniVoice** — State-of-the-art multilingual zero-shot Text-to-Speech API.\n\n"
        "Supports **600+ languages**, voice cloning, voice design, and auto voice modes.\n\n"
        "## Features\n"
        "- **Voice Cloning**: Clone any voice from a short reference audio (3-10s)\n"
        "- **Voice Design**: Create custom voices via text description (gender, age, pitch, accent)\n"
        "- **Auto Voice**: Let the model pick a voice automatically\n"
        "- **Non-verbal Control**: Insert `[laughter]`, `[sigh]`, etc.\n"
        "- **Pronunciation Control**: Fine-tune pronunciation via pinyin or CMU phonemes\n\n"
        "## Quick Links\n"
        "- [GitHub](https://github.com/k2-fsa/OmniVoice)\n"
        "- [Paper](https://arxiv.org/abs/2604.00688)\n"
        "- [HuggingFace](https://huggingface.co/k2-fsa/OmniVoice)\n"
    ),
    version="0.1.5",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — allow all origins for development
# TODO(security): In production, restrict allowed_origins to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get_model() -> OmniVoice:
    """Get the loaded model or raise an error."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return _model


def _audio_to_bytes(
    audio: np.ndarray, sampling_rate: int, fmt: AudioFormat
) -> tuple[bytes, str]:
    """Encode numpy audio array to bytes in the requested format."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sampling_rate, format=fmt.value)
    buffer.seek(0)

    content_types = {
        AudioFormat.wav: "audio/wav",
        AudioFormat.flac: "audio/flac",
        AudioFormat.mp3: "audio/mpeg",
    }
    return buffer.read(), content_types[fmt]


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="Check if the server and model are ready.",
)
async def health_check():
    return HealthResponse(
        status="ok" if _model is not None else "loading",
        model_loaded=_model is not None,
    )


@app.get(
    "/v1/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Get model information",
    description="Returns information about the loaded model, device, and capabilities.",
)
async def get_model_info():
    model = _get_model()
    from omnivoice import __version__

    return ModelInfoResponse(
        model_name=_model_name,
        version=__version__,
        device=str(model.device),
        sampling_rate=model.sampling_rate,
        asr_loaded=model._asr_pipe is not None,
        num_languages=len(LANG_NAMES),
    )


@app.get(
    "/v1/languages",
    response_model=list[LanguageInfo],
    tags=["Model"],
    summary="List supported languages",
    description="Returns all 600+ supported languages with their names and ISO codes.",
)
async def list_languages():
    _get_model()  # Ensure model is loaded
    languages = []
    for name in sorted(LANG_NAMES):
        # Find the corresponding ID
        display = lang_display_name(name)
        # Extract code from display name if available, otherwise use name
        languages.append(LanguageInfo(name=display, code=name))
    return languages


@app.get(
    "/v1/voice-design/attributes",
    response_model=VoiceDesignAttributes,
    tags=["Voice Design"],
    summary="List voice design attributes",
    description=(
        "Returns all available attributes for voice design mode. "
        "Combine attributes from different categories in the `instruct` parameter."
    ),
)
async def get_voice_design_attributes():
    return VoiceDesignAttributes(
        gender=["male", "female"],
        age=["child", "teenager", "young adult", "middle-aged", "elderly"],
        pitch=["very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"],
        style=["whisper"],
        english_accent=[
            "American accent",
            "Australian accent",
            "British accent",
            "Chinese accent",
            "Canadian accent",
            "Indian accent",
            "Korean accent",
            "Portuguese accent",
            "Russian accent",
            "Japanese accent",
        ],
        chinese_dialect=[
            "河南话",
            "陕西话",
            "四川话",
            "贵州话",
            "云南话",
            "桂林话",
            "济南话",
            "石家庄话",
            "甘肃话",
            "宁夏话",
            "青岛话",
            "东北话",
        ],
    )


@app.post(
    "/v1/tts",
    tags=["Text-to-Speech"],
    summary="Synthesize speech (JSON body)",
    description=(
        "Generate speech audio from text using voice design or auto voice mode.\n\n"
        "**Modes**:\n"
        "- **Voice Design**: Set `instruct` to describe the voice (e.g., 'female, british accent')\n"
        "- **Auto Voice**: Leave `instruct` empty — the model picks a voice automatically\n\n"
        "For voice cloning with a reference audio, use the `/v1/tts/clone` endpoint instead.\n\n"
        "Returns the generated audio file directly."
    ),
    response_class=Response,
    responses={
        200: {
            "description": "Generated audio file",
            "content": {
                "audio/wav": {},
                "audio/flac": {},
                "audio/mpeg": {},
            },
        },
        422: {"description": "Validation error"},
        503: {"description": "Model not loaded"},
    },
)
async def synthesize_speech(request: TTSRequest):
    model = _get_model()

    gen_config = OmniVoiceGenerationConfig(
        num_step=request.num_step,
        guidance_scale=request.guidance_scale,
        denoise=request.denoise,
        postprocess_output=request.postprocess_output,
    )

    kwargs = {
        "text": request.text,
        "language": request.language,
        "instruct": request.instruct,
        "generation_config": gen_config,
    }
    if request.speed is not None:
        kwargs["speed"] = request.speed
    if request.duration is not None:
        kwargs["duration"] = request.duration

    try:
        audios = model.generate(**kwargs)
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    audio = audios[0]
    audio_bytes, content_type = _audio_to_bytes(audio, model.sampling_rate, request.format)

    duration_seconds = round(len(audio) / model.sampling_rate, 3)
    filename = f"omnivoice_{uuid.uuid4().hex[:8]}.{request.format.value}"

    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Audio-Sampling-Rate": str(model.sampling_rate),
            "X-Audio-Duration-Seconds": str(duration_seconds),
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.post(
    "/v1/tts/clone",
    tags=["Text-to-Speech"],
    summary="Synthesize speech with voice cloning",
    description=(
        "Clone a voice from a short reference audio (3-10 seconds recommended) "
        "and synthesize the target text in that voice.\n\n"
        "Upload the reference audio as a file. Optionally provide the "
        "reference text transcript (auto-transcribed via Whisper if omitted).\n\n"
        "Returns the generated audio file directly."
    ),
    response_class=Response,
    responses={
        200: {
            "description": "Generated audio file",
            "content": {
                "audio/wav": {},
                "audio/flac": {},
                "audio/mpeg": {},
            },
        },
        422: {"description": "Validation error"},
        503: {"description": "Model not loaded"},
    },
)
async def synthesize_with_clone(
    ref_audio: UploadFile = File(
        ...,
        description="Reference audio file (WAV, FLAC, MP3). 3-10 seconds recommended.",
    ),
    text: str = Form(
        ...,
        description="Text to synthesize into speech.",
        min_length=1,
        max_length=5000,
    ),
    ref_text: Optional[str] = Form(
        None,
        description="Transcript of reference audio. Auto-transcribed if omitted.",
    ),
    language: Optional[str] = Form(
        None,
        description="Language name or ISO code. Leave empty for auto-detection.",
    ),
    instruct: Optional[str] = Form(
        None,
        description="Additional voice design instruction (optional).",
    ),
    num_step: int = Form(32, ge=4, le=64, description="Diffusion steps."),
    guidance_scale: float = Form(2.0, ge=0.0, le=4.0, description="CFG scale."),
    speed: Optional[float] = Form(None, ge=0.5, le=2.0, description="Speed factor."),
    duration: Optional[float] = Form(None, ge=0.1, le=120.0, description="Fixed duration (seconds)."),
    denoise: bool = Form(True, description="Enable denoising."),
    postprocess_output: bool = Form(True, description="Post-process output."),
    format: AudioFormat = Form(AudioFormat.wav, description="Output format."),
):
    model = _get_model()

    # Validate file type by extension
    allowed_extensions = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".wma"}
    if ref_audio.filename:
        ext = os.path.splitext(ref_audio.filename)[1].lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported audio format: {ext}. Allowed: {', '.join(sorted(allowed_extensions))}",
            )

    # Validate file size (max 50MB)
    content = await ref_audio.read()
    max_size = 50 * 1024 * 1024  # 50MB
    if len(content) > max_size:
        raise HTTPException(
            status_code=422,
            detail=f"Reference audio file too large. Maximum size: {max_size // (1024*1024)}MB.",
        )

    # Save to a temp file for the model to read
    suffix = os.path.splitext(ref_audio.filename or ".wav")[1]
    # Use a UUID-based filename to prevent path traversal
    safe_filename = f"{uuid.uuid4().hex}{suffix}"
    tmp_path = os.path.join(tempfile.gettempdir(), safe_filename)
    try:
        with open(tmp_path, "wb") as f:
            f.write(content)

        gen_config = OmniVoiceGenerationConfig(
            num_step=num_step,
            guidance_scale=guidance_scale,
            denoise=denoise,
            postprocess_output=postprocess_output,
        )

        kwargs = {
            "text": text,
            "language": language,
            "ref_audio": tmp_path,
            "ref_text": ref_text,
            "instruct": instruct,
            "generation_config": gen_config,
        }
        if speed is not None:
            kwargs["speed"] = speed
        if duration is not None:
            kwargs["duration"] = duration

        try:
            audios = model.generate(**kwargs)
        except Exception as e:
            logger.exception("Voice cloning generation failed")
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    finally:
        # Always clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    audio = audios[0]
    audio_bytes, content_type = _audio_to_bytes(audio, model.sampling_rate, format)

    duration_seconds = round(len(audio) / model.sampling_rate, 3)
    filename = f"omnivoice_clone_{uuid.uuid4().hex[:8]}.{format.value}"

    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Audio-Sampling-Rate": str(model.sampling_rate),
            "X-Audio-Duration-Seconds": str(duration_seconds),
            "X-Content-Type-Options": "nosniff",
        },
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-api",
        description="Launch OmniVoice REST API server with Swagger UI.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1). "
        "Note: Each worker loads its own model copy.",
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper ASR model.",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="ASR model for auto-transcription.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload for development.",
    )
    return parser


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args(argv)

    # Set module-level config (read by lifespan)
    global _model_name, _device, _load_asr, _asr_model
    _model_name = args.model
    _device = args.device or get_best_device()
    _load_asr = not args.no_asr
    _asr_model = args.asr_model

    logger.info("Starting OmniVoice API server...")
    logger.info("  Model:   %s", _model_name)
    logger.info("  Device:  %s", _device)
    logger.info("  Host:    %s:%d", args.host, args.port)
    logger.info("  Swagger: http://%s:%d/docs", args.host, args.port)

    uvicorn.run(
        "omnivoice.cli.api:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
