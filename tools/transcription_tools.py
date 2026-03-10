#!/usr/bin/env python3
"""
Transcription Tools Module

Provides speech-to-text transcription using faster-whisper (local) 
with optional fallback to OpenAI's Whisper API.
Used by the messaging gateway to automatically transcribe voice messages
sent by users on Telegram, Discord, WhatsApp, and Slack.

Local models (faster-whisper):
  - tiny, tiny.en      (fastest, lowest quality)
  - base, base.en
  - small, small.en
  - medium, medium.en
  - large-v3           (best quality, slower)

OpenAI fallback models:
  - whisper-1          (cheapest, good quality)
  - gpt-4o-mini-transcribe  (better quality, higher cost)
  - gpt-4o-transcribe       (best quality, highest cost)

Supported input formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, ogg

Usage:
    from tools.transcription_tools import transcribe_audio

    result = transcribe_audio("/path/to/audio.ogg")
    if result["success"]:
        print(result["transcript"])
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Default local model (good balance of speed/quality)
DEFAULT_LOCAL_MODEL = os.getenv("WHISPER_LOCAL_MODEL", "base")

# Default OpenAI model for fallback
DEFAULT_OPENAI_MODEL = "whisper-1"

# Supported audio formats
SUPPORTED_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg"}

# Maximum file size (25MB - OpenAI limit, but also reasonable for local)
MAX_FILE_SIZE = 25 * 1024 * 1024

# Cache for loaded whisper model
_whisper_model = None


def _get_local_model(model_size: str = None):
    """Load or return cached faster-whisper model."""
    global _whisper_model
    
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            model_name = model_size or DEFAULT_LOCAL_MODEL
            logger.info("Loading faster-whisper model: %s", model_name)
            # Use CPU for broader compatibility, can be configured via env var
            device = os.getenv("WHISPER_DEVICE", "cpu")
            compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
            _whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)
            logger.info("faster-whisper model loaded successfully")
        except ImportError as e:
            logger.warning("faster-whisper not available: %s", e)
            return None
        except Exception as e:
            logger.error("Failed to load faster-whisper model: %s", e, exc_info=True)
            return None
    
    return _whisper_model


def _transcribe_local(file_path: str, model_size: str = None) -> Dict[str, Any]:
    """Transcribe using local faster-whisper."""
    model = _get_local_model(model_size)
    if model is None:
        return {
            "success": False,
            "transcript": "",
            "error": "faster-whisper not available",
        }
    
    try:
        segments, info = model.transcribe(file_path, beam_size=5)
        transcript_text = "".join(segment.text for segment in segments).strip()
        
        logger.info(
            "Local transcription of %s: %d chars (%.2fs audio, %.2fs processing)",
            Path(file_path).name,
            len(transcript_text),
            info.duration,
            info.transcription_time if hasattr(info, 'transcription_time') else 0,
        )
        
        return {
            "success": True,
            "transcript": transcript_text,
            "method": "faster-whisper",
        }
    except Exception as e:
        logger.error("Local transcription failed: %s", e, exc_info=True)
        return {
            "success": False,
            "transcript": "",
            "error": f"Local transcription failed: {e}",
        }


def _transcribe_openai(file_path: str, model: str = None) -> Dict[str, Any]:
    """Transcribe using OpenAI API (fallback)."""
    api_key = os.getenv("VOICE_TOOLS_OPENAI_KEY")
    if not api_key:
        return {
            "success": False,
            "transcript": "",
            "error": "VOICE_TOOLS_OPENAI_KEY not set",
        }
    
    if model is None:
        model = DEFAULT_OPENAI_MODEL
    
    try:
        from openai import OpenAI, APIError, APIConnectionError, APITimeoutError

        client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")

        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="text",
            )

        transcript_text = str(transcription).strip()

        logger.info("OpenAI transcription of %s: %d chars", Path(file_path).name, len(transcript_text))

        return {
            "success": True,
            "transcript": transcript_text,
            "method": "openai",
        }

    except PermissionError:
        return {"success": False, "transcript": "", "error": f"Permission denied: {file_path}"}
    except Exception as e:
        logger.error("OpenAI transcription failed: %s", e, exc_info=True)
        return {"success": False, "transcript": "", "error": f"OpenAI error: {e}"}


def transcribe_audio(file_path: str, model: Optional[str] = None, prefer_local: bool = True) -> Dict[str, Any]:
    """
    Transcribe an audio file using faster-whisper (local) with OpenAI fallback.

    Args:
        file_path:   Absolute path to the audio file to transcribe.
        model:       Model to use. For local: tiny/base/small/medium/large-v3.
                     For OpenAI: whisper-1, gpt-4o-mini-transcribe, gpt-4o-transcribe.
        prefer_local: If True (default), try local transcription first.

    Returns:
        dict with keys:
          - "success" (bool): Whether transcription succeeded
          - "transcript" (str): The transcribed text (empty on failure)
          - "error" (str, optional): Error message if success is False
          - "method" (str, optional): "faster-whisper" or "openai"
    """
    audio_path = Path(file_path)
    
    # Validate file exists
    if not audio_path.exists():
        return {"success": False, "transcript": "", "error": f"Audio file not found: {file_path}"}
    
    if not audio_path.is_file():
        return {"success": False, "transcript": "", "error": f"Path is not a file: {file_path}"}
    
    # Validate file extension
    if audio_path.suffix.lower() not in SUPPORTED_FORMATS:
        return {
            "success": False,
            "transcript": "",
            "error": f"Unsupported format: {audio_path.suffix}. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
        }
    
    # Validate file size
    try:
        file_size = audio_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            return {
                "success": False,
                "transcript": "",
                "error": f"File too large: {file_size / (1024*1024):.1f}MB (max {MAX_FILE_SIZE / (1024*1024)}MB)",
            }
    except OSError as e:
        return {"success": False, "transcript": "", "error": f"Failed to access file: {e}"}

    # Try local transcription first if preferred
    if prefer_local:
        result = _transcribe_local(file_path, model)
        if result["success"]:
            return result
        logger.info("Local transcription failed, trying OpenAI fallback: %s", result.get("error"))
    
    # Fallback to OpenAI
    return _transcribe_openai(file_path, model)
