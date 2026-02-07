"""
RunPod Serverless Handler for Pocket TTS.

This module provides a RunPod serverless handler for the Pocket TTS model.
It loads the model at module level (outside the handler) to avoid reloading
on each request, following RunPod best practices.

Usage:
    # Local testing
    python rp_handler.py --test_input '{"input": {"text_transcription": "Hello", "reference_audio": "<base64>"}}'

    # Or with test_input.json file in the same directory
    python rp_handler.py
"""

import base64
import io
import logging
import os

import runpod
import soundfile as sf
import torch

from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.default_parameters import DEFAULT_VARIANT
from pocket_tts.models.tts_model import TTSModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Model Loading (outside handler for RunPod best practices)
# ============================================================================

logger.info("Loading TTS Model...")
config = os.environ.get("POCKET_TTS_CONFIG", DEFAULT_VARIANT)
tts_model: TTSModel = TTSModel.load_model(config)
logger.info(f"TTS Model loaded successfully with config: {config}")


def validate_input(job_input: dict) -> tuple[str, str]:
    """
    Validate the input from the job request.
    
    Args:
        job_input: Dictionary containing 'text_transcription' and 'reference_audio'
        
    Returns:
        Tuple of (text_transcription, reference_audio)
        
    Raises:
        ValueError: If required fields are missing or invalid
    """
    if not isinstance(job_input, dict):
        raise ValueError("Input must be a dictionary")
    
    text_transcription = job_input.get("text_transcription")
    if not text_transcription:
        raise ValueError("'text_transcription' is required and cannot be empty")
    
    if not isinstance(text_transcription, str):
        raise ValueError("'text_transcription' must be a string")
    
    reference_audio = job_input.get("reference_audio")
    if not reference_audio:
        raise ValueError("'reference_audio' is required and cannot be empty")
    
    if not isinstance(reference_audio, str):
        raise ValueError("'reference_audio' must be a base64-encoded string")
    
    return text_transcription, reference_audio


def process_tts(text_transcription: str, reference_audio_b64: str) -> str:
    """
    Process the TTS request and generate audio.
    
    Args:
        text_transcription: Text to synthesize
        reference_audio_b64: Base64-encoded reference audio for voice cloning
        
    Returns:
        Base64-encoded WAV audio of the generated speech
    """
    # 1. Decode base64 audio
    try:
        audio_bytes = base64.b64decode(reference_audio_b64)
    except Exception as e:
        raise ValueError(f"Invalid base64 encoding: {e}")

    # 2. Load audio to tensor using soundfile
    try:
        with io.BytesIO(audio_bytes) as audio_io:
            data, sample_rate = sf.read(audio_io, dtype="float32")
        
        # Handle channels (convert to mono)
        if data.ndim > 1:
            data = data.mean(axis=1)
        
        # Convert to tensor [channels, samples]
        audio_tensor = torch.from_numpy(data).unsqueeze(0)
        
        # Resample if necessary
        target_sr = tts_model.config.mimi.sample_rate
        if sample_rate != target_sr:
            audio_tensor = convert_audio(audio_tensor, sample_rate, target_sr, 1)

    except Exception as e:
        raise ValueError(f"Failed to process audio data: {e}")

    # 3. Get model state from reference audio
    model_state = tts_model.get_state_for_audio_prompt(audio_tensor)

    # 4. Generate audio
    generated_audio = tts_model.generate_audio(
        model_state=model_state,
        text_to_generate=text_transcription
    )

    # 5. Convert generated tensor back to WAV bytes
    output_io = io.BytesIO()
    generated_numpy = generated_audio.cpu().numpy().T  # [samples, channels]
    sf.write(output_io, generated_numpy, tts_model.config.mimi.sample_rate, format='WAV')
    output_io.seek(0)
    
    # 6. Encode to base64
    return base64.b64encode(output_io.read()).decode("utf-8")


def handler(job: dict) -> dict:
    """
    RunPod serverless handler function.
    
    Args:
        job: Dictionary containing 'id' and 'input' keys
        
    Returns:
        Dictionary with 'audio' key containing base64-encoded WAV
    """
    job_input = job.get("input", {})
    
    try:
        # Validate input
        text_transcription, reference_audio = validate_input(job_input)
        
        logger.info(f"Processing TTS request: {len(text_transcription)} chars")
        
        # Process TTS
        audio_b64 = process_tts(text_transcription, reference_audio)
        
        logger.info("TTS generation completed successfully")
        
        return {"audio": audio_b64}
        
    except ValueError as e:
        # Input validation errors
        logger.error(f"Validation error: {e}")
        return {"error": str(e)}
    except Exception as e:
        # Unexpected errors
        logger.error(f"Error processing request: {e}", exc_info=True)
        return {"error": f"Internal error: {str(e)}"}


# Start the serverless worker
runpod.serverless.start({"handler": handler})
