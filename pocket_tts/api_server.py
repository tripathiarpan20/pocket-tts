
import asyncio
import base64
import io
import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import soundfile as sf

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.default_parameters import DEFAULT_VARIANT

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response Models
class TTSRequest(BaseModel):
    text_transcription: str
    reference_audio: str  # Base64 encoded audio string

class TTSResponse(BaseModel):
    audio: str  # Base64 encoded audio string

# Global State
model_lock = asyncio.Lock()
request_queue = asyncio.Queue()
tts_model: Optional[TTSModel] = None

async def process_queue():
    """Background task to process requests from the queue sequentially."""
    logger.info("Starting background worker for TTS queue.")
    while True:
        request_data = await request_queue.get()
        future, request = request_data
        
        try:
            # Process the request
            result = await asyncio.to_thread(process_tts_request, request)
            if not future.done():
                future.set_result(result)
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            if not future.done():
                future.set_exception(e)
        finally:
            request_queue.task_done()

def process_tts_request(request: TTSRequest) -> str:
    """Synchronous function to run the model inference."""
    global tts_model
    if tts_model is None:
        raise RuntimeError("Model not initialized")

    # 1. Decode base64 audio
    try:
        audio_bytes = base64.b64decode(request.reference_audio)
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
        
        # Resample if necessary (TTSModel expects specific handling, but get_state_for_audio_prompt handles tensors)
        # Note: get_state_for_audio_prompt calls convert_audio internally if passing Path,
        # but if passing Tensor, it assumes it matches self.config.mimi.sample_rate?
        # Let's check TTSModel.get_state_for_audio_prompt impl.
        # It calls _encode_audio directly on the tensor. _encode_audio expects 24kHz usually.
        # We should probably use convert_audio if the sample rate differs.
        
        from pocket_tts.data.audio_utils import convert_audio
        target_sr = tts_model.config.mimi.sample_rate
        if sample_rate != target_sr:
             audio_tensor = convert_audio(audio_tensor, sample_rate, target_sr, 1)

    except Exception as e:
        raise ValueError(f"Failed to process audio data: {e}")

    # 3. Get model state
    # We don't truncate by default unless specified, but user didn't specify. 
    # Let's assume we proceed.
    model_state = tts_model.get_state_for_audio_prompt(audio_tensor)

    # 4. Generate audio
    # generate_audio returns a tensor [1, T] 
    generated_audio = tts_model.generate_audio(
        model_state=model_state,
        text_to_generate=request.text_transcription
    )

    # 5. Convert generated tensor back to WAV bytes
    # generated_audio is [channels, samples]
    # We need to write to bytes.
    output_io = io.BytesIO()
    # Convert back to numpy
    generated_numpy = generated_audio.cpu().numpy().T # [samples, channels]
    
    # Write to buffer
    sf.write(output_io, generated_numpy, tts_model.config.mimi.sample_rate, format='WAV')
    output_io.seek(0)
    
    # 6. Encode to base64
    return base64.b64encode(output_io.read()).decode("utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model
    global tts_model
    logger.info("Loading TTS Model...")
    
    import os
    config = os.environ.get("POCKET_TTS_CONFIG", DEFAULT_VARIANT)
    tts_model = TTSModel.load_model(config)
    logger.info(f"TTS Model loaded with config: {config}")
    
    # Start worker
    worker_task = asyncio.create_task(process_queue())
    
    yield
    
    # Cleanup
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)

@app.post("/generate", response_model=TTSResponse)
async def generate(request: TTSRequest):
    future = asyncio.get_running_loop().create_future()
    await request_queue.put((future, request))
    try:
        result = await future
        return TTSResponse(audio=result)
    except Exception as e:
        if isinstance(e, ValueError):
             raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "queue_size": request_queue.qsize()}

def start_server(host: str = "0.0.0.0", port: int = 8001, config: str = DEFAULT_VARIANT):
    import uvicorn
    # We need to pass the config to lifespan potentially, but lifespan uses global.
    # We can handle this by pre-loading or setting an env var.
    # Or simplified: just load it before running uvicorn if valid.
    # Actually, uvicorn runs in a separate process if using reload, but here we run programmatically.
    
    # Let's set the global tts_model here if we are not using lifespan to load it OR 
    # relying on lifespan to load it. 
    # Since start_server is called by CLI, we can just run uvicorn. 
    # But wait, lifespan is better for proper async startup.
    # To pass config, we can set a global var in this module before starting.
    
    # Hack: set a module-level variable for config that lifespan can read if needed.
    # But simpler: The lifespan loads DEFAULT_VARIANT. 
    # If we want to support config override, we need to pass it.
    
    # Let's use a closure or partial if possible, but FastAPI lifespan doesn't easily support args.
    # We can use os.environ.
    import os
    os.environ["POCKET_TTS_CONFIG"] = config
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
