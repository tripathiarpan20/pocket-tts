# Deploying to RunPod Serverless

RunPod Serverless provides an excellent platform for deploying Pocket TTS as an on-demand API. You only pay for the compute time you use, making it cost-effective for variable workloads.

## Prerequisites

1. A [RunPod account](https://runpod.io)
2. Docker installed locally
3. A container registry (Docker Hub, or RunPod's built-in registry)

## Quick Start

### 1. Build the Docker Image

```bash
git clone https://github.com/tripathiarpan20/pocket-tts
cd pocket-tts

# Build the RunPod-specific image
docker build -t pocket-tts-runpod:latest -f Dockerfile.runpod .
```

The build process will pre-download the TTS model, which takes a few minutes but significantly reduces cold start time in production.

### 2. Push to Container Registry

**Option A: Docker Hub**

```bash
# Tag for Docker Hub
docker tag pocket-tts-runpod:latest YOUR_DOCKERHUB_USERNAME/pocket-tts-runpod:latest

# Push
docker push YOUR_DOCKERHUB_USERNAME/pocket-tts-runpod:latest
```

**Option B: RunPod Registry**

RunPod provides a built-in registry. Follow [RunPod's container registry guide](https://docs.runpod.io/serverless/workers/deploy) for details.

### 3. Create a Serverless Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Configure the endpoint:
   - **Container Image**: `YOUR_DOCKERHUB_USERNAME/pocket-tts-runpod:latest`
   - **GPU**: None (CPU-only, select a CPU worker type)
   - **Min Workers**: 0 (scales to zero when idle)
   - **Max Workers**: Set based on your expected load
4. Click **Deploy**

### 4. Test Your Endpoint

Once deployed, you can test your endpoint using the RunPod API:

```bash
# Get your endpoint ID and API key from the RunPod dashboard
ENDPOINT_ID="your-endpoint-id"
RUNPOD_API_KEY="your-api-key"

# Prepare a test request (replace with actual base64 audio)
curl -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text_transcription": "Hello world, this is a test.",
      "reference_audio": "<base64_encoded_audio_string>"
    }
  }'
```

## API Reference

### Request Format

**Endpoint**: `POST https://api.runpod.ai/v2/{endpoint_id}/runsync`

**Headers**:
```
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

**Body**:
```json
{
  "input": {
    "text_transcription": "The text you want to synthesize.",
    "reference_audio": "<base64_encoded_audio_file>"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text_transcription` | string | **Required**. The text to convert to speech. |
| `reference_audio` | string | **Required**. Base64-encoded audio file (WAV, MP3, etc.) for voice cloning. |

### Response Format

**Success**:
```json
{
  "id": "job-id",
  "status": "COMPLETED",
  "output": {
    "audio": "<base64_encoded_wav_audio>"
  }
}
```

**Error**:
```json
{
  "id": "job-id",
  "status": "COMPLETED",
  "output": {
    "error": "Error description"
  }
}
```

### Async Requests

For longer texts, use the async endpoint:

```bash
# Submit job
curl -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"input": {...}}'

# Returns: {"id": "job-id", "status": "IN_QUEUE"}

# Check status
curl "https://api.runpod.ai/v2/${ENDPOINT_ID}/status/${JOB_ID}" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}"
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POCKET_TTS_CONFIG` | Model configuration to use | `DEFAULT_VARIANT` |
| `HF_TOKEN` | HuggingFace token (if using private/gated models) | None |

Set environment variables in the RunPod endpoint configuration under **Environment Variables**.

## Local Testing

You can test the handler locally before deploying:

```bash
cd pocket-tts

# Install runpod SDK
uv add runpod

# Download a sample audio file for testing
wget "https://storage.googleapis.com/kagglesdsdata/datasets/829978/1417968/harvard.wav?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20260207%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260207T070105Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=ad77f7277b2f7e838ac91a6307fd0191965365b991436b8bfcac84b95351074921d7e5adb71c2cef13fb0056deb34f0d5dc0cb1f23d59022e0d66099050d93964fe42af01f8d480b35aa6214de73fc9c5be187d63d06aa284dd2c71cadfad9b4c78747acf148f070db0231158a110cd4ff496a5f0b0f76eba8aa10d24660427592138a60577a008629c455c7b8871a89d6ab0dd4c7e9ac5b8215a8d69d58a22192ba0f1855362e241d2b99a62371e8aade4d42c73eae7e3bd059b8c1d98e593fcfb43cc9b0b6b0e4de384b4a54732f7c3c00e45d2b80427d144202444bb7956668d32f942fbcd5546131289e2448f72688bc09d3514cabd69efd3d89643d33ad" -O "kaggle_sample.wav"

# Create a test input file with base64 encoded audio
echo '{
  "input": {
    "text_transcription": "Hello world, this is a test.",
    "reference_audio": "'$(base64 -w0 kaggle_sample.wav)'"
  }
}' > test_input.json

# Run the handler (it will auto-detect test_input.json)
uv run python pocket_tts/rp_handler.py
```

### Testing with Docker

```bash
# Build the image
docker build -t pocket-tts-runpod:latest -f Dockerfile.runpod .

# Run with mounted test input
docker run --rm -v $(pwd)/test_input.json:/app/test_input.json pocket-tts-runpod:latest python -u pocket_tts/rp_handler.py
```

## Troubleshooting

### Cold Start Time

The first request after the worker scales up may take longer due to model initialization. The Dockerfile pre-downloads the model to minimize this, but initial loading still takes ~10-20 seconds.

**Solution**: Set `Min Workers` to 1 to keep a worker always warm.

### Out of Memory

If you encounter memory issues, ensure you're using a worker with at least 4GB RAM.

### Base64 Encoding Issues

Ensure your audio file is properly base64-encoded:

```python
import base64

with open("reference.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")
```

### Model Download Fails

If the model fails to download during build, ensure you have a stable internet connection. For gated models, set the `HF_TOKEN` build argument:

```bash
docker build --build-arg HF_TOKEN=your_token -t pocket-tts-runpod:latest -f Dockerfile.runpod .
```

## Cost Optimization

1. **Scale to Zero**: Set `Min Workers` to 0 to avoid charges during idle periods
2. **Use CPU Workers**: Pocket TTS is optimized for CPU, no GPU needed
3. **Batch Requests**: If possible, batch multiple TTS requests to reuse the warm worker

## Python Client Example

```python
import base64
import requests

ENDPOINT_ID = "your-endpoint-id"
API_KEY = "your-api-key"

def generate_speech(text: str, reference_audio_path: str) -> bytes:
    """Generate speech using RunPod-hosted Pocket TTS."""
    
    # Read and encode reference audio
    with open(reference_audio_path, "rb") as f:
        reference_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    # Make request
    response = requests.post(
        f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "input": {
                "text_transcription": text,
                "reference_audio": reference_b64
            }
        }
    )
    
    result = response.json()
    
    if "error" in result.get("output", {}):
        raise Exception(result["output"]["error"])
    
    # Decode the generated audio
    audio_b64 = result["output"]["audio"]
    return base64.b64decode(audio_b64)

# Usage
audio_bytes = generate_speech(
    "Hello world, this is a test.",
    "reference_voice.wav"
)

with open("output.wav", "wb") as f:
    f.write(audio_bytes)
```
