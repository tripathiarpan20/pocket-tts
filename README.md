# Pocket TTS

<img width="1446" height="622" alt="pocket-tts-logo-v2-transparent" src="https://github.com/user-attachments/assets/637b5ed6-831f-4023-9b4c-741be21ab238" />

A lightweight text-to-speech (TTS) application designed to run efficiently on CPUs.
Forget about the hassle of using GPUs and web APIs serving TTS models. With Kyutai's Pocket TTS, generating audio is just a pip install and a function call away.

Supports Python 3.10, 3.11, 3.12, 3.13 and 3.14. Requires PyTorch 2.5+. Does not require the gpu version of PyTorch.

[🔊 Demo](https://kyutai.org/pocket-tts) | 
[🐱‍💻GitHub Repository](https://github.com/kyutai-labs/pocket-tts) | 
[🤗 Hugging Face Model Card](https://huggingface.co/kyutai/pocket-tts) | 
[⚙️ Tech report](https://kyutai.org/blog/2026-01-13-pocket-tts) |
[📄 Paper](https://arxiv.org/abs/2509.06926) | 
[📚 Documentation](https://github.com/kyutai-labs/pocket-tts/tree/main/docs)


## Main takeaways
* Runs on CPU
* Small model size, 100M parameters
* Audio streaming
* Low latency, ~200ms to get the first audio chunk
* Faster than real-time, ~6x real-time on a CPU of MacBook Air M4
* Uses only 2 CPU cores
* Python API and CLI
* Voice cloning
* English only at the moment
* Can handle infinitely long text inputs
* [Can run on client-side in the browser](#in-browser-implementations)

## Trying it from the website, without installing anything

Navigate to the [Kyutai website](https://kyutai.org/pocket-tts) to try it out directly in your browser. You can input text, select different voices, and generate speech without any installation.

## Trying it with the CLI

### The `generate` command
You can use pocket-tts directly from the command line. We recommend using
`uv` as it installs any dependencies on the fly in an isolated environment (uv installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)).
You can also use `pip install pocket-tts` to install it manually.

This will generate a wav file `./tts_output.wav` saying the default text with the default voice, and display some speed statistics.
```bash
uvx pocket-tts generate
# or if you installed it manually with pip:
pocket-tts generate
```
Modify the voice with `--voice` and the text with `--text`. We provide a small catalog of voices.

You can take a look at [this page](https://huggingface.co/kyutai/tts-voices) which details the licenses
for each voice.

* [alba](https://huggingface.co/kyutai/tts-voices/blob/main/alba-mackenna/casual.wav)
* [marius](https://huggingface.co/kyutai/tts-voices/blob/main/voice-donations/Selfie.wav)
* [javert](https://huggingface.co/kyutai/tts-voices/blob/main/voice-donations/Butter.wav)
* [jean](https://huggingface.co/kyutai/tts-voices/blob/main/ears/p010/freeform_speech_01.wav)
* [fantine](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p244_023.wav)
* [cosette](https://huggingface.co/kyutai/tts-voices/blob/main/expresso/ex04-ex02_confused_001_channel1_499s.wav)
* [eponine](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p262_023.wav)
* [azelma](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p303_023.wav)

The `--voice` argument can also take a plain wav file as input for voice cloning.
You can use your own or check out our [voice repository](https://huggingface.co/kyutai/tts-voices).
We recommend [cleaning the sample](https://podcast.adobe.com/en/enhance) before using it with Pocket TTS, because the audio quality of the sample is also reproduced.

Feel free to check out the [generate documentation](https://github.com/kyutai-labs/pocket-tts/tree/main/docs/generate.md) for more details and examples.
For trying multiple voices and prompts quickly, prefer using the `serve` command.

### The `serve` command

You can also run a local server to generate audio via HTTP requests.
```bash
uvx pocket-tts serve
# or if you installed it manually with pip:
pocket-tts serve
```
Navigate to `http://localhost:8000` to try the web interface, it's faster than the command line as the model is kept in memory between requests.

You can check out the [serve documentation](https://github.com/kyutai-labs/pocket-tts/tree/main/docs/serve.md) for more details and examples.

### The `serve-api` command (Scalable)

You can also run a scalable, queue-based API server optimized for concurrent requests.
```bash
uvx pocket-tts serve-api
# or if you installed it manually with pip:
pocket-tts serve-api
```
This server runs on port `8001` by default and exposes a `POST /generate` endpoint. It processes requests sequentially in a background worker to ensure thread safety while managing concurrency via a queue.

**API Usage Example**:

**Endpoint**: `POST http://localhost:8001/generate`

**Payload**:
```json
{
  "text_transcription": "Hello world, this is a test.",
  "reference_audio": "<base64_encoded_audio_string>"
}
```

**Response**:
```json
{
  "audio": "<base64_encoded_generated_wav>"
}
```

### The `export-voice` command

Processing an audio file (e.g., a .wav or .mp3) for voice cloning is relatively slow, but loading a safetensors file -- a voice embedding converted from an audio file -- is very fast. You can use the `export-voice` command to do this conversion. See the [export-voice documentation](https://github.com/kyutai-labs/pocket-tts/tree/main/docs/export_voice.md) for more details and examples.


## Using it as a Python library

You can try out the Python library on Colab [here](https://colab.research.google.com/github/kyutai-labs/pocket-tts/blob/main/docs/pocket-tts-example.ipynb).

Install the package with
```bash
pip install pocket-tts
# or
uv add pocket-tts
```

You can use this package as a simple Python library to generate audio from text.
```python
from pocket_tts import TTSModel
import scipy.io.wavfile

tts_model = TTSModel.load_model()
voice_state = tts_model.get_state_for_audio_prompt(
    "alba"  # One of the pre-made voices, see above
    # You can also use any voice file you have locally or from Hugging Face:
    # "./some_audio.wav"
    # or "hf://kyutai/tts-voices/expresso/ex01-ex02_default_001_channel2_198s.wav"
)
audio = tts_model.generate_audio(voice_state, "Hello world, this is a test.")
# Audio is a 1D torch tensor containing PCM data.
scipy.io.wavfile.write("output.wav", tts_model.sample_rate, audio.numpy())
```

You can have multiple voice states around if 
you have multiple voices you want to use. `load_model()` 
and `get_state_for_audio_prompt()` are relatively slow operations,
so we recommend to keep the model and voice states in memory if you can.

You can check out the [Python API documentation](https://github.com/kyutai-labs/pocket-tts/tree/main/docs/python-api.md) for more details and examples.

## Unsupported features

At the moment, we do not support (but would love pull requests adding):
- [Running the TTS inside a web browser (WebAssembly)](https://github.com/kyutai-labs/pocket-tts/issues/1)
- [A compiled version with for example `torch.compile()` or `candle`.](https://github.com/kyutai-labs/pocket-tts/issues/2)
- [Adding silence in the text input to generate pauses.](https://github.com/kyutai-labs/pocket-tts/issues/6)
- [Quantization to run the computation in int8.](https://github.com/kyutai-labs/pocket-tts/issues/7)

We tried running this TTS model on the GPU but did not observe a speedup compared to CPU execution,
notably because we use a batch size of 1 and a very small model.

## Development and local setup

We accept contributions! Feel free to open issues or pull requests on GitHub.

You can find development instructions in the [CONTRIBUTING.md](https://github.com/kyutai-labs/pocket-tts/tree/main/CONTRIBUTING.md) file. You'll also find there how to have an editable install of the package for local development.

## In-browser implementations

Pocket TTS is small enough to run directly in your browser in WebAssembly/JavaScript.
We don't have official support for this yet, but you can try out one of these community implementations:

- [babybirdprd/pocket-tts](https://github.com/babybirdprd/pocket-tts): Candle version (Rust) with WebAssembly and PyO3 bindings, meaning it can run on the web too.
- [ekzhang/jax-js](https://github.com/ekzhang/jax-js/tree/main/website/src/routes/tts): Using jax-js, a ML library for the web. Demo [here](https://jax-js.com/tts)
- [KevinAHM/pocket-tts-onnx-export](https://github.com/KevinAHM/pocket-tts-onnx-export): Model exported to .onnx and run using [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/). Demo [here](https://huggingface.co/spaces/KevinAHM/pocket-tts-web)

## Alterative implementations
- [jishnuvenugopal/pocket-tts-mlx](https://github.com/jishnuvenugopal/pocket-tts-mlx) - MLX backend optimized for Apple Silicon
- [babybirdprd/pocket-tts](https://github.com/babybirdprd/pocket-tts) - Candle version (Rust) with WebAssembly and PyO3 bindings.
## Running with Docker

You can run the servers using Docker.

1. Build the image:
```bash
git clone https://github.com/tripathiarpan20/pocket-tts
cd pocket-tts
docker build -t pocket-tts:latest -f Dockerfile .
```

2. Run the API Server (Scalable, Port 8001):
```bash
docker run -d \
  -p 8001:8001 \
  --name pocket_tts_api \
  --entrypoint "" \
  pocket-tts \
  uv run pocket-tts serve-api --host 0.0.0.0 --port 8001
```

3. Run the UI Server (Port 8000):
```bash
docker run -d \
  -p 8000:8000 \
  --name pocket_tts_ui \
  --entrypoint "" \
  pocket-tts \
  uv run pocket-tts serve --host 0.0.0.0 --port 8000
```

## Deploying to Fly.io

Fly.io is a great option for running this application cheaply and easily. We have included a `fly.toml` configuration file to get you started.

1.  **Install flyctl**: [Follow instructions here](https://fly.io/docs/hands-on/install-flyctl/)
2.  **Login**: `fly auth login`
3.  **Launch the app**:
    ```bash
    # Initialize the app (creates it on your account)
    # When asked if you want to copy the configuration to the new app, say yes.
    # When asked to deploy now, you can say yes.
    fly launch --no-deploy --copy-config --name pocket-tts-api-YOURNAME
    ```
4.  **Deploy**:
    ```bash
    fly deploy
    ```

The `fly.toml` is configured to run the scalable `serve-api` command on a `shared-cpu-2x` machine with 4GB RAM, which is a good balance of cost and performance.

### Configuration

The `fly.toml` file sets the internal port to `8080` (which matches standard cloud defaults) and overrides the default command to:
`uv run pocket-tts serve-api --host 0.0.0.0 --port 8080`

If you are using the default `fly.toml`, your API will be available at:
`https://pocket-tts-api-YOURNAME.fly.dev/generate`

## Projects using Pocket TTS

- [lukasmwerner/pocket-reader](https://github.com/lukasmwerner/pocket-reader) - Browser screen reader
- [ikidd/pocket-tts-wyoming](https://github.com/ikidd/pocket-tts-wyoming) - Docker container for pocket-tts using Wyoming protocol, ready for Home Assistant Voice use.
- [slaughters85j/pocket-tts](https://github.com/slaughters85j/pocket-tts) - Mac Desktop App + macOS Quick Action
- [teddybear082/pocket-tts-openai_streaming_server](https://github.com/teddybear082/pocket-tts-openai_streaming_server) - OpenAI-compatible streaming server, dockerized and with an `.exe` release

## Prohibited use

Use of our model must comply with all applicable laws and regulations and must not result in, involve, or facilitate any illegal, harmful, deceptive, fraudulent, or unauthorized activity. Prohibited uses include, without limitation, voice impersonation or cloning without explicit and lawful consent; misinformation, disinformation, or deception (including fake news, fraudulent calls, or presenting generated content as genuine recordings of real people or events); and the generation of unlawful, harmful, libelous, abusive, harassing, discriminatory, hateful, or privacy-invasive content. We disclaim all liability for any non-compliant use.


## Authors

Manu Orsini*, Simon Rouard*, Gabriel De Marmiesse*, Václav Volhejn, Neil Zeghidour, Alexandre Défossez

*equal contribution
