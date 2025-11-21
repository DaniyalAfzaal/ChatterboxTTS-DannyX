"""
Chatterbox TTS - Native Modal Job Queue
Fixed version with voices mount
"""
import modal
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import time
import uuid
import sys
import os

# --- Modal App Setup ---
app = modal.App("chatterbox-async-v1")
vol = modal.Volume.from_name("chatterbox-data", create_if_missing=True)

# Storage paths inside volume
OUTPUT_DIR = "/data/outputs"
VOICES_DIR = "/data/voices"
REF_AUDIO_DIR = "/data/reference_audio"
MODEL_DIR = "/data/model_cache"

# --- Docker Images ---
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "fastapi[standard]",
        "requests",
        "soundfile",
        "httpx",
        "pyyaml",
        "pydub",
        "numpy",
    )
)

worker_image = (
    base_image
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        "numpy<3.0.0",
        "soundfile",
        "librosa",
        "safetensors",
        "descript-audio-codec",
        "PyYAML",
        "python-multipart",
        "requests",
        "Jinja2",
        "watchdog",
        "aiofiles",
        "unidecode",
        "inflect",
        "tqdm",
        "hf_transfer",
        "pydub",
        "audiotsm",
        "praat-parselmouth",
        extra_index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install("git+https://github.com/devnen/chatterbox.git")
    .run_commands("rm -rf /app && git clone --branch main https://github.com/DaniyalAfzaal/ChatterboxDANNYX /app")
    .add_local_dir("voices", remote_path="/app/voices")
)

web_image = (
    base_image
    .run_commands("rm -rf /app && git clone --branch main https://github.com/DaniyalAfzaal/ChatterboxDANNYX /app")
    .run_commands("rm -rf /app && git clone --branch main https://github.com/DaniyalAfzaal/ChatterboxDANNYX /app")
)

# --- TTS Worker ---
@app.function(
    image=worker_image,
    gpu="any",
    timeout=3600,
    volumes={"/data": vol}
)
def tts_worker(job_id: str, config: dict):
    """Background worker that generates TTS audio."""
    print(f"👷 [Worker] Starting job {job_id}")
    
    sys.path.append("/app")
    import logging
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
    logger = logging.getLogger("tts_worker")
    
    
    import engine
    import torch
    import numpy as np
    import soundfile as sf
    from config import config_manager
    
    job_dir = Path(OUTPUT_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Capture logs to file
    debug_log_path = job_dir / "debug.log"
    file_handler = logging.FileHandler(debug_log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logging.getLogger("engine").addHandler(file_handler)
    
    status_file = job_dir / "status.json"
    output_file = job_dir / "output.wav"
    
    def update_status(status: str, progress: float, message: str, error: str = None):
        data = {
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "message": message,
            "error": error,
            "updated_at": time.time()
        }
        with open(status_file, "w") as f:
            json.dump(data, f)
        vol.commit()

    try:
        update_status("processing", 0.0, "Initializing engine...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"👷 [Worker] Using device: {device}")
        
        repo_id = config.get("repo_id", "ResembleAI/chatterbox")
        print(f"👷 [Worker] Configuring model {repo_id}...")
        
        if "model" not in config_manager.config: config_manager.config["model"] = {}
        if "tts_engine" not in config_manager.config: config_manager.config["tts_engine"] = {}
        
        config_manager.config["model"]["repo_id"] = repo_id
        config_manager.config["tts_engine"]["device"] = device
        
        print(f"👷 [Worker] Config: {config_manager.config}")
        print(f"👷 [Worker] Loading model...")
        success = engine.load_model()
        print(f"👷 [Worker] load_model returned: {success}")
        
        if not success:
            raise Exception("Failed to load model")
        
        update_status("processing", 10.0, "Generating audio...")
        
        # Determine voice path
        voice_path = None
        voice_mode = config.get("voice_mode", "predefined")
        text = config.get("text", "")
        
        if voice_mode == "predefined":
            voice_id = config.get("predefined_voice_id", "Olivia.wav")
            # Check mounted voices first
            p = Path("/app/voices") / voice_id
            if p.exists():
                voice_path = p
            else:
                # Check volume
                p = Path(VOICES_DIR) / voice_id
                if p.exists():
                    voice_path = p
                    
        elif voice_mode == "clone":
            ref_file = config.get("reference_file")
            if ref_file:
                p = job_dir / ref_file
                if p.exists():
                    voice_path = p
                else:
                    p = Path(REF_AUDIO_DIR) / ref_file
                    if p.exists():
                        voice_path = p
        
        if not voice_path or not voice_path.exists():
            print(f"⚠️ [Worker] Voice path {voice_path} not found. Using default.")
            voice_path = Path("/app/voices/Olivia.wav")
            if not voice_path.exists():
                raise Exception(f"Default voice file not found at {voice_path}")
            
        print(f"👷 [Worker] Using voice: {voice_path}")

        gen_params = config.get("generation_params", {})
        temperature = float(gen_params.get("temperature", 0.7))
        
        print(f"👷 [Worker] Synthesizing text: {text[:50]}...")
        
        audio_tensor, sr = engine.synthesize(
            text, 
            audio_prompt_path=str(voice_path), 
            temperature=temperature
        )
        
        if audio_tensor is None:
            raise Exception("Synthesis returned None")

        # Convert to numpy and ensure proper shape for soundfile
        if isinstance(audio_tensor, torch.Tensor):
            audio_np = audio_tensor.detach().cpu().numpy()
        else:
            audio_np = audio_tensor
        
        # Squeeze to remove singleton dimensions (e.g., [1, N] -> [N])
        audio_np = audio_np.squeeze()
        
        print(f"👷 [Worker] Audio shape: {audio_np.shape}, dtype: {audio_np.dtype}, sample_rate: {sr}")
        
        # Ensure it's float32 or float64 for soundfile (using dtype.name to avoid np reference)
        if audio_np.dtype.name not in ['float32', 'float64']:
            audio_np = audio_np.astype('float32')
        
        sf.write(str(output_file), audio_np, sr)
        
        print(f"👷 [Worker] Job completed. Saved to {output_file}")
        update_status("completed", 100.0, "Audio generation complete.")
        
    except Exception as e:
        print(f"❌ [Worker] Error: {e}")
        import traceback
        traceback.print_exc()
        update_status("failed", 0.0, "Generation failed", str(e))

# --- Web Server ---
web_app = FastAPI(title="Chatterbox Native Queue")

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ui_path = "/app/ui" if os.path.exists("/app/ui") else "ui"
if os.path.exists(ui_path):
    web_app.mount("/ui", StaticFiles(directory=ui_path), name="ui")

@web_app.get("/")
async def index():
    p = Path(ui_path) / "index.html"
    if p.exists():
        return FileResponse(p)
    return HTMLResponse("<h1>Chatterbox TTS</h1>")

@web_app.get("/script.js")
async def script():
    p = Path(ui_path) / "script.js"
    if p.exists():
        return FileResponse(p, media_type="application/javascript")
    return JSONResponse({"error": "Not found"}, status_code=404)

@web_app.get("/styles.css")
async def styles():
    p = Path(ui_path) / "styles.css"
    if p.exists():
        return FileResponse(p, media_type="text/css")
    return JSONResponse({"error": "Not found"}, status_code=404)

@web_app.post("/jobs/submit")
async def submit_job(request: Request):
    """Submit a job to the worker."""
    try:
        data = await request.json()
        text = data.get("text")
        if not text:
            return JSONResponse({"error": "Text is required"}, status_code=400)
            
        job_id = str(uuid.uuid4())
        job_dir = Path(OUTPUT_DIR) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        with open(job_dir / "status.json", "w") as f:
            json.dump({
                "job_id": job_id,
                "status": "queued",
                "progress": 0.0,
                "message": "Job queued",
                "created_at": time.time()
            }, f)
            
        # Pass text in config
        config = data.copy()
        config["text"] = text
        
        tts_worker.spawn(job_id, config)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Job submitted successfully."
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@web_app.get("/jobs/{job_id}/status")
async def get_status(job_id: str):
    """Check job status."""
    status_file = Path(OUTPUT_DIR) / job_id / "status.json"
    vol.reload()
    
    if not status_file.exists():
        if (Path(OUTPUT_DIR) / job_id).exists():
            return {"status": "queued", "message": "Waiting for worker..."}
        return JSONResponse({"error": "Job not found"}, status_code=404)
        
    try:
        with open(status_file, "r") as f:
            return json.load(f)
    except Exception as e:
        return JSONResponse({"error": f"Failed to read status: {e}"}, status_code=500)

@web_app.get("/jobs/{job_id}/result")
async def get_result(job_id: str):
    """Download result file."""
    output_file = Path(OUTPUT_DIR) / job_id / "output.wav"
    vol.reload()
    if not output_file.exists():
        return JSONResponse({"error": "Result not ready"}, status_code=404)
    return FileResponse(output_file, media_type="audio/wav", filename=f"chatterbox_{job_id}.wav")

@web_app.get("/jobs/{job_id}/log")
async def get_log(job_id: str):
    """Download debug log."""
    log_file = Path(OUTPUT_DIR) / job_id / "debug.log"
    vol.reload()
    if not log_file.exists():
        return JSONResponse({"error": "Log not found"}, status_code=404)
    return FileResponse(log_file, media_type="text/plain")

@web_app.get("/get_predefined_voices")
async def get_predefined_voices():
    """List available voices."""
    voices = []
    if Path(VOICES_DIR).exists():
        for f in Path(VOICES_DIR).iterdir():
            if f.suffix in ['.wav', '.mp3']:
                voices.append({"id": f.name, "name": f.stem})
    
    app_voices = Path("/app/voices")
    if app_voices.exists():
        for f in app_voices.iterdir():
            if f.suffix in ['.wav', '.mp3']:
                if not any(v['id'] == f.name for v in voices):
                    voices.append({"id": f.name, "name": f.stem})
    return voices

# --- Entrypoint ---
@app.function(image=web_image, volumes={"/data": vol})
@modal.asgi_app()
def entrypoint():
    return web_app
