# File: server.py
# Main FastAPI application for the TTS Server.
# Handles API requests for text-to-speech generation, UI serving,
# configuration management, and file uploads.

import os
import io
import logging
import logging.handlers  # For RotatingFileHandler
import shutil
import time
import uuid
import sys  # For sys.exit() on fatal CUDA errors
import threading
import requests
import yaml  # For loading presets
import numpy as np
import torch  # For GPU memory management
import librosa  # For potential direct use if needed, though utils.py handles most
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    BackgroundTasks,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
    FileResponse,
    PlainTextResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import engine
import validation  # Input validation functions
from config import (
    config_manager,
    get_log_file_path,
    get_output_path,
    get_reference_audio_path,
    get_predefined_voices_path,
    get_audio_output_format,
    get_audio_sample_rate,
    get_host,
    get_port,
    get_ui_title,
    get_model_repo_id,
    get_model_cache_path,
    get_device_setting,
    get_gen_default_speed_factor,
    get_default_split_text_setting,
    get_default_chunk_size,
    get_gen_default_temperature,
    get_gen_default_exaggeration,
    get_gen_default_cfg_weight,
    get_gen_default_seed,
)
from models import (
    CustomTTSRequest,
    UIInitialDataResponse,
    FileListResponse,
    ErrorResponse,
    UpdateStatusResponse,
    JobSubmissionResponse,
    JobStatusResponse,
    JobResultResponse,
)
import utils
import gpu_health
import jobs  # Job queue management
from jobs import JobStatus

# Import asyncio for request queuing
import asyncio

# --------------------------------------------------------------------------------------
# Global: Request Queuing Semaphore
# --------------------------------------------------------------------------------------
# Limit to 1 concurrent TTS synthesis to prevent GPU OOM
# Note: Initialized in lifespan to avoid "no running event loop" error
tts_semaphore: Optional[asyncio.Semaphore] = None
logger_init = logging.getLogger("tts_server_init")


class OpenAISpeechRequest(BaseModel):
    model: str
    input_: str = Field(..., alias="input")
    voice: str
    response_format: Literal["wav", "opus", "mp3"] = "wav"
    speed: float = 1.0
    seed: Optional[int] = None


# --------------------------------------------------------------------------------------
# Logging Configuration
# --------------------------------------------------------------------------------------

log_file_path = get_log_file_path()
log_dir = os.path.dirname(log_file_path)
os.makedirs(log_dir, exist_ok=True)

file_handler = logging.handlers.RotatingFileHandler(
    log_file_path,
    maxBytes=config_manager.get_int("logging.max_bytes", 5 * 1024 * 1024),
    backupCount=config_manager.get_int("logging.backup_count", 3),
    encoding="utf-8",
)

formatter = logging.Formatter(
    fmt=config_manager.get_string(
        "logging.format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    ),
    datefmt=config_manager.get_string("logging.datefmt", "%Y-%m-%d %H:%M:%S"),
)

file_handler.setFormatter(formatter)

logging.basicConfig(
    level=config_manager.get_string("logging.level", "INFO"),
    handlers=[file_handler, logging.StreamHandler()],
)

logger = logging.getLogger("tts_server")

# --------------------------------------------------------------------------------------
# FastAPI Initialization with Lifespan
# --------------------------------------------------------------------------------------

templates = Jinja2Templates(directory="ui")

from threading import Event

startup_complete_event = Event()
model_loaded_event = Event()


def _delayed_browser_open(host: str, port: int):
    try:
        import webbrowser

        for _ in range(10):
            if engine.MODEL_LOADED:
                break
            time.sleep(1.5)
        display_host = "localhost" if host == "0.0.0.0" else host
        browser_url = f"http://{display_host}:{port}/"
        logger.info(f"Attempting to open web browser to: {browser_url}")
        webbrowser.open(browser_url)
    except Exception as e:
        logger.error(f"Failed to open browser automatically: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_semaphore
    logger.info("TTS Server: Initializing...")
    
    # Initialize request semaphore (needs running event loop)
    tts_semaphore = asyncio.Semaphore(1)
    logger.info("TTS request semaphore initialized (max concurrent: 1)")

    output_path = get_output_path(ensure_absolute=True)
    reference_audio_path = get_reference_audio_path(ensure_absolute=True)
    predefined_voices_path = get_predefined_voices_path(ensure_absolute=True)

    for path in [output_path, reference_audio_path, predefined_voices_path]:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {path}")

    ui_dir = Path("ui")
    if not ui_dir.exists():
        logger.warning("UI directory not found. Creating empty 'ui' directory.")
        ui_dir.mkdir(parents=True, exist_ok=True)

    model_cache_path = get_model_cache_path(ensure_absolute=True)
    model_cache_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Model cache directory: {model_cache_path}")

    repo_id = get_model_repo_id()
    cache_dir = get_model_cache_path(ensure_absolute=True)
    device = get_device_setting()

    logger.info(f"Model repository: {repo_id}")
    logger.info(f"Model cache directory: {cache_dir}")
    logger.info(f"Device for TTS: {device}")

    logger.info("Loading TTS model...")
    try:
        engine.load_model(repo_id, str(cache_dir), device)
        logger.info("TTS model loaded successfully.")
        model_loaded_event.set()
    except Exception as e:
        logger.exception(f"Exception during model loading: {e}")
        # Optionally, re-raise or exit if model loading is critical
        sys.exit(1) # Exit if model loading fails

    try:
        server_host = get_host()
        server_port = get_port()
        import threading as _threading

        browser_thread = _threading.Thread(
            target=lambda: _delayed_browser_open(server_host, server_port),
            daemon=True,
        )
        browser_thread.start()
    except Exception as e:
        logger.error(f"Error starting browser thread: {e}")

    startup_complete_event.set()
    logger.info("Startup sequence completed.")
    
    # Cleanup old files on startup
    try:
        output_path = get_output_path(ensure_absolute=True)
        cutoff = datetime.now() - timedelta(days=7)
        count = 0
        
        for file in output_path.iterdir():
            if file.is_file() and datetime.fromtimestamp(file.stat().st_mtime) < cutoff:
                try:
                    file.unlink()
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete old file {file}: {e}")
        
        if count > 0:
            logger.info(f"Cleaned up {count} old output files from {output_path}")
    except Exception as e:
        logger.error(f"Failed to cleanup old files: {e}")
    
    # Start periodic job cleanup task
    async def periodic_job_cleanup():
        """Background task to clean up old jobs every hour"""
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                job_manager = jobs.get_job_manager()
                deleted_count = job_manager.cleanup_old_jobs(max_age_hours=24)
                if deleted_count > 0:
                    logger.info(f"Periodic cleanup: removed {deleted_count} old jobs")
            except Exception as e:
                logger.error(f"Periodic job cleanup failed: {e}")
    
    cleanup_task = asyncio.create_task(periodic_job_cleanup())
    logger.info("Started periodic job cleanup task (runs every hour)")
    
    yield
    
    # Shutdown
    logger.info("Application shutdown sequence initiated.")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("Application shutdown sequence completed.")


app = FastAPI(
    title=get_ui_title(),
    description="Text-to-Speech server with advanced UI and API capabilities.",
    version="2.0.2",
    lifespan=lifespan,
)

# --------------------------------------------------------------------------------------
# CORS and Static Files
# --------------------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.mount("/ui", StaticFiles(directory="ui"), name="ui")

outputs_static_path = get_output_path(ensure_absolute=True)
if outputs_static_path.exists():
    app.mount("/outputs", StaticFiles(directory=str(outputs_static_path)), name="outputs")
else:
    logger.warning(
        f"Outputs directory {outputs_static_path} does not exist; skipping static mount."
    )

vendor_dir = Path("ui") / "vendor"
if vendor_dir.exists():
    app.mount("/vendor", StaticFiles(directory=str(vendor_dir)), name="vendor")
else:
    logger.info("No vendor directory found under ui/. Skipping vendor static mount.")

# --------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------


def load_presets() -> Dict[str, Any]:
    presets_path = Path("ui") / "presets.yaml"
    if not presets_path.is_file():
        logger.info("No presets.yaml found in ui/. Using empty presets.")
        return {}
    try:
        with open(presets_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to load presets.yaml: {e}")
        return {}


def list_reference_files() -> List[str]:
    ref_dir = get_reference_audio_path(ensure_absolute=True)
    if not ref_dir.exists():
        return []
    return utils.get_valid_reference_files(ref_dir)


def list_predefined_voices() -> List[Dict[str, str]]:
    voices_dir = get_predefined_voices_path(ensure_absolute=True)
    if not voices_dir.exists():
        return []
    return utils.get_predefined_voices()



def get_ui_state_from_config() -> Dict[str, Any]:
    ui_state = config_manager.get_dict("ui_state", default={})
    if "last_selected_mode" not in ui_state:
        ui_state["last_selected_mode"] = "clone"
    if "last_selected_predefined_voice" not in ui_state:
        ui_state["last_selected_predefined_voice"] = ""
    if "last_selected_reference_file" not in ui_state:
        ui_state["last_selected_reference_file"] = ""
    if "split_text" not in ui_state:
        ui_state["split_text"] = get_default_split_text_setting()
    if "chunk_size" not in ui_state:
        ui_state["chunk_size"] = get_default_chunk_size()
    return ui_state


# --------------------------------------------------------------------------------------
# Routes: UI and Static
# --------------------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    index_path = Path("ui") / "index.html"
    if not index_path.is_file():
        return HTMLResponse(
            content="<h1>UI not found</h1><p>index.html missing in ui/ directory.</p>",
            status_code=404,
        )
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/styles.css", response_class=PlainTextResponse)
async def styles():
    css_path = Path("ui") / "styles.css"
    if not css_path.is_file():
        return PlainTextResponse(content="", status_code=404)
    with open(css_path, "r", encoding="utf-8") as f:
        css_content = f.read()
    return PlainTextResponse(content=css_content, status_code=200)


@app.get("/script.js", response_class=PlainTextResponse)
async def script():
    js_path = Path("ui") / "script.js"
    if not js_path.is_file():
        return PlainTextResponse(content="// script.js not found", status_code=404)
    with open(js_path, "r", encoding="utf-8") as f:
        js_content = f.read()
    return PlainTextResponse(content=js_content, status_code=200)


# --------------------------------------------------------------------------------------
# Routes: API - Initial UI Data
# --------------------------------------------------------------------------------------


@app.get("/api/ui/initial-data", response_model=UIInitialDataResponse)
async def get_ui_initial_data():
    try:
        reference_files = list_reference_files()
        predefined_voice_files = list_predefined_voices()
        ui_state = get_ui_state_from_config()
        presets = load_presets()

        return UIInitialDataResponse(
            reference_files=reference_files,
            predefined_voices=predefined_voice_files,
            ui_state=ui_state,
            presets=presets,
        )
    except Exception as e:
        logger.exception(f"Error building UI initial data: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to load UI initial data.",
        )


# --------------------------------------------------------------------------------------
# Routes: Configuration Management
# --------------------------------------------------------------------------------------


@app.post("/save_settings", response_model=UpdateStatusResponse)
async def save_settings(new_settings: Dict[str, Any]):
    try:
        logger.info(f"Received settings update: {new_settings}")
        config_manager.update(new_settings)
        config_manager.save()
        restart_flag = True
        return UpdateStatusResponse(
            success=True,
            message="Settings saved successfully. A server restart may be required.",
            restart_required=restart_flag,
        )
    except Exception as e:
        logger.exception(f"Failed to save settings: {e}")
        return UpdateStatusResponse(
            success=False,
            message=f"Failed to save settings: {e}",
            restart_required=False,
        )


@app.post("/reset_settings", response_model=UpdateStatusResponse)
async def reset_settings():
    try:
        config_manager.reset_and_save()
        return UpdateStatusResponse(
            success=True,
            message="Settings reset to default successfully.",
            restart_required=True,
        )
    except Exception as e:
        logger.exception(f"Failed to reset settings: {e}")
        return UpdateStatusResponse(
            success=False,
            message=f"Failed to reset settings: {e}",
            restart_required=False,
        )


@app.post("/restart_server", response_model=UpdateStatusResponse)
async def restart_server():
    logger.warning(
        "Received request to restart server. This is a placeholder in this implementation."
    )
    return UpdateStatusResponse(
        success=True,
        message="Server restart requested. Please restart the process externally if needed.",
        restart_required=True,
    )


# --------------------------------------------------------------------------------------
# Routes: File Management
# --------------------------------------------------------------------------------------


@app.get("/get_reference_files", response_model=FileListResponse)
async def get_reference_files():
    try:
        files = list_reference_files()
        return FileListResponse(files=files)
    except Exception as e:
        logger.exception(f"Failed to list reference files: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to list reference files.",
        )


@app.get("/get_predefined_voices", response_model=FileListResponse)
async def get_predefined_voices():
    try:
        files = list_predefined_voices()
        return FileListResponse(files=files)
    except Exception as e:
        logger.exception(f"Failed to list predefined voices: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to list predefined voices.",
        )


@app.post("/upload_reference")
async def upload_reference(files: List[UploadFile] = File(...)):
    ref_dir = get_reference_audio_path(ensure_absolute=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    errors = []

    for file in files:
        try:
            filename = utils.sanitize_filename(file.filename)
            dest_path = ref_dir / filename
            with open(dest_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(filename)
            logger.info(f"Uploaded reference audio: {filename}")
        except Exception as e:
            logger.exception(f"Failed to save uploaded reference file {file.filename}: {e}")
            errors.append(f"Failed to save {file.filename}: {e}")
        finally:
            await file.close()

    return JSONResponse(
        content={
            "uploaded_files": saved_files,
            "errors": errors,
        },
        status_code=200 if saved_files else 400,
    )


@app.post("/upload_predefined_voice")
async def upload_predefined_voice(files: List[UploadFile] = File(...)):
    voices_dir = get_predefined_voices_path(ensure_absolute=True)
    voices_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    errors = []

    for file in files:
        try:
            filename = utils.sanitize_filename(file.filename)
            dest_path = voices_dir / filename
            with open(dest_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(filename)
            logger.info(f"Uploaded predefined voice: {filename}")
        except Exception as e:
            logger.exception(f"Failed to save uploaded predefined voice {file.filename}: {e}")
            errors.append(f"Failed to save {file.filename}: {e}")
        finally:
            await file.close()

    return JSONResponse(
        content={
            "uploaded_files": saved_files,
            "errors": errors,
        },
        status_code=200 if saved_files else 400,
    )


# --------------------------------------------------------------------------------------
# Background Job Processing
# --------------------------------------------------------------------------------------

async def process_tts_job_background(job_id: str, request: CustomTTSRequest):
    """
    Background task that processes a TTS job.
    This is the async equivalent of the /tts endpoint.
    Updates job status throughout the process.
    """
    job_manager = None
    try:
        # Get managers early to ensure we can update job status on any failure
        job_manager = jobs.get_job_manager()
        gpu_monitor = gpu_health.get_gpu_monitor()
        # Mark job as processing
        job_manager.update_job_status(job_id, JobStatus.PROCESSING, current_step="Starting synthesis")
        logger.info(f"Job {job_id}: Starting background TTS processing")
        
        # This function contains the full TTS synthesis logic from custom_tts_endpoint
        # but adapted for background execution with job status updates
        
        # Use semaphore to prevent concurrent GPU access
        async with tts_semaphore:
            job_manager.update_progress(job_id, 0, 1, "Validating inputs")
            
            # Input validation (already done in submit, but re-validate)
            validated_text = validation.validate_text_input(request.text)
            chunk_size = validation.validate_chunk_size(request.chunk_size)
            
            # Prepare audio prompt path
            audio_prompt_path_for_engine: Optional[Path] = None
            
            # Voice mode selection
            if request.voice_mode == "predefined" and request.predefined_voice_id:
                voices_dir = get_predefined_voices_path(ensure_absolute=True)
                potential_path = voices_dir / request.predefined_voice_id
                if potential_path.is_file():
                    audio_metadata = validation.validate_reference_audio(potential_path)
                    audio_prompt_path_for_engine = potential_path
            elif request.voice_mode == "clone" and request.reference_audio_filename:
                ref_dir = get_reference_audio_path(ensure_absolute=True)
                potential_path = ref_dir / request.reference_audio_filename
                if potential_path.is_file():
                    audio_metadata = validation.validate_reference_audio(potential_path)
                    audio_prompt_path_for_engine = potential_path
            
            # Text splitting
            job_manager.update_progress(job_id, 0, 1, "Splitting text into chunks")
            text_chunks = utils.chunk_text_by_sentences(validated_text, chunk_size)
            
            job_manager.update_job_status(
                job_id,
                JobStatus.PROCESSING,
                chunk_count=len(text_chunks),
                current_step=f"Processing {len(text_chunks)} chunks"
            )
            
            # Synthesize chunks
            all_audio_segments_np: List[np.ndarray] = []
            final_output_sample_rate = get_audio_sample_rate()
            engine_output_sample_rate: Optional[int] = None
            
            for i, chunk in enumerate(text_chunks):
                job_manager.update_progress(
                    job_id,
                    i + 1,  # FIX: Use 1-indexed progress (1/168 not 0/168)
                    len(text_chunks),
                    f"Synthesizing chunk {i+1}/{len(text_chunks)}"
                )
                
                try:
                    chunk_audio_tensor, chunk_sr_from_engine = engine.synthesize(
                        text=chunk.strip(),
                        audio_prompt_path=str(audio_prompt_path_for_engine) if audio_prompt_path_for_engine else None,
                        temperature=validation.validate_temperature(request.temperature),
                        exaggeration=validation.validate_exaggeration(request.exaggeration),
                        cfg_weight=validation.validate_cfg_weight(request.cfg_weight),
                        seed=validation.validate_seed(request.seed),
                    )
                    
                    if chunk_audio_tensor is None or chunk_sr_from_engine is None:
                        raise RuntimeError("Engine returned no audio")
                    
                    if engine_output_sample_rate is None:
                        engine_output_sample_rate = chunk_sr_from_engine
                    
                    # Apply speed factor if needed
                    current_processed_audio_tensor = chunk_audio_tensor
                    speed_factor = validation.validate_speed_factor(request.speed_factor)
                    if speed_factor != 1.0:
                        current_processed_audio_tensor, _ = utils.apply_speed_factor(
                            current_processed_audio_tensor,
                            chunk_sr_from_engine,
                            speed_factor,
                        )
                    
                    processed_audio_np = current_processed_audio_tensor.cpu().numpy().squeeze()
                    all_audio_segments_np.append(processed_audio_np)
                    
                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Record success
                    gpu_monitor.record_success()
                    
                except Exception as e:
                    logger.error(f"Job {job_id}: Chunk {i+1} failed: {e}")
                    job_manager.add_failed_chunk(job_id, {
                        "chunk_number": i + 1,
                        "chunk_text": chunk[:100],
                        "error": str(e)
                    })
                    
                    if request.allow_partial_success:
                        # Add silence placeholder
                        silence = np.zeros(int(1.0 * (engine_output_sample_rate or final_output_sample_rate)))
                        all_audio_segments_np.append(silence)
                    else:
                        raise
            
            # Combine all audio segments
            job_manager.update_progress(
                job_id,
                len(text_chunks),
                len(text_chunks),
                "Combining audio segments"
            )
            
            if not all_audio_segments_np:
                raise RuntimeError("No audio segments generated")
            
            combined_audio_np = np.concatenate(all_audio_segments_np)
            
            # Save result file
            output_dir = get_output_path()
            result_filename = f"job_{job_id}.wav"
            result_path = output_dir / result_filename
            
            job_manager.update_progress(
                job_id,
                len(text_chunks),
                len(text_chunks),
                "Saving audio file"
            )
            
            utils.save_audio(
                combined_audio_np,
                engine_output_sample_rate or final_output_sample_rate,
                str(result_path),
                "wav"  # Always save as WAV for now
            )
            
            # Mark job as completed
            job_manager.update_job_status(
                job_id,
                JobStatus.COMPLETED,
                result_file=str(result_path),
                current_step="Completed",
                progress_percent=100.0
            )
            
            logger.info(f"Job {job_id}: Completed successfully. Result: {result_path}")
    
    except Exception as e:
        logger.critical(f"Job {job_id} FATAL ERROR: {e}", exc_info=True)
        if job_manager:
            try:
                job_manager.update_job_status(
                    job_id,
                    JobStatus.FAILED,
                    error_message=str(e),
                    current_step="Failed"
                )
            except Exception as update_error:
                logger.critical(f"Could not update job {job_id} status: {update_error}")
        else:
            logger.critical(f"Job {job_id} failed before job_manager was initialized")


# --------------------------------------------------------------------------------------
# Routes: Async Job Queue (for long-running TTS beyond 10-minute timeout)
# --------------------------------------------------------------------------------------

# Debug logging setup
DEBUG_LOG_PATH = Path("/data/debug.log")

def log_debug(msg: str):
    """Write debug message to persistent log file"""
    try:
        # Ensure directory exists (local testing support)
        if not DEBUG_LOG_PATH.parent.exists():
            DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(f"{datetime.now().isoformat()} - {msg}\n")
    except Exception as e:
        print(f"Failed to write debug log: {e}")

@app.get("/debug")
async def get_debug_logs():
    """Retrieve last 100 lines of debug log"""
    if DEBUG_LOG_PATH.exists():
        try:
            content = DEBUG_LOG_PATH.read_text()
            return {"logs": content.splitlines()[-100:]}
        except Exception as e:
            return {"error": str(e)}
    return {"logs": ["Log file not found"]}

@app.post("/jobs/submit", response_model=JobSubmissionResponse)
async def submit_tts_job(request: Request, tts_request: CustomTTSRequest):
    """
    Submit a TTS job for asynchronous processing.
    Returns a Job ID immediately.
    """
    request_id = str(uuid.uuid4())
    log_debug(f"[{request_id}] Received submission request")
    
    try:
        # 1. Validate Input
        log_debug(f"[{request_id}] Validating input text length: {len(tts_request.text)}")
        validation.validate_text_input(tts_request.text)
        
        log_debug(f"[{request_id}] Validating configuration")
        # Validate configuration
        tts_request.cfg_weight = validation.validate_cfg_weight(tts_request.cfg_weight)
        
        # 2. Create Job
        log_debug(f"[{request_id}] Getting job manager")
        job_manager = jobs.get_job_manager()
        
        log_debug(f"[{request_id}] Creating job")
        job_id = job_manager.create_job(text_length=len(tts_request.text))
        log_debug(f"[{request_id}] Job created with ID: {job_id}")

        # 3. Start Background Task
        log_debug(f"[{request_id}] Starting background task")
        asyncio.create_task(process_tts_job_background(job_id, tts_request))
        log_debug(f"[{request_id}] Background task started")

        return JobSubmissionResponse(
            job_id=job_id,
            status=JobStatus.QUEUED,
            message="Job submitted successfully. Use job_id to track progress.",
            estimated_chunks=1 # Initial estimate, will be updated
        )

    except validation.ValidationError as e:
        log_debug(f"[{request_id}] Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log_debug(f"[{request_id}] Unexpected error: {e}")
        logger.error(f"Job submission failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")



@app.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Check the status of a TTS job.
    """
    job_manager = jobs.get_job_manager()
    job_info = job_manager.get_job(job_id)
    
    if not job_info:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return JobStatusResponse(
        job_id=job_info.job_id,
        status=job_info.status.value,
        progress_percent=job_info.progress_percent,
        current_step=job_info.current_step,
        processed_chunks=job_info.processed_chunks if job_info.chunk_count > 0 else None,
        total_chunks=job_info.chunk_count if job_info.chunk_count > 0 else None,
        created_at=job_info.created_at.isoformat(),
        started_at=job_info.started_at.isoformat() if job_info.started_at else None,
        completed_at=job_info.completed_at.isoformat() if job_info.completed_at else None,
        result_available=(job_info.status == JobStatus.COMPLETED),
        error_message=job_info.error_message,
        failed_chunks_count=len(job_info.failed_chunks) if job_info.failed_chunks else None
    )


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """
    Get the result of a completed TTS job.
    Returns the generated audio file if successful.
    """
    job_manager = jobs.get_job_manager()
    job_info = job_manager.get_job(job_id)
    
    if not job_info:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job_info.status == JobStatus.QUEUED:
        raise HTTPException(
            status_code=202,
            detail="Job is still queued. Check /jobs/{job_id}/status for progress."
        )
    
    if job_info.status == JobStatus.PROCESSING:
        raise HTTPException(
            status_code=202,
            detail=f"Job is still processing ({job_info.progress_percent:.1f}% complete). Check /jobs/{job_id}/status for progress."
        )
    
    if job_info.status == JobStatus.FAILED:
        return JSONResponse(
            content=JobResultResponse(
                job_id=job_id,
                status="failed",
                error_message=job_info.error_message or "Job failed"
            ).dict(),
            status_code=500
        )
    
    if job_info.status == JobStatus.COMPLETED:
        if not job_info.result_file or not Path(job_info.result_file).exists():
            raise HTTPException(
                status_code=404,
                detail="Result file not found. It may have been cleaned up."
            )
        
        # Return the audio file
        result_path = Path(job_info.result_file)
        return FileResponse(
            path=str(result_path),
            media_type="audio/wav",  # TODO: Support other formats
            filename=f"tts_job_{job_id}.wav"
        )
    
    raise HTTPException(status_code=400, detail=f"Unexpected job status: {job_info.status}")


# --------------------------------------------------------------------------------------
# Routes: Health & Monitoring
# --------------------------------------------------------------------------------------

@app.get("/health/gpu")
async def gpu_health_endpoint():
    """
    Check GPU health status and circuit breaker state.
    Returns detailed GPU memory info and error counts.
    """
    monitor = gpu_health.get_gpu_monitor()
    health = monitor.check_health()
    
    return JSONResponse(
        content={
            "status": "healthy" if (health.is_available and not health.circuit_open) else "unhealthy",
            "gpu_available": health.is_available,
            "circuit_breaker_open": health.circuit_open,
            "error_count": health.error_count,
            "last_error": health.last_error,
            "memory": {
                "allocated_gb": health.memory_allocated_gb,
                "reserved_gb": health.memory_reserved_gb,
                "free_gb": health.memory_free_gb,
                "total_gb": health.memory_total_gb
            },
            "device_count": health.device_count,
            "current_device": health.current_device
        },
        status_code=200 if (health.is_available and not health.circuit_open) else 503
    )


# --------------------------------------------------------------------------------------
# Routes: Core TTS Generation
# --------------------------------------------------------------------------------------


@app.post("/tts")
async def custom_tts_endpoint(
    request: CustomTTSRequest, background_tasks: BackgroundTasks
):
    """
    Generates speech audio from text using specified parameters.
    Handles various voice modes (predefined, clone) and audio processing options.
    Returns audio as a stream (WAV or Opus).
    """
    
    # Use semaphore to limit concurrent TTS to 1 (prevent GPU OOM)
    async with tts_semaphore:
        logger.info(f"TTS request acquired semaphore (queued requests: {tts_semaphore._value})")
        
        perf_monitor = utils.PerformanceMonitor()
        perf_monitor.start()
        
        # Check GPU health before starting
        gpu_monitor = gpu_health.get_gpu_monitor()
        if not gpu_monitor.is_healthy():
            health = gpu_monitor.check_health()
            error_detail = (
                f"GPU unhealthy or circuit breaker open. "
                f"Error count: {health.error_count}. "
                f"Last error: {health.last_error or 'None'}. "
                f"Try again in a few minutes."
            )
            logger.error(error_detail)
            raise HTTPException(status_code=503, detail=error_detail)
        
        logger.info("GPU health check passed")

        # ============================================
        # STEP 1: INPUT VALIDATION
        # ============================================
        # Validate all inputs before any processing to fail fast
        
        # Check disk space before starting (require 1GB free)
        import shutil
        disk_usage = shutil.disk_usage("/")
        available_gb = disk_usage.free / 1e9
        if available_gb < 1.0:
            raise HTTPException(
                status_code=507,
                detail=f"Insufficient disk space: {available_gb:.2f}GB available (minimum 1GB required). Contact administrator."
            )
        
        # Validate text input
        try:
            validated_text = validation.validate_text_input(request.text)
        except validation.ValidationError as e:
            raise e
        
        # Validate all numeric parameters
        validated_chunk_size = validation.validate_chunk_size(request.chunk_size)
        validated_speed_factor = validation.validate_speed_factor(request.speed_factor)
        validated_temperature = validation.validate_temperature(request.temperature)
        validated_cfg_weight = validation.validate_cfg_weight(request.cfg_weight)
        validated_exaggeration = validation.validate_exaggeration(request.exaggeration)
        validated_seed = validation.validate_seed(request.seed)
    
    logger.info(f"Input validation passed. Text length: {len(validated_text)} chars, Disk free: {available_gb:.2f}GB")

    # ============================================
    # STEP 2: VOICE MODE SELECTION
    # ============================================
    audio_prompt_path_for_engine: Optional[Path] = None

    if request.voice_mode == "predefined":
        if not request.predefined_voice_id:
            raise HTTPException(
                status_code=400,
                detail="Predefined voice ID must be provided in 'predefined' mode.",
            )
        voices_dir = get_predefined_voices_path(ensure_absolute=True)
        potential_path = voices_dir / request.predefined_voice_id
        if not potential_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Predefined voice file '{request.predefined_voice_id}' not found.",
            )
        
        # Validate reference audio using new validation module
        try:
            audio_metadata = validation.validate_reference_audio(potential_path)
            logger.info(f"Predefined voice validated: {audio_metadata}")
        except validation.ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e.detail))
        
        audio_prompt_path_for_engine = potential_path
        logger.info(f"Using predefined voice: {request.predefined_voice_id}")

    elif request.voice_mode == "clone":
        if not request.reference_audio_filename:
            raise HTTPException(
                status_code=400,
                detail="Reference audio filename must be provided in 'clone' mode.",
            )
        ref_dir = get_reference_audio_path(ensure_absolute=True)
        potential_path = ref_dir / request.reference_audio_filename
        if not potential_path.is_file():
            logger.error(
                f"Reference audio file for cloning not found: {potential_path}"
            )
            raise HTTPException(
                status_code=404,
                detail=f"Reference audio file '{request.reference_audio_filename}' not found.",
            )
        
        # Validate reference audio using new validation module
        try:
            audio_metadata = validation.validate_reference_audio(potential_path)
            logger.info(f"Reference audio validated: {audio_metadata}")
        except validation.ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e.detail))
        
        audio_prompt_path_for_engine = potential_path
        logger.info(
            f"Using reference audio for cloning: {request.reference_audio_filename}"
        )

    # ============================================
    # STEP 3: TEXT SPLITTING
    # ============================================
    text_chunks: List[str] = []
    if request.split_text and len(request.text) > (
        request.chunk_size or get_default_chunk_size()
    ):
        chunk_size = request.chunk_size or get_default_chunk_size()
        logger.info(
            f"Splitting text into chunks of approx {chunk_size} chars, "
            f"using sentence-aware logic."
        )
        text_chunks = utils.chunk_text_by_sentences(request.text, chunk_size)
    else:
        text_chunks = [request.text]
        logger.info(
            "Processing text as a single chunk (splitting not enabled or text too short)."
        )

    if not text_chunks:
        raise HTTPException(
            status_code=400, detail="Text processing resulted in no usable chunks."
        )

    all_audio_segments_np: List[np.ndarray] = []
    final_output_sample_rate = get_audio_sample_rate()
    engine_output_sample_rate: Optional[int] = None
    
    # Track failed chunks for partial success mode
    failed_chunks_info: List[Dict[str, Any]] = []
    MAX_RETRIES = 2  # Number of retry attempts with progressively aggressive filtering

    for i, chunk in enumerate(text_chunks):
        logger.info(f"Synthesizing chunk {i+1}/{len(text_chunks)}...")
        
        # DIAGNOSTIC: Log chunk characteristics to identify problematic patterns
        chunk_stripped = chunk.strip()
        logger.debug(f"Chunk {i+1} length: {len(chunk_stripped)} chars")
        logger.debug(f"Chunk {i+1} first 100 chars: {repr(chunk_stripped[:100])}")
        logger.debug(f"Chunk {i+1} last 100 chars: {repr(chunk_stripped[-100:])}")
        
        # Check for suspicious characters
        non_ascii = [c for c in chunk_stripped if ord(c) > 127]
        if non_ascii:
            logger.warning(f"Chunk {i+1} contains {len(non_ascii)} non-ASCII characters: {non_ascii[:10]}")
        
        synthesis_successful = False
        chunk_audio_tensor = None
        chunk_sr_from_engine = None
        
        # Attempt synthesis with retry and progressive filtering
        for attempt in range(MAX_RETRIES + 1):
            try:
                # Apply progressively aggressive filtering on retries
                text_to_synthesize = chunk_stripped
                if attempt == 1:
                    # First retry: Ultra-aggressive ASCII filtering
                    logger.warning(f"Chunk {i+1} retry {attempt}: Applying ultra-aggressive filtering")
                    text_to_synthesize = ''.join(c for c in chunk_stripped if c.isalnum() or c.isspace() or c in '.,!?-')
                    text_to_synthesize = ' '.join(text_to_synthesize.split())  # Normalize whitespace
                elif attempt == 2:
                    # Second retry: Alphanumeric + spaces only
                    logger.warning(f"Chunk {i+1} retry {attempt}: Alphanumeric and spaces only")
                    text_to_synthesize = ''.join(c for c in chunk_stripped if c.isalnum() or c.isspace())
                    text_to_synthesize = ' '.join(text_to_synthesize.split())
                
                if not text_to_synthesize.strip():
                    logger.error(f"Chunk {i+1} attempt {attempt}: Text became empty after filtering")
                    raise ValueError("Text empty after aggressive filtering")
                
                chunk_audio_tensor, chunk_sr_from_engine = engine.synthesize(
                    text=text_to_synthesize,
                    audio_prompt_path=(
                        str(audio_prompt_path_for_engine)
                        if audio_prompt_path_for_engine
                        else None
                    ),
                    temperature=(
                        request.temperature
                        if request.temperature is not None
                        else get_gen_default_temperature()
                    ),
                    exaggeration=(
                        request.exaggeration
                        if request.exaggeration is not None
                        else get_gen_default_exaggeration()
                    ),
                    cfg_weight=(
                        request.cfg_weight
                        if request.cfg_weight is not None
                        else get_gen_default_cfg_weight()
                    ),
                    seed=(
                        request.seed if request.seed is not None else get_gen_default_seed()
                    ),
                )
                
                # Check for None return (engine caught error but didn't raise)
                if chunk_audio_tensor is None or chunk_sr_from_engine is None:
                    logger.error(f"Chunk {i+1} attempt {attempt}: Engine returned None")
                    raise RuntimeError("Engine returned no audio for this chunk.")
                
                # Success!
                synthesis_successful = True
                if attempt > 0:
                    logger.info(f"Chunk {i+1} succeeded on retry {attempt}")
                break  # Exit retry loop
                
            except HTTPException as http_exc:
                raise http_exc
            except RuntimeError as runtime_err:
                error_msg = str(runtime_err)
                
                # Check if this is a CUDA error
                if "CUDA" in error_msg or "device-side assert" in error_msg:
                    logger.error(f"Chunk {i+1} attempt {attempt}: CUDA error occurred")
                    
                    if attempt < MAX_RETRIES:
                        logger.warning(f"Chunk {i+1}: Will retry with more aggressive filtering")
                        continue  # Try next retry level
                    else:
                        # All retries exhausted
                        logger.critical(f"Chunk {i+1} failed after {MAX_RETRIES} retry attempts")
                        
                        # Save full chunk to file for debugging and manual regeneration
                        chunk_dump_path = get_output_path() / f"failed_chunk_{i+1}_{uuid.uuid4().hex[:8]}.txt"
                        with open(chunk_dump_path, 'w', encoding='utf-8') as f:
                            f.write(f"Failed Chunk {i+1}/{len(text_chunks)}\n")
                            f.write(f"Length: {len(chunk)} characters\n")
                            f.write(f"Error: {error_msg}\n")
                            f.write(f"\n{'='*60}\n")
                            f.write(f"ORIGINAL TEXT:\n")
                            f.write(f"{'='*60}\n\n")
                            f.write(chunk)
                        
                        logger.critical(f"Full chunk content saved to: {chunk_dump_path}")
                        
                        # Check partial success mode
                        if request.allow_partial_success:
                            # Skip this chunk and continue
                            logger.warning(f"Partial success mode: Skipping chunk {i+1}")
                            failed_chunks_info.append({
                                "chunk_number": i + 1,
                                "chunk_text": chunk,
                                "chunk_file": str(chunk_dump_path),
                                "error": error_msg
                            })
                            break  # Exit retry loop, continue to next chunk
                        else:
                            # Crash and restart (original behavior)
                            logger.critical("Partial success disabled. Forcing process exit.")
                            logger.critical(f"Error: {error_msg}")
                            logger.critical(f"Chunk content (first 500): {chunk[:500]}")
                            logger.critical("Exiting for Modal restart with clean CUDA context")
                            sys.exit(1)
                else:
                    # Non-CUDA error
                    logger.error(f"Chunk {i+1} non-CUDA error: {error_msg}")
                    raise
            except Exception as e:
                logger.error(f"Chunk {i+1} attempt {attempt}: Unexpected error: {e}")
                if attempt >= MAX_RETRIES:
                    raise
        
        # If synthesis was successful, process the audio
        if synthesis_successful:
            perf_monitor.record(f"Chunk {i+1} synthesized")
            if engine_output_sample_rate is None:
                engine_output_sample_rate = chunk_sr_from_engine
                logger.info(
                    f"Engine sample rate set from first chunk: {engine_output_sample_rate}Hz"
                )
            elif engine_output_sample_rate != chunk_sr_from_engine:
                logger.warning(
                    f"Inconsistent sample rate from engine: chunk {i+1} ({chunk_sr_from_engine}Hz) "
                    f"differs from previous ({engine_output_sample_rate}Hz). Using first chunk's SR."
                )

            current_processed_audio_tensor = chunk_audio_tensor

            speed_factor_to_use = (
                request.speed_factor
                if request.speed_factor is not None
                else get_gen_default_speed_factor()
            )
            if speed_factor_to_use != 1.0:
                current_processed_audio_tensor, _ = utils.apply_speed_factor(
                    current_processed_audio_tensor,
                    chunk_sr_from_engine,
                    speed_factor_to_use,
                )
                perf_monitor.record(f"Speed factor applied to chunk {i+1}")

            processed_audio_np = current_processed_audio_tensor.cpu().numpy().squeeze()
            all_audio_segments_np.append(processed_audio_np)
            
            # Clear GPU memory cache after each successful chunk to prevent accumulation
            # This helps prevent CUDA assertion errors on later chunks, especially the final chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug(f"GPU memory cache cleared after chunk {i+1}")
        else:
            # Chunk was skipped due to failure - add silence placeholder if in partial success mode
            if request.allow_partial_success:
                logger.warning(f"Adding silence placeholder for skipped chunk {i+1}")
                silence_duration = 1.0  # 1 second of silence
                silence = np.zeros(int(silence_duration * (engine_output_sample_rate or final_output_sample_rate)))
                all_audio_segments_np.append(silence)

    # After all chunks: Log summary of failed chunks
    if failed_chunks_info:
        logger.warning(f"\n{'='*60}")
        logger.warning(f"PARTIAL SUCCESS: {len(failed_chunks_info)} chunk(s) failed")
        logger.warning(f"{'='*60}")
        for failed_info in failed_chunks_info:
            logger.warning(f"  - Chunk {failed_info['chunk_number']}: {failed_info['chunk_file']}")
        logger.warning(f"{'='*60}\n")
        
        # Also save a summary file
        summary_path = get_output_path() / f"failed_chunks_summary_{uuid.uuid4().hex[:8]}.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Failed Chunks Summary\n")
            f.write(f"Total chunks processed: {len(text_chunks)}\n")
            f.write(f"Failed chunks: {len(failed_chunks_info)}\n\n")
            for failed_info in failed_chunks_info:
                f.write(f"\n{'='*60}\n")
                f.write(f"Chunk #{failed_info['chunk_number']}\n")
                f.write(f"Error: {failed_info['error']}\n")
                f.write(f"Full text saved to: {failed_info['chunk_file']}\n")
                f.write(f"Text preview (first 200 chars):\n")
                f.write(f"{failed_info['chunk_text'][:200]}\n")
        logger.warning(f"Failed chunks summary saved to: {summary_path}")

    if not all_audio_segments_np:
        logger.error("No audio chunks were generated.")
        raise HTTPException(
            status_code=500,
            detail="No audio was generated from the provided text.",
        )

    try:
        final_audio_np = np.concatenate(all_audio_segments_np)
        perf_monitor.record("Audio concatenation completed")
    except Exception as e_concat:
        logger.exception(f"Error concatenating audio segments: {e_concat}")
        for idx, seg in enumerate(all_audio_segments_np):
            logger.error(f"Segment {idx} shape: {seg.shape}, dtype: {seg.dtype}")
        raise HTTPException(
            status_code=500, detail=f"Audio concatenation error: {e_concat}"
        )

    output_format_str = (
        request.output_format if request.output_format else get_audio_output_format()
    )

    encoded_audio_bytes = utils.encode_audio(
        audio_array=final_audio_np,
        sample_rate=engine_output_sample_rate,
        output_format=output_format_str,
        target_sample_rate=final_output_sample_rate,
    )
    perf_monitor.record(
        f"Final audio encoded to {output_format_str} (target SR: {final_output_sample_rate}Hz "
        f"from engine SR: {engine_output_sample_rate}Hz)"
    )

    if encoded_audio_bytes is None or len(encoded_audio_bytes) < 100:
        logger.error(
            f"Failed to encode final audio to format: {output_format_str} "
            f"or output is too small ({len(encoded_audio_bytes or b'')} bytes)."
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to encode audio to {output_format_str} or generated invalid audio.",
        )

    media_type = f"audio/{output_format_str}"
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    suggested_filename_base = f"tts_output_{timestamp_str}"
    download_filename = utils.sanitize_filename(
        f"{suggested_filename_base}.{output_format_str}"
    )
    headers = {"Content-Disposition": f'attachment; filename="{download_filename}"'}

    # Save generated audio to disk in the configured output folder
    try:
        output_dir = get_output_path(ensure_absolute=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir / download_filename

        with open(output_file_path, "wb") as f:
            f.write(encoded_audio_bytes)

        logger.info(f"Saved generated audio file to: {output_file_path}")
    except Exception as e_save:
        logger.error(f"Failed to save generated audio to disk: {e_save}")

    logger.info(
        f"Successfully generated audio: {download_filename}, {len(encoded_audio_bytes)} bytes, type {media_type}."
    )
    logger.debug(perf_monitor.report())

    return StreamingResponse(
        io.BytesIO(encoded_audio_bytes), media_type=media_type, headers=headers
    )


# --------------------------------------------------------------------------------------
# Background Job Endpoint: /tts_job
# --------------------------------------------------------------------------------------


def _run_tts_job_worker(payload: dict):
    """
    Internal worker that calls the existing /tts endpoint from within the server
    so that long-form generation can proceed even after the original client
    disconnects. The /tts endpoint is responsible for saving the final audio
    file into the configured outputs directory.
    """
    try:
        port = get_port()
        base_url = f"http://127.0.0.1:{port}"
        resp = requests.post(f"{base_url}/tts", json=payload, stream=True, timeout=None)

        for _ in resp.iter_content(chunk_size=1024 * 1024):
            pass

        logger.info(
            f"/tts_job worker finished with status {resp.status_code} "
            f"and content-length={resp.headers.get('Content-Length')}"
        )
    except Exception as e:
        logger.error(f"/tts_job worker failed: {e}", exc_info=True)


@app.post("/tts_job")
async def tts_job_endpoint(request: CustomTTSRequest):
    """
    Starts a background TTS job using the same pipeline as /tts, but decoupled
    from the client connection. The request returns quickly with a job_id while
    the actual synthesis continues in a background worker.

    The generated audio file will be saved into the outputs directory by the
    /tts endpoint, and can be accessed later (e.g., via your Modal file manager).
    """
    job_payload = request.dict()
    job_id = str(uuid.uuid4())

    worker_thread = threading.Thread(
        target=_run_tts_job_worker,
        args=(job_payload,),
        daemon=True,
    )
    worker_thread.start()

    logger.info(f"Started background TTS job with id={job_id}")
    return {"job_id": job_id, "status": "started"}


# --------------------------------------------------------------------------------------
# OpenAI-Compatible Endpoint
# --------------------------------------------------------------------------------------


@app.post("/v1/audio/speech", tags=["OpenAI Compatible"])
async def openai_speech_endpoint(request: OpenAISpeechRequest):
    if not engine.MODEL_LOADED:
        logger.error("OpenAI TTS request failed: Model not loaded.")
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded. Please check server logs.",
        )

    if not request.input_ or not request.input_.strip():
        raise HTTPException(
            status_code=400,
            detail="Text input cannot be empty.",
        )

    voice_name = request.voice
    audio_prompt_path_for_engine: Optional[Path] = None

    predefined_dir = get_predefined_voices_path(ensure_absolute=True)
    ref_dir = get_reference_audio_path(ensure_absolute=True)

    predefined_candidate = predefined_dir / voice_name
    reference_candidate = ref_dir / voice_name

    if predefined_candidate.is_file():
        audio_prompt_path_for_engine = predefined_candidate
        logger.info(f"OpenAI TTS using predefined voice file: {predefined_candidate}")
    elif reference_candidate.is_file():
        audio_prompt_path_for_engine = reference_candidate
        logger.info(f"OpenAI TTS using reference audio file: {reference_candidate}")
    else:
        logger.warning(
            f"OpenAI TTS: voice file '{voice_name}' not found in predefined or reference directories."
        )

    try:
        audio_tensor, sr_from_engine = engine.synthesize(
            text=request.input_,
            audio_prompt_path=(
                str(audio_prompt_path_for_engine)
                if audio_prompt_path_for_engine
                else None
            ),
            temperature=0.7,
            exaggeration=0.0,
            cfg_weight=3.0,
            seed=request.seed or 0,
        )
    except Exception as e:
        logger.exception(f"OpenAI-compatible TTS failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI-compatible TTS failed: {e}",
        )

    if audio_tensor is None or sr_from_engine is None:
        raise HTTPException(
            status_code=500,
            detail="Engine returned no audio.",
        )

    if request.speed != 1.0:
        audio_tensor, _ = utils.apply_speed_factor(
            audio_tensor, sr_from_engine, request.speed
        )

    audio_np = audio_tensor.cpu().numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np.squeeze()

    encoded_audio = utils.encode_audio(
        audio_array=audio_np,
        sample_rate=sr_from_engine,
        output_format=request.response_format,
        target_sample_rate=get_audio_sample_rate(),
    )

    if encoded_audio is None or len(encoded_audio) < 100:
        raise HTTPException(
            status_code=500,
            detail="Failed to encode audio or generated invalid audio.",
        )

    media_type = f"audio/{request.response_format}"
    return StreamingResponse(
        io.BytesIO(encoded_audio),
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="speech.{request.response_format}"'
        },
    )


# --------------------------------------------------------------------------------------
# Main Entrypoint (for local runs)
# --------------------------------------------------------------------------------------


if __name__ == "__main__":
    server_host = get_host()
    server_port = get_port()
    logger.info(f"Starting TTS server on {server_host}:{server_port}")
    import uvicorn

    uvicorn.run(
        "server:app",
        host=server_host,
        port=server_port,
        log_level="info",
        workers=1,
        reload=False,
    )
