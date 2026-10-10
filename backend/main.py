import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.config import MAX_FILE_SIZE, ALLOWED_EXTENSIONS
from backend.services.inference import inference_service
from backend.services.samples import sample_service
from backend.services.rpi_stream import rpi_stream_store
from backend.services import frame_collector
from backend.services.edge_ingest import EdgeIngestError, parse_meta, build_state

app = FastAPI(
    title="AquaScan AI — Marine Waste Perception Service",
    description="High-performance object detection REST service for real-time floating water pollution perception powered by YOLOv8",
    version="1.0.0"
)

# Enable CORS for cross-origin frontend & Raspberry Pi LAN support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend files are served without cache headers by StaticFiles, so browsers may heuristically
# reuse a stale app.js for days after it changes. Force revalidation (cheap 304 via ETag).
@app.middleware("http")
async def no_cache_frontend(request, call_next):
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response

@app.get("/api/status")
async def get_status():
    """Returns active model status, device configuration, loaded classes, and CUDA availability."""
    try:
        return inference_service.get_status()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "status": "unavailable", "error": str(e)}
        )

@app.get("/api/samples")
async def list_samples():
    """Returns curated demo sample images for presentation testing."""
    try:
        return {"success": True, "samples": sample_service.get_sample_images()}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "samples": [], "error": str(e)}
        )

@app.get("/api/samples/{sample_id}/image")
async def get_sample_image(sample_id: str):
    """Serves the raw JPEG image bytes for a demo sample thumbnail/preview."""
    try:
        image_bytes, filename = sample_service.get_sample_bytes(sample_id)
        return Response(content=image_bytes, media_type="image/jpeg")
    except FileNotFoundError as e:
        return JSONResponse(
            status_code=404,
            content={"success": False, "error_type": "sample_not_found", "error": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error_type": "server_error", "error": str(e)}
        )

@app.get("/api/rpi/latest")
async def get_latest_rpi_frame():
    """Returns metadata and prediction for the latest Raspberry Pi frame received."""
    return rpi_stream_store.get_latest_state()

@app.get("/api/rpi/latest/image")
async def get_latest_rpi_image():
    """Serves the raw JPEG image bytes of the latest Raspberry Pi frame."""
    image_bytes = rpi_stream_store.get_latest_image_bytes()
    if not image_bytes:
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": "No Raspberry Pi image available yet"}
        )
    return Response(content=image_bytes, media_type="image/jpeg")

@app.post("/api/edge/ingest")
async def edge_ingest(file: UploadFile = File(...), meta: str = Form(...)):
    """Edge mode: the Raspberry Pi already ran inference. Store its frame + detections for the
    Raspberry Pi dashboard panel. No inference on the laptop for this path."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(status_code=400, content={"success": False, "error_type": "invalid_extension",
                            "error": f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"})
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        return JSONResponse(status_code=400, content={"success": False, "error_type": "empty_file", "error": "Uploaded image file is empty (0 bytes)."})
    if len(image_bytes) > MAX_FILE_SIZE:
        return JSONResponse(status_code=400, content={"success": False, "error_type": "oversized_upload",
                            "error": f"File size exceeds maximum allowed upload limit of {MAX_FILE_SIZE // (1024*1024)} MB."})
    try:
        prediction, metadata = await run_in_threadpool(build_state, image_bytes, parse_meta(meta))
    except EdgeIngestError as e:
        return JSONResponse(status_code=400, content={"success": False, "error_type": "invalid_edge_payload", "error": str(e)})
    rpi_stream_store.update(image_bytes, prediction, metadata)
    return {"success": True, "frame_id": metadata["frame_id"], "count": prediction["count"], "received_at": metadata["received_at"]}

@app.post("/api/predict")
async def predict(
    request: Request,
    file: UploadFile = File(None),
    sample_id: str = Form(None),
    conf_threshold: float = Form(0.25),
    source: str = Form(None),
    device_id: str = Form(None),
    camera_id: str = Form(None),
    frame_id: int = Form(None),
    timestamp: str = Form(None)
):
    """
    Main prediction endpoint.
    Accepts uploaded image file OR demo sample_id, with optional hardware metadata.
    """
    # Validate confidence threshold range
    if conf_threshold < 0.0 or conf_threshold > 1.0:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error_type": "invalid_confidence",
                "error": f"Confidence threshold must be between 0.0 and 1.0 (received {conf_threshold})."
            }
        )

    # Build optional hardware metadata dictionary if provided
    hardware_metadata = {}
    if source: hardware_metadata["source"] = source
    if device_id: hardware_metadata["device_id"] = device_id
    if camera_id: hardware_metadata["camera_id"] = camera_id
    if frame_id is not None: hardware_metadata["frame_id"] = frame_id
    if timestamp: hardware_metadata["timestamp"] = timestamp

    image_bytes = None
    
    if file and file.filename:
        # Extension Validation
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error_type": "invalid_extension",
                    "error": f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                }
            )
            
        image_bytes = await file.read()

        if len(image_bytes) == 0:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error_type": "empty_file",
                    "error": "Uploaded image file is empty (0 bytes)."
                }
            )

        if len(image_bytes) > MAX_FILE_SIZE:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error_type": "oversized_upload",
                    "error": f"File size exceeds maximum allowed upload limit of {MAX_FILE_SIZE // (1024*1024)} MB."
                }
            )
            
    elif sample_id:
        try:
            image_bytes, _ = sample_service.get_sample_bytes(sample_id)
        except Exception as e:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error_type": "sample_not_found", "error": str(e)}
            )
    else:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error_type": "missing_input",
                "error": "No input image provided. Please upload a file or specify a valid sample_id."
            }
        )

    try:
        source_str = (source or "").lower().strip()
        is_laptop_webcam = (source_str in ["laptop_webcam", "laptop", "webcam", "laptop-webcam", "camera"] or (hardware_metadata and str(hardware_metadata.get("source", "")).lower().strip() in ["laptop_webcam", "laptop", "webcam", "laptop-webcam", "camera"]))
        skip_annotated_image = is_laptop_webcam
        if is_laptop_webcam and image_bytes:
            try:
                debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch")
                os.makedirs(debug_dir, exist_ok=True)
                debug_path = os.path.join(debug_dir, "webcam_debug.jpg")
                with open(debug_path, "wb") as f:
                    f.write(image_bytes)
                print(f"[AquaScan Debug] Saved laptop webcam frame to {debug_path} ({len(image_bytes)} bytes)")
            except Exception as debug_err:
                print(f"[AquaScan Debug] Failed to save debug frame: {debug_err}")

        inference_conf = min(conf_threshold, 0.20) if is_laptop_webcam else conf_threshold
        capture = {} if frame_collector.ENABLED else None
        results = await run_in_threadpool(
            inference_service.predict,
            image_bytes,
            conf_threshold=inference_conf,
            hardware_metadata=hardware_metadata if hardware_metadata else None,
            skip_annotated_image=skip_annotated_image,
            capture=capture
        )

        if capture is not None:
            await run_in_threadpool(
                frame_collector.save, frame_collector.resolve_phase(request), image_bytes, capture,
                source, frame_id, conf_threshold, results
            )

        # Cache in-memory if coming from Raspberry Pi hardware client
        if source == "raspberry_pi" or (hardware_metadata and hardware_metadata.get("source") == "raspberry_pi"):
            rpi_stream_store.update(image_bytes, results, hardware_metadata)

        return JSONResponse(content=results)
    except ValueError as ve:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error_type": "corrupt_image",
                "error": str(ve)
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_type": "inference_failure",
                "error": f"Model inference execution failed: {str(e)}"
            }
        )

# Mount static frontend Single Page Application
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


