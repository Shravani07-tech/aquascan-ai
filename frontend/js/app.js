/* ==========================================================================
   AquaScan AI — Unified Dual Camera & Perception Platform Controller
   Supports: Laptop Webcam, Raspberry Pi Camera, & Image Upload / Demo Gallery
   ========================================================================== */

// Fixed laptop-webcam confidence threshold (the backend also caps webcam requests at 0.20)
const WEBCAM_CONF = 0.20;

document.addEventListener('DOMContentLoaded', () => {
  
  // Application State
  const state = {
    activeSourceMode: 'webcam', // 'webcam' | 'rpi' | 'upload'
    selectedFile: null,
    selectedSampleId: null,
    currentResult: null,
    activeViewMode: 'canvas',
    confThreshold: 0.25,
    status: null,

    // Laptop Webcam State
    webcamStream: null,
    webcamTimer: null,
    isPredictingWebcam: false,
    isStartingWebcam: false,
    webcamFrameSeq: 0,
    latestAppliedSeq: 0,
    webcamAbortController: null,

    // Raspberry Pi Polling State
    rpiTimer: null,
    isPollingRPi: false
  };

  // DOM Elements — Header & Status
  const modelStatusPill = document.getElementById('model-status-pill');
  const modelStatusText = document.getElementById('model-status-text');
  const deviceInfoText = document.getElementById('device-info-text');
  const standbyBanner = document.getElementById('standby-banner');

  // Source Selector & Sidebar Panels
  const sourceSelector = document.getElementById('source-selector');
  const sourceBtns = sourceSelector.querySelectorAll('.source-btn');
  const panelWebcam = document.getElementById('panel-section-webcam');
  const panelRPi = document.getElementById('panel-section-rpi');
  const panelUpload = document.getElementById('panel-section-upload');
  const panelSamples = document.getElementById('panel-section-samples');
  const uploadActionGroup = document.getElementById('upload-action-group');

  // Webcam Controls
  const btnStartWebcam = document.getElementById('btn-start-webcam');
  const btnStopWebcam = document.getElementById('btn-stop-webcam');
  const webcamDot = document.getElementById('webcam-dot');
  const webcamStatusText = document.getElementById('webcam-status-text');

  // Raspberry Pi Telemetry
  const rpiValSource = document.getElementById('rpi-val-source');
  const rpiValDevice = document.getElementById('rpi-val-device');
  const rpiValCamera = document.getElementById('rpi-val-camera');
  const rpiValFrame = document.getElementById('rpi-val-frame');
  const rpiValStatus = document.getElementById('rpi-val-status');
  const rpiValInference = document.getElementById('rpi-val-inference');
  const EDGE_STALE_S = 3; // edge frames older than this are shown as OFFLINE, not live

  // Upload & Demo Controls
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const dropzoneIdle = document.getElementById('dropzone-idle');
  const dropzonePreview = document.getElementById('dropzone-preview');
  const previewThumb = document.getElementById('preview-thumb');
  const previewFilename = document.getElementById('preview-filename');
  const previewFilesize = document.getElementById('preview-filesize');
  const btnRemoveFile = document.getElementById('btn-remove-file');
  const sampleGrid = document.getElementById('sample-grid');
  
  const confSlider = document.getElementById('conf-slider');
  const confValText = document.getElementById('conf-val');
  const btnAnalyze = document.getElementById('btn-analyze');
  const btnReset = document.getElementById('btn-reset');
  const btnSpinner = document.getElementById('btn-spinner');

  // Viewport & View States
  const viewToggle = document.getElementById('view-toggle');
  const toggleBtns = viewToggle.querySelectorAll('.toggle-btn');
  const stateIdle = document.getElementById('state-idle');
  const idleTitle = document.getElementById('idle-title');
  const idleDesc = document.getElementById('idle-desc');
  const stateLoading = document.getElementById('state-loading');
  const stateClean = document.getElementById('state-clean');
  const stateError = document.getElementById('state-error');
  const btnErrorRetry = document.getElementById('btn-error-retry');

  const canvasWrapper = document.getElementById('canvas-wrapper');
  const canvasContainer = document.getElementById('canvas-container');
  const webcamVideo = document.getElementById('webcam-video');
  const rpiImage = document.getElementById('rpi-image');
  const detectionCanvas = document.getElementById('detection-canvas');
  const annotatedImage = document.getElementById('annotated-image');

  // Analytics Bar
  const detectionCountBadge = document.getElementById('detection-count-badge');
  const countBottle = document.getElementById('count-bottle');
  const countCan = document.getElementById('count-can');
  const countCarton = document.getElementById('count-carton');
  const countPaper = document.getElementById('count-paper');
  const countPlastic = document.getElementById('count-plastic');
  const statLatency = document.getElementById('stat-latency');
  const statFps = document.getElementById('stat-fps');

  // Offscreen Canvas for Webcam Frame Extraction
  const offscreenCanvas = document.createElement('canvas');
  const offscreenCtx = offscreenCanvas.getContext('2d');

  // Initialize
  init();

  function init() {
    confSlider.value = Math.round(state.confThreshold * 100);
    confValText.textContent = `${confSlider.value}%`;
    checkModelStatus();
    loadDemoSamples();
    setupEventListeners();
    setSourceMode('webcam');
  }

  // Check Backend Status
  async function checkModelStatus() {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) throw new Error('Status API unavailable');
      const data = await res.json();
      state.status = data;

      const modelName = data.model || 'yolov8n';
      const devName = data.cuda_available ? 'CUDA' : 'CPU';
      
      if (data.is_fallback) {
        modelStatusText.textContent = 'STANDBY (yolov8n.pt)';
        standbyBanner.classList.remove('hidden');
      } else {
        modelStatusText.textContent = 'MODEL ONLINE';
        standbyBanner.classList.add('hidden');
      }

      deviceInfoText.textContent = `${modelName} • ${devName} • Ready`;
    } catch (e) {
      console.warn('Backend status check failed:', e);
      modelStatusText.textContent = 'API OFFLINE';
      modelStatusPill.style.background = 'rgba(255, 0, 84, 0.15)';
      deviceInfoText.textContent = 'Offline';
    }
  }

  // Load Demo Samples
  async function loadDemoSamples() {
    try {
      const res = await fetch('/api/samples');
      if (!res.ok) throw new Error('Failed to load samples');
      const data = await res.json();
      
      sampleGrid.innerHTML = '';
      if (data.samples && data.samples.length > 0) {
        data.samples.forEach(sample => {
          const item = document.createElement('div');
          item.className = 'sample-item';
          item.dataset.sampleId = sample.id;
          const imgUrl = sample.image_url || `/api/samples/${sample.id}/image`;
          item.innerHTML = `
            <img src="${imgUrl}" alt="${sample.name}" loading="lazy">
            <span class="sample-item-label">${sample.name}</span>
          `;
          item.addEventListener('click', () => selectSample(sample, item));
          sampleGrid.appendChild(item);
        });
      } else {
        sampleGrid.innerHTML = '<p class="section-desc">No sample images found.</p>';
      }
    } catch (e) {
      sampleGrid.innerHTML = '<p class="section-desc">Sample gallery unavailable.</p>';
    }
  }

  // Setup Event Listeners
  function setupEventListeners() {
    // Source Selector Buttons
    sourceBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const targetSource = btn.dataset.source;
        setSourceMode(targetSource);
      });
    });

    // Laptop Webcam Buttons
    btnStartWebcam.addEventListener('click', () => {
      console.log('[AquaScan Webcam] "Start Camera" button clicked.');
      startWebcam();
    });
    btnStopWebcam.addEventListener('click', () => {
      console.log('[AquaScan Webcam] "Stop Camera" button clicked.');
      stopWebcam();
    });

    // File Dropzone Events
    dropzone.addEventListener('click', (e) => {
      if (e.target !== btnRemoveFile && !btnRemoveFile.contains(e.target)) {
        fileInput.click();
      }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      }, false);
    });

    dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        handleFileSelect(files[0]);
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files && fileInput.files.length > 0) {
        handleFileSelect(fileInput.files[0]);
      }
    });

    btnRemoveFile.addEventListener('click', (e) => {
      e.stopPropagation();
      resetInputState();
    });

    // Confidence Slider
    confSlider.addEventListener('input', (e) => {
      state.confThreshold = parseInt(e.target.value) / 100.0;
      confValText.textContent = `${e.target.value}%`;
      
      if (state.currentResult) {
        renderDetections();
      }
    });

    // View Toggle Buttons (Interactive Canvas vs Rendered Overlay)
    toggleBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        toggleBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.activeViewMode = btn.dataset.mode;
        
        if (state.activeViewMode === 'canvas') {
          detectionCanvas.classList.remove('hidden');
          annotatedImage.classList.add('hidden');
        } else {
          detectionCanvas.classList.add('hidden');
          annotatedImage.classList.remove('hidden');
        }
      });
    });

    // Upload Mode Buttons
    btnAnalyze.addEventListener('click', runInference);
    btnReset.addEventListener('click', resetAll);
    btnErrorRetry.addEventListener('click', () => {
      if (state.activeSourceMode === 'upload') {
        runInference();
      } else if (state.activeSourceMode === 'webcam') {
        stopWebcam();
        startWebcam();
      } else if (state.activeSourceMode === 'rpi') {
        startRPiPolling();
      }
    });

    window.addEventListener('resize', () => {
      if (state.activeSourceMode === 'webcam' && !webcamVideo.classList.contains('hidden')) {
        alignCanvasWithElement(webcamVideo, detectionCanvas, webcamVideo.videoWidth, webcamVideo.videoHeight);
      } else if (state.activeSourceMode === 'rpi' && !rpiImage.classList.contains('hidden')) {
        alignCanvasWithElement(rpiImage, detectionCanvas, rpiImage.naturalWidth, rpiImage.naturalHeight);
      }
    });
  }

  // =========================================================================
  // CAMERA SOURCE MODE SWITCHER
  // =========================================================================
  function setSourceMode(mode) {
    // Teardown active streams/timers from previous mode
    if (state.activeSourceMode === 'webcam' && mode !== 'webcam') {
      stopWebcam();
    }
    if (state.activeSourceMode === 'rpi' && mode !== 'rpi') {
      stopRPiPolling();
    }

    state.activeSourceMode = mode;
    state.currentResult = null;

    // Webcam mode uses a fixed confidence threshold; show it instead of a slider that has no effect
    confSlider.disabled = mode === 'webcam';
    confValText.textContent = mode === 'webcam' ? `${Math.round(WEBCAM_CONF * 100)}% (fixed in webcam mode)` : `${confSlider.value}%`;
    resetAnalytics();

    // Update Tab UI
    sourceBtns.forEach(btn => {
      if (btn.dataset.source === mode) btn.classList.add('active');
      else btn.classList.remove('active');
    });

    // Hide all panel sections first
    [panelWebcam, panelRPi, panelUpload, panelSamples, uploadActionGroup].forEach(el => el.classList.add('hidden'));

    // Hide viewport elements
    webcamVideo.classList.add('hidden');
    rpiImage.classList.add('hidden');
    canvasContainer.classList.remove('overlay-mode');

    if (mode === 'webcam') {
      panelWebcam.classList.remove('hidden');
      idleTitle.textContent = 'Laptop Webcam Ready';
      idleDesc.textContent = 'Click "Start Camera" to enable browser live video and execute YOLOv8 marine waste perception.';
      showState('idle');
    } else if (mode === 'rpi') {
      panelRPi.classList.remove('hidden');
      idleTitle.textContent = 'Raspberry Pi Camera Stream';
      idleDesc.textContent = 'Connecting to edge camera client via LAN API pipeline...';
      showState('idle');
      startRPiPolling();
    } else if (mode === 'upload') {
      panelUpload.classList.remove('hidden');
      panelSamples.classList.remove('hidden');
      uploadActionGroup.classList.remove('hidden');
      idleTitle.textContent = 'Awaiting Marine Image';
      idleDesc.textContent = 'Upload a photo or click any sample image from the left panel to execute real-time YOLOv8 waste detection.';
      showState('idle');
    }
  }

  // =========================================================================
  // MODE 1: LAPTOP WEBCAM CONTROLLER
  // =========================================================================
  async function startWebcam() {
    console.log('[AquaScan Webcam Startup] 1. "Start Camera" button clicked...');
    
    // Idempotency check: prevent duplicate simultaneous camera initialization
    if (state.isStartingWebcam) {
      console.warn('[AquaScan Webcam Startup] Camera initialization already in progress. Ignoring duplicate start request.');
      return;
    }

    // If stream is already active and video is playing, do nothing
    if (state.webcamStream && webcamVideo && !webcamVideo.paused && webcamVideo.srcObject) {
      console.log('[AquaScan Webcam Startup] Camera is already active and playing.');
      return;
    }

    state.isStartingWebcam = true;

    // 1. Complete teardown of any previous webcam session before starting anew
    stopWebcam();

    try {
      webcamStatusText.textContent = 'Requesting camera access...';
      webcamDot.className = 'status-dot-sm streaming';

      console.log('[AquaScan Webcam Startup] 2. Checking navigator.mediaDevices availability:', !!(navigator && navigator.mediaDevices));
      if (!navigator || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('navigator.mediaDevices.getUserMedia is not supported on this browser context.');
      }

      // Explicitly reset prediction lock & sequence counters
      state.isPredictingWebcam = false;
      state.webcamFrameSeq = 0;
      state.latestAppliedSeq = 0;

      console.log('[AquaScan Webcam Startup] 3. Calling getUserMedia() for fresh stream...');

      let stream = null;
      const primaryConstraints = {
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false
      };

      try {
        stream = await navigator.mediaDevices.getUserMedia(primaryConstraints);
      } catch (primaryErr) {
        console.warn('[AquaScan Webcam Startup] Primary constraints failed:', primaryErr.name, primaryErr.message);
        if (primaryErr.name === 'NotAllowedError' || primaryErr.name === 'PermissionDeniedError') {
          throw primaryErr;
        }
        console.log('[AquaScan Webcam Startup] Retrying with fallback video: true constraint...');
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      }

      console.log('[AquaScan Webcam Startup] 4. getUserMedia() succeeded! Tracks:', stream.getVideoTracks().map(t => `${t.label} (enabled=${t.enabled}, readyState=${t.readyState})`));

      // Monitor track readyState & handle unexpected track termination
      stream.getVideoTracks().forEach(track => {
        track.addEventListener('ended', () => {
          console.warn('[AquaScan Webcam] Stream track ended unexpectedly! Triggering auto-recovery teardown...');
          stopWebcam();
          webcamStatusText.textContent = 'Camera Disconnected';
          webcamDot.className = 'status-dot-sm error';
        });
      });

      state.webcamStream = stream;

      // Configure video element attributes safely
      webcamVideo.muted = true;
      webcamVideo.autoplay = true;
      webcamVideo.playsInline = true;
      webcamVideo.srcObject = stream;

      console.log('[AquaScan Webcam Startup] 5. Stream assigned to #webcam-video.srcObject. readyState:', webcamVideo.readyState);

      if (webcamVideo.readyState < 1) {
        console.log('[AquaScan Webcam Startup] Waiting for video metadata event...');
        await new Promise(resolve => {
          let done = false;
          const onMeta = () => {
            if (!done) {
              done = true;
              webcamVideo.removeEventListener('loadedmetadata', onMeta);
              resolve();
            }
          };
          webcamVideo.addEventListener('loadedmetadata', onMeta);
          setTimeout(() => {
            if (!done) {
              done = true;
              console.warn('[AquaScan Webcam Startup] Metadata timeout (500ms fallback). Proceeding...');
              webcamVideo.removeEventListener('loadedmetadata', onMeta);
              resolve();
            }
          }, 500);
        });
      }

      console.log('[AquaScan Webcam Startup] Video readyState:', webcamVideo.readyState, '| Video dimensions:', webcamVideo.videoWidth, 'x', webcamVideo.videoHeight);

      try {
        await webcamVideo.play();
        console.log('[AquaScan Webcam Startup] 6. video.play() resolved successfully. Paused:', webcamVideo.paused);
      } catch (playErr) {
        console.error('[AquaScan Webcam Startup] video.play() error:', playErr);
      }

      webcamVideo.classList.remove('hidden');
      rpiImage.classList.add('hidden');
      canvasContainer.classList.add('overlay-mode');

      btnStartWebcam.classList.add('hidden');
      btnStopWebcam.classList.remove('hidden');
      webcamStatusText.textContent = 'Live Webcam Active (YOLO ~3 FPS)';
      webcamDot.className = 'status-dot-sm online';

      showState('canvas');
      console.log('[AquaScan Webcam Startup] 7. Canvas state activated.');

      // Initial canvas alignment with video element
      alignCanvasWithElement(webcamVideo, detectionCanvas, webcamVideo.videoWidth || 640, webcamVideo.videoHeight || 480);

      // Start non-overlapping inference loop at ~300ms interval
      if (state.webcamTimer) clearInterval(state.webcamTimer);
      state.webcamTimer = setInterval(captureAndPredictWebcamFrame, 300);
      console.log('[AquaScan Webcam Startup] 8. Inference interval timer started (300ms).');

    } catch (err) {
      console.error('[AquaScan Webcam Startup] CRITICAL STARTUP ERROR:', err.name, err.message, err);
      stopWebcam();

      webcamDot.className = 'status-dot-sm error';
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        webcamStatusText.textContent = 'Permission Denied';
        showError(
          'Camera Permission Required',
          'Camera access was denied by browser or system permissions. Please allow camera access in browser settings (click lock/camera icon in address bar), then click "Try Again" or use "Upload / Demo" mode.'
        );
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        webcamStatusText.textContent = 'Camera Occupied';
        showError(
          'Camera Occupied / Blocked',
          'The camera is currently locked or in use by another app (e.g., the hardware\\rpi_client.py terminal test, another browser, Zoom, Teams, Skype, or Windows Camera app). Windows allows only one app to use the camera at a time. Please close competing camera apps and click "Try Again", or use "Upload / Demo" mode on the left for presentation.'
        );
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        webcamStatusText.textContent = 'No Camera Found';
        showError(
          'No Camera Device Found',
          'No webcam hardware was detected on this device. Please connect a camera or use "Upload / Demo" mode on the left panel.'
        );
      } else {
        webcamStatusText.textContent = `Camera Unavailable: ${err.name || 'Error'}`;
        showError(
          'Unable to Access Camera',
          `Camera error (${err.name || 'Failed'}): ${err.message || 'Could not start video source. Check browser permissions or try Upload/Demo mode.'}`
        );
      }
    } finally {
      state.isStartingWebcam = false;
    }
  }

  function stopWebcam() {
    console.log('[AquaScan Webcam] Stopping and tearing down webcam session...');
    if (state.webcamTimer) {
      clearInterval(state.webcamTimer);
      state.webcamTimer = null;
    }

    if (state.webcamAbortController) {
      state.webcamAbortController.abort();
      state.webcamAbortController = null;
    }

    if (state.webcamStream) {
      state.webcamStream.getTracks().forEach(track => {
        console.log('[AquaScan Webcam] Stopping state track:', track.label, 'readyState:', track.readyState);
        try { track.stop(); } catch (e) {}
      });
      state.webcamStream = null;
    }

    if (webcamVideo) {
      if (webcamVideo.srcObject) {
        if (typeof webcamVideo.srcObject.getTracks === 'function') {
          webcamVideo.srcObject.getTracks().forEach(track => {
            console.log('[AquaScan Webcam] Stopping element track:', track.label);
            try { track.stop(); } catch (e) {}
          });
        }
        webcamVideo.srcObject = null;
      }
      try { webcamVideo.pause(); } catch (e) {}
      webcamVideo.classList.add('hidden');
    }

    canvasContainer.classList.remove('overlay-mode');

    btnStartWebcam.classList.remove('hidden');
    btnStopWebcam.classList.add('hidden');
    webcamStatusText.textContent = 'Webcam Standby';
    webcamDot.className = 'status-dot-sm';

    // Instantly clear detection canvas & state
    state.currentResult = null;
    state.webcamFrameSeq = 0;
    state.latestAppliedSeq = 0;
    state.isPredictingWebcam = false;
    const ctx = detectionCanvas.getContext('2d');
    ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
    resetAnalytics();

    if (state.activeSourceMode === 'webcam') {
      showState('idle');
    }
  }

  async function captureAndPredictWebcamFrame() {
    if (state.isPredictingWebcam || !state.webcamStream || webcamVideo.paused || webcamVideo.ended) {
      return;
    }

    // Health check: Ensure camera tracks are live and active (recovery after sleep/wake)
    const videoTrack = state.webcamStream.getVideoTracks()[0];
    if (!videoTrack || videoTrack.readyState !== 'live' || videoTrack.ended) {
      console.warn(`[AquaScan Webcam] Video track health check failed (readyState=${videoTrack ? videoTrack.readyState : 'none'}). Stopping session.`);
      stopWebcam();
      return;
    }

    const vw = webcamVideo.videoWidth;
    const vh = webcamVideo.videoHeight;
    if (!vw || !vh) return;

    state.isPredictingWebcam = true;
    const currentSeq = ++state.webcamFrameSeq;

    const controller = new AbortController();
    state.webcamAbortController = controller;
    const timeoutId = setTimeout(() => controller.abort(), 4000);

    try {
      offscreenCanvas.width = vw;
      offscreenCanvas.height = vh;
      offscreenCtx.drawImage(webcamVideo, 0, 0, vw, vh);

      const blob = await new Promise(resolve => offscreenCanvas.toBlob(resolve, 'image/jpeg', 0.85));
      if (!blob) {
        console.error(`[AquaScan Webcam] Frame #${currentSeq} toBlob returned null/undefined`);
        return;
      }

      const formData = new FormData();
      formData.append('file', blob, 'webcam_frame.jpg');
      formData.append('conf_threshold', WEBCAM_CONF);
      formData.append('source', 'laptop_webcam');
      formData.append('frame_id', currentSeq);

      console.log(`[AquaScan Webcam] FRAME SENT #${currentSeq} (${vw}x${vh}, ${blob.size} bytes)`);

      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });

      console.log(`[AquaScan Webcam] HTTP RESPONSE #${currentSeq} Status: ${res.status}`);

      if (!res.ok) {
        if (state.status && !state.status.success) {
          showError('AquaScan Server Offline', 'AquaScan server is offline.');
        }
        return;
      }

      const result = await res.json();
      console.log(`[AquaScan Webcam] DETECTIONS #${currentSeq} Count: ${result.count} | Raw:`, result.detections);

      if (result.success && state.activeSourceMode === 'webcam') {
        const resFrameId = (result.hardware_metadata && result.hardware_metadata.frame_id) || currentSeq;
        if (resFrameId >= state.latestAppliedSeq || (result.count > 0 && resFrameId >= state.latestAppliedSeq - 2)) {
          if (resFrameId > state.latestAppliedSeq) {
            state.latestAppliedSeq = resFrameId;
          }
          state.currentResult = result;
          result.image_dimensions = { width: vw, height: vh };
          console.log(`[AquaScan Webcam] RENDER #${resFrameId} Detections: ${result.count}`);
          renderDetections();
        } else {
          console.log(`[AquaScan Webcam] Discarding stale frame #${resFrameId} (latest applied: #${state.latestAppliedSeq})`);
        }
      }
    } catch (e) {
      if (e.name === 'AbortError') {
        console.warn(`[AquaScan Webcam] Frame #${currentSeq} request aborted / timed out (>4000ms)`);
      } else {
        console.warn(`[AquaScan Webcam] Frame #${currentSeq} exception:`, e);
      }
    } finally {
      clearTimeout(timeoutId);
      state.isPredictingWebcam = false;
      state.webcamAbortController = null;
      console.log(`[AquaScan Webcam] LOCK RELEASED #${currentSeq}`);
    }
  }

  // =========================================================================
  // MODE 2: RASPBERRY PI CAMERA CONTROLLER
  // =========================================================================
  function startRPiPolling() {
    stopRPiPolling();
    state.isPollingRPi = true;
    pollRPiLatestFrame(); // Immediate execution
    state.rpiTimer = setInterval(pollRPiLatestFrame, 400);
  }

  function stopRPiPolling() {
    state.isPollingRPi = false;
    if (state.rpiTimer) {
      clearInterval(state.rpiTimer);
      state.rpiTimer = null;
    }
    rpiValStatus.textContent = 'OFFLINE';
    rpiValStatus.className = 'connection-badge offline';
  }

  async function pollRPiLatestFrame() {
    if (state.activeSourceMode !== 'rpi') return;

    try {
      const res = await fetch('/api/rpi/latest');
      if (!res.ok) throw new Error('API server error');
      const data = await res.json();
      const isEdge = !!(data.hardware_metadata && data.hardware_metadata.inference_location === 'edge');
      const edgeStale = isEdge && data.age_seconds > EDGE_STALE_S;

      if (edgeStale) {
        // Edge mode: no new frame from the Pi for > EDGE_STALE_S; keep the last frame but don't look live
        rpiValStatus.textContent = `OFFLINE — last frame ${Math.round(data.age_seconds)} s ago`;
        rpiValStatus.className = 'connection-badge offline';
      } else if (data.success && data.active && data.prediction) {
        // Active Pi Stream Frame Received
        rpiValStatus.textContent = 'ONLINE';
        rpiValStatus.className = 'connection-badge online';

        const meta = data.hardware_metadata || {};
        if (isEdge) {
          const num = (v, d) => (typeof v === 'number' ? v.toFixed(d) : '--');
          rpiValInference.textContent = `Edge inference (on Pi) · ${num(meta.edge_fps, 1)} FPS · ${num(meta.inference_ms, 0)} ms · ${num(meta.cpu_temp_c, 1)} °C`;
        } else {
          rpiValInference.textContent = 'Laptop inference';
        }
        rpiValSource.textContent = meta.source || 'raspberry_pi';
        rpiValDevice.textContent = meta.device_id || 'aquascan-pi-01';
        rpiValCamera.textContent = meta.camera_id || 'picam-v2';
        rpiValFrame.textContent = meta.frame_id ? `#${String(meta.frame_id).padStart(4, '0')}` : '#----';

        state.currentResult = data.prediction;

        if (data.image_url) {
          rpiImage.src = data.image_url;
          rpiImage.classList.remove('hidden');
          webcamVideo.classList.add('hidden');
          canvasContainer.classList.add('overlay-mode');
        }

        renderDetections();
      } else {
        // No Pi frames or inactive (>5s)
        rpiValStatus.textContent = 'OFFLINE';
        rpiValStatus.className = 'connection-badge offline';

        if (!state.currentResult) {
          showState('idle');
          idleTitle.textContent = 'Waiting for Raspberry Pi camera...';
          idleDesc.textContent = 'Ensure python3 rpi_client.py is running on your Raspberry Pi edge device.';
        }
      }
    } catch (e) {
      rpiValStatus.textContent = 'OFFLINE';
      rpiValStatus.className = 'connection-badge offline';
    }
  }

  // =========================================================================
  // MODE 3: IMAGE UPLOAD & DEMO SAMPLES CONTROLLER
  // =========================================================================
  function handleFileSelect(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['jpg', 'jpeg', 'png', 'webp'].includes(ext)) {
      showError('Invalid File Type', `Unsupported file extension '.${ext}'. Allowed extensions: JPG, JPEG, PNG, WebP.`);
      return;
    }

    if (file.size > 15 * 1024 * 1024) {
      showError('File Too Large', 'Selected image exceeds the 15 MB upload size limit.');
      return;
    }

    clearSampleSelection();
    state.selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      previewThumb.src = e.target.result;
      previewFilename.textContent = file.name;
      previewFilesize.textContent = formatBytes(file.size);
      
      dropzoneIdle.classList.add('hidden');
      dropzonePreview.classList.remove('hidden');
      btnAnalyze.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function selectSample(sample, element) {
    clearFileInput();
    state.selectedSampleId = sample.id;
    
    document.querySelectorAll('.sample-item').forEach(el => el.classList.remove('selected'));
    element.classList.add('selected');
    
    btnAnalyze.disabled = false;
    
    const imgUrl = sample.image_url || `/api/samples/${sample.id}/image`;
    previewThumb.src = imgUrl;
    previewFilename.textContent = sample.name;
    previewFilesize.textContent = 'Demo Sample';
    
    dropzoneIdle.classList.add('hidden');
    dropzonePreview.classList.remove('hidden');

    runInference();
  }

  function clearFileInput() {
    state.selectedFile = null;
    fileInput.value = '';
  }

  function clearSampleSelection() {
    state.selectedSampleId = null;
    document.querySelectorAll('.sample-item').forEach(el => el.classList.remove('selected'));
  }

  function resetInputState() {
    clearFileInput();
    clearSampleSelection();
    dropzoneIdle.classList.remove('hidden');
    dropzonePreview.classList.add('hidden');
    btnAnalyze.disabled = true;
  }

  function resetAll() {
    resetInputState();
    state.currentResult = null;
    resetAnalytics();
    showState('idle');
  }

  async function runInference() {
    if (!state.selectedFile && !state.selectedSampleId) return;

    showState('loading');
    btnAnalyze.disabled = true;
    btnSpinner.classList.remove('hidden');

    const formData = new FormData();
    formData.append('conf_threshold', state.confThreshold);

    if (state.selectedFile) {
      formData.append('file', state.selectedFile);
    } else if (state.selectedSampleId) {
      formData.append('sample_id', state.selectedSampleId);
    }

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(result.error || 'Inference failure');
      }

      state.currentResult = result;
      webcamVideo.classList.add('hidden');
      rpiImage.classList.add('hidden');
      canvasContainer.classList.remove('overlay-mode');

      renderDetections();
    } catch (e) {
      showError('Inference Failed', e.message);
    } finally {
      btnSpinner.classList.add('hidden');
      btnAnalyze.disabled = false;
    }
  }

  // =========================================================================
  // UNIFIED DETECTION RENDERING & CANVAS OVERLAY
  // =========================================================================
  function renderDetections() {
    const res = state.currentResult;
    if (!res || !res.success) return;

    const effectiveConf = (state.activeSourceMode === 'webcam') ? WEBCAM_CONF : state.confThreshold;
    const filteredDetections = (res.detections || []).filter(d => d.confidence >= effectiveConf);

    const counts = { bottle: 0, can: 0, carton: 0, paper: 0, plastic: 0 };
    filteredDetections.forEach(d => {
      if (counts.hasOwnProperty(d.class)) {
        counts[d.class]++;
      }
    });

    updateCategoryCounts(counts);
    detectionCountBadge.textContent = `${filteredDetections.length} Objects`;

    const lat = res.latency_ms || res.latency;
    if (lat) {
      statLatency.textContent = `${lat.total} ms`;
      statFps.textContent = `${lat.fps} FPS`;
      
      const devName = state.status && state.status.cuda_available ? 'CUDA' : 'CPU';
      const modName = res.model || 'yolov8n';
      deviceInfoText.textContent = `${modName} • ${devName} • ${lat.total} ms • ${lat.fps} FPS`;
    }

    annotatedImage.src = res.image || res.annotated_image || '';

    // Show canvas workspace
    showState('canvas');

    if (state.activeSourceMode === 'webcam') {
      drawStreamOverlay(filteredDetections, webcamVideo.videoWidth || 1280, webcamVideo.videoHeight || 720);
    } else if (state.activeSourceMode === 'rpi') {
      const dim = res.image_dimensions || { width: rpiImage.naturalWidth || 640, height: rpiImage.naturalHeight || 480 };
      drawStreamOverlay(filteredDetections, dim.width, dim.height);
    } else {
      drawInteractiveCanvas(filteredDetections, res.image_dimensions || {});
    }
  }

  function alignCanvasWithElement(targetElement, canvas, nativeWidth, nativeHeight) {
    if (!targetElement || targetElement.classList.contains('hidden')) return;

    const offsetLeft = targetElement.offsetLeft;
    const offsetTop = targetElement.offsetTop;
    const clientWidth = targetElement.clientWidth;
    const clientHeight = targetElement.clientHeight;

    if (clientWidth && clientHeight) {
      canvas.style.left = `${offsetLeft}px`;
      canvas.style.top = `${offsetTop}px`;
      canvas.style.width = `${clientWidth}px`;
      canvas.style.height = `${clientHeight}px`;
    }

    if (nativeWidth && nativeHeight) {
      canvas.width = nativeWidth;
      canvas.height = nativeHeight;
    } else if (clientWidth && clientHeight) {
      canvas.width = clientWidth;
      canvas.height = clientHeight;
    }
  }

  // Canvas drawing for live streams (Webcam & Pi) over video/image layer
  function drawStreamOverlay(detections, width, height) {
    const canvas = detectionCanvas;
    const ctx = canvas.getContext('2d');

    const targetElement = state.activeSourceMode === 'webcam' ? webcamVideo : rpiImage;
    alignCanvasWithElement(targetElement, canvas, width, height);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // If 0 detections, clear previous box immediately
    if (!detections || detections.length === 0) {
      return;
    }

    let isMirrored = false;
    if (state.activeSourceMode === 'webcam' && webcamVideo) {
      const transformStyle = window.getComputedStyle(webcamVideo).transform;
      isMirrored = transformStyle.includes('-1') || webcamVideo.classList.contains('mirrored');
    }

    detections.forEach(d => {
      let [x1, y1, x2, y2] = d.bbox;

      if (isMirrored) {
        const origX1 = x1;
        x1 = width - x2;
        x2 = width - origX1;
      }

      const bw = x2 - x1;
      const bh = y2 - y1;
      const color = d.color || getCategoryColor(d.class);

      // Bounding box stroke
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;
      ctx.strokeRect(x1, y1, bw, bh);

      // Transparent fill
      ctx.fillStyle = hexToRgba(color, 0.15);
      ctx.fillRect(x1, y1, bw, bh);

      // Label background & text
      ctx.shadowBlur = 0;
      const labelText = `${d.class.toUpperCase()} ${Math.round(d.confidence * 100)}%`;
      ctx.font = '600 13px "JetBrains Mono", monospace';
      const textWidth = ctx.measureText(labelText).width;

      const labelY = y1 - 24 > 0 ? y1 - 24 : y1;
      ctx.fillStyle = color;
      ctx.fillRect(x1, labelY, textWidth + 12, 22);

      ctx.fillStyle = '#040914';
      ctx.fillText(labelText, x1 + 6, labelY + 15);
    });
  }

  // Canvas drawing for static uploaded/demo image
  function drawInteractiveCanvas(detections, dimensions) {
    const img = new Image();
    img.src = annotatedImage.src;
    img.onload = () => {
      const canvas = detectionCanvas;
      const ctx = canvas.getContext('2d');

      const width = dimensions.width || img.naturalWidth || 640;
      const height = dimensions.height || img.naturalHeight || 360;

      canvas.width = width;
      canvas.height = height;

      ctx.drawImage(img, 0, 0, width, height);

      detections.forEach(d => {
        const [x1, y1, x2, y2] = d.bbox;
        const bw = x2 - x1;
        const bh = y2 - y1;
        const color = d.color || getCategoryColor(d.class);

        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.shadowColor = color;
        ctx.shadowBlur = 8;
        ctx.strokeRect(x1, y1, bw, bh);

        ctx.fillStyle = hexToRgba(color, 0.15);
        ctx.fillRect(x1, y1, bw, bh);

        ctx.shadowBlur = 0;
        const labelText = `${d.class.toUpperCase()} ${Math.round(d.confidence * 100)}%`;
        ctx.font = '600 12px "JetBrains Mono", monospace';
        const textWidth = ctx.measureText(labelText).width;

        ctx.fillStyle = color;
        ctx.fillRect(x1, y1 - 22 > 0 ? y1 - 22 : y1, textWidth + 10, 20);

        ctx.fillStyle = '#040914';
        ctx.fillText(labelText, x1 + 5, y1 - 22 > 0 ? y1 - 7 : y1 + 14);
      });
    };
  }

  function getCategoryColor(cls) {
    const colors = {
      bottle: '#00f5d4',
      can: '#00bbf9',
      carton: '#f15bb5',
      paper: '#fee440',
      plastic: '#ff0054'
    };
    return colors[cls] || '#00f5d4';
  }

  function updateCategoryCounts(counts) {
    countBottle.textContent = counts.bottle || 0;
    countCan.textContent = counts.can || 0;
    countCarton.textContent = counts.carton || 0;
    countPaper.textContent = counts.paper || 0;
    countPlastic.textContent = counts.plastic || 0;
  }

  function resetAnalytics() {
    updateCategoryCounts({ bottle: 0, can: 0, carton: 0, paper: 0, plastic: 0 });
    detectionCountBadge.textContent = '0 Objects';
    statLatency.textContent = '-- ms';
    statFps.textContent = '-- FPS';
  }

  function showState(stateName) {
    [stateIdle, stateLoading, stateClean, stateError, canvasWrapper].forEach(el => el.classList.add('hidden'));

    if (stateName === 'idle') stateIdle.classList.remove('hidden');
    else if (stateName === 'loading') stateLoading.classList.remove('hidden');
    else if (stateName === 'clean') stateClean.classList.remove('hidden');
    else if (stateName === 'error') stateError.classList.remove('hidden');
    else if (stateName === 'canvas') canvasWrapper.classList.remove('hidden');
  }

  function showError(title, message) {
    document.getElementById('error-title').textContent = title;
    document.getElementById('error-message').textContent = message;
    showState('error');
  }

  function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  function hexToRgba(hex, alpha) {
    let c = hex.replace('#', '');
    if (c.length === 3) c = c.split('').map(x => x + x).join('');
    const num = parseInt(c, 16);
    return `rgba(${(num >> 16) & 255}, ${(num >> 8) & 255}, ${num & 255}, ${alpha})`;
  }

});
