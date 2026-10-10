# AquaScan AI — Regression Tests

One exact command per result listed in the handoff §9 ("Tests Already Passed").
All commands run from the project root in **PowerShell**, using the project venv.

## Prerequisites

| Requirement | Applies to |
|---|---|
| Backend running: `.venv\Scripts\python.exe run_server.py` | every test except #0, #6 |
| Node.js ≥ 22 (built-in WebSocket) and Google Chrome installed | browser tests #7–#14 |
| **Webcam free**: no `rpi_client.py`, Teams/Zoom/Camera app, or Chrome tab using it | #7–#11 (real webcam) |
| **Webcam held by nothing else at the end of each test** | never run `rpi_client.py --backend opencv` while a browser test owns the camera |

Browser tests launch a separate headless Chrome (temporary profile) driven over the
DevTools protocol by `tests/browser/cdp.mjs`. `--use-fake-ui-for-media-stream` auto-accepts
the camera prompt; the camera itself is the **real** laptop webcam unless `FAKECAM` is set.
`PRELOAD=tests/browser/preload.js` installs observe-only instrumentation (counts
`/api/predict` responses and inference timers); it does not change app behaviour.

## Tests

| # | §9 test | Expected result | Command |
|---|---|---|---|
| 0 | JS syntax | no output, exit 0 | `node -c frontend/js/app.js` |
| 1 | API suite | `ALL 13 API TEST SUITE ASSERTIONS PASSED` | `.venv\Scripts\python.exe scripts\test_api.py` |
| 2 | Raspberry Pi streaming (simulated; **opens the real webcam via OpenCV** — camera must be free) | `10 / 10`, `PASSED 100%` | `.venv\Scripts\python.exe scripts\test_rpi_streaming.py` |
| 3 | Hardware simulation | `SIMULATION SUCCESSFUL` | `.venv\Scripts\python.exe scripts\test_hardware_simulation.py` |
| 4 | Real detection | exit 0, `HONEST EVALUATION SUMMARY REPORT` | `.venv\Scripts\python.exe scripts\test_real_detection.py` |
| 5 | Laptop E2E detection | `completed successfully` | `.venv\Scripts\python.exe scripts\verify_laptop_e2e_detection.py` |
| 6 | Dataset validation | `100% of labels are valid` (7644 files, 21111 boxes) | `.venv\Scripts\python.exe scripts\verify_clean_dataset.py` |
| 7 | Stop→Start cycles (backend) | `ALL 5 START/STOP RESTART CYCLES PASSED` (25/25 HTTP 200) | `.venv\Scripts\python.exe scripts\test_webcam_restart_cycles.py` |
| 8 | Real browser frame flow + 5 browser Stop→Start cycles (**real webcam**) | `total.http200 == total.n`; every cycle `timers 1`, `errorShown null`; `finalStop.webcamTimers 0`; `pageErrors []` | `$env:PRELOAD='tests/browser/preload.js'; node tests/browser/cdp.mjs http://localhost:8000/ tests/browser/flow.js` |
| 8b | Reload mid-stream, then flow (**real webcam**) | as #8, plus `PRE_SCRIPT` shows `liveBeforeReload: true` | `$env:PRELOAD='tests/browser/preload.js'; $env:PRE_SCRIPT='tests/browser/pre_live.js'; node tests/browser/cdp.mjs http://localhost:8000/ tests/browser/flow.js` |
| 9 | Controlled browser rendering (**synthetic camera**) | `http200 == frames`, `bottleFramesAllPredicts == frames`, `canvasHasBoxPixelsAtBboxEdge: true`, `canvasEmptyAwayFromBox: true`, `canvasAlign` all 0 | `.venv\Scripts\python.exe tests\make_fake_camera.py; $env:PRELOAD='tests/browser/preload.js'; $env:FAKECAM='C:\Users\Public\aquascan_fakecam.mjpeg'; node tests/browser/cdp.mjs http://localhost:8000/ tests/browser/render.js` |
| 10 | Camera busy / Try Again (**real webcam**) | `whileHeld.errorTitle = "Camera Occupied / Blocked"`, `timers 0`; `afterTryAgain.playing true`, `http200 == frames`, `timers 1`. The single console line `CRITICAL STARTUP ERROR: NotReadableError` is the expected, intentional log. | `Start-Process -NoNewWindow .venv\Scripts\python.exe -ArgumentList 'tests\hold_camera.py','12'; Start-Sleep 4; $env:PRELOAD='tests/browser/preload.js'; node tests/browser/cdp.mjs http://localhost:8000/ tests/browser/occupied.js` |
| 11 | Ended-track recovery (**real webcam**) | `after.trackReadyState ended`, `after.timers 0`, `after.noFramesAfterEnded true`; `restarted.readyState live`, `restarted.timers 1` | `$env:PRELOAD='tests/browser/preload.js'; node tests/browser/cdp.mjs http://localhost:8000/ tests/browser/ended.js` |
| 12 | Backend webcam soak | `300/300 HTTP 200`, `bad JSON: 0` | `.venv\Scripts\python.exe tests\soak_webcam.py scratch_frame.jpg 300` |
| 13 | Upload/Demo + Raspberry Pi UI (Pi **simulated** by `rpi_client.py --backend file`) | `PASS: true`; `rpi.status ONLINE`; every sample `finished`, `badgeMatchesApi: true` (badge read after its `/api/predict` completes), no `error` | `Start-Process -NoNewWindow .venv\Scripts\python.exe -ArgumentList 'hardware\rpi_client.py','--server','http://127.0.0.1:8000','--backend','file','--image','test.jpg','--loop','--max-frames','40','--interval','0.5'; Start-Sleep 3; node tests/browser/cdp.mjs http://localhost:8000/ tests/browser/ui_rpi_upload.js` |

| 14 | Box geometry vs displayed picture (**synthetic camera**), any window size / display scale | `PASS: true`, `maxBoxErrorPx` ≤ 1.5, `letterboxPx` [0,0] | `.venv\Scripts\python.exe tests\make_fake_camera.py; $env:PRELOAD='tests/browser/preload.js'; $env:FAKECAM='C:\Users\Public\aquascan_fakecam.mjpeg'; $env:VIEWPORT='1275,800,1.5'; node tests/browser/cdp.mjs http://localhost:8000/ tests/browser/geometry.js` (repeat with `VIEWPORT` e.g. `1920,1080,1`, `700,800,1`, `400,800,2`) |
| 15 | Mirrored video: boxes flipped with the picture (**synthetic camera**) | `PASS: true`; `overlayBottleXRange` ≈ `mirroredExpected` (±6 px); `drawnAtUnflippedPosition: false` | `$env:PRELOAD='tests/browser/preload.js'; $env:FAKECAM='C:\Users\Public\aquascan_fakecam.mjpeg'; $env:VIEWPORT='1275,800,1.5'; node tests/browser/cdp.mjs http://localhost:8000/ tests/browser/mirror.js` |
| 16 | Static-object box position, **real webcam** (diagnostic; needs an object in view) | `WORST EDGE OFFSET` ≤ ~4 px | `$env:PRELOAD='tests/browser/preload_static.js'; $env:VIEWPORT='1275,800,1.5'; $env:SHOT='static.png'; node tests/browser/cdp.mjs http://localhost:8000/ tests/browser/static.js > static.txt; .venv\Scripts\python.exe tests\browser\compare_static.py static.txt static.png static_compare.jpg` (output images may contain people: do not commit) |

| 17 | Confidence slider: fixed 20% in webcam mode, honoured in Upload/Demo (**synthetic camera**) | `PASS: true`; webcam slider disabled, label `20% (fixed in webcam mode)`, `webcamRequestsConf` ["0.2"] even after setting 10%; `uploadRequestConf` ["0.3"] at 30% | `$env:PRELOAD='tests/browser/preload.js'; $env:FAKECAM='C:\Users\Public\aquascan_fakecam.mjpeg'; node tests/browser/cdp.mjs http://localhost:8000/ tests/browser/conf_cap.js` |

Clear env vars between browser tests: `Remove-Item Env:PRELOAD, Env:PRE_SCRIPT, Env:FAKECAM, Env:VIEWPORT, Env:SHOT -ErrorAction SilentlyContinue`

## Notes

- **Raspberry Pi**: #2, #3 and #13 simulate the Pi with the laptop (`rpi_client.py` with
  `--backend file`, or OpenCV on the laptop webcam). No real Raspberry Pi hardware has been tested.
- **`scratch_frame.jpg`** (replaced 2026-10-07) is an in-domain, person-free 640×480 test frame:
  dataset test image `frame_000041` (Roboflow floating-waste v10, CC BY 4.0), cropped to drop the
  pool-deck strip and resized to webcam size. Expected at webcam settings (imgsz=416, conf 0.20):
  bottle ≈92% at ≈[411, 141, 554, 243], plastic ≈82%. The previous frame (a real webcam frame
  containing a person; the imgsz=416 evidence of handoff §5) is kept locally, untracked, at
  `scratch/private/webcam_reference_frame_with_person.jpg`; it remains in commit 36bbe9d history.
- **#7**, `scripts/evaluate_models.py` and `scratch/inspect_webcam_adaptation_status.py` always use
  `scratch_frame.jpg` (expected: BOTTLE ≈92% at webcam settings). They no longer read the rolling
  `scratch\webcam_debug.jpg`, which any webcam-mode request overwrites.
- `scripts/test_20_frame_webcam_stability.py` captures the **live webcam** (falls back to
  `scratch_frame.jpg` only if the camera is unavailable); its detections depend on the scene.
- **#9** uses a synthetic camera and verifies software rendering/alignment only. It is not
  evidence that a physical bottle is detected.
- **#12** previously ran against a real browser frame captured on 2026-10-07; the reproducible
  command uses the committed test frame `scratch_frame.jpg`.
- Python clients should use `http://127.0.0.1:8000`: on this machine `localhost` adds ~2 s
  per request from Python (IPv6 tried first). Browsers are unaffected.
- `tests/browser/probe.js` is a diagnostic (camera enumeration / constraints), not a pass/fail test:
  `node tests/browser/cdp.mjs http://localhost:8000/ tests/browser/probe.js`

## Frame collector (debug only, handoff v2 §14)

Off by default. Start the backend with it enabled (PowerShell):

```
$env:AQUASCAN_SAVE_FRAMES='1'; $env:AQUASCAN_PHASE='p0a'; .venv\Scripts\python.exe run_server.py
```

Each `/api/predict` request then writes `debug_frames/<phase>/<seq>_<timestamp>.jpg` (the exact bytes
received; decoding them reproduces the model input bit-for-bit) and a `.json` with: `path` (A = browser
`laptop_webcam`, B = `raspberry_pi` i.e. `rpi_client.py`, `upload`), `source`, `client_frame_seq`,
`imgsz` and `conf_effective` (after the 0.20 webcam cap), `conf_requested`, `shape`, `dtype`,
`image_sha256`, the deployed detections, and `raw_boxes` down to conf 0.01 (separate debug inference at
the same imgsz). Phase name, first match wins: `?phase=` on `/api/predict`; `?phase=` in the page URL
(open `http://localhost:8000/?phase=p2` — Chrome sends it as the Referer); `AQUASCAN_PHASE`; `unsorted`.
Path B (`rpi_client.py`, unmodified) is tagged via `AQUASCAN_PHASE`. With the collector on, each frame
runs inference twice, so frame rate roughly halves. `debug_frames/` is git-ignored (frames may contain
people). Never present with `AQUASCAN_SAVE_FRAMES` set. When enabled, the server window prints a
`FRAME COLLECTOR ON -> ...debug_frames` banner at startup. After a collection session, **close that
PowerShell window** (or run `Remove-Item Env:AQUASCAN_SAVE_FRAMES, Env:AQUASCAN_PHASE`) so the variables
cannot carry over into the presentation server.

## Results log

| Date | Commit | # | Result | Evidence |
|---|---|---|---|---|
| 2026-10-07 22:44 | b41e27c | 0 | PASS | `node -c` exit 0 |
| 2026-10-07 22:44 | b41e27c | 1 | PASS | 13/13 API assertions |
| 2026-10-07 22:45 | b41e27c | 2 | PASS | 10/10 frames HTTP 200 (opened real webcam via OpenCV) |
| 2026-10-07 22:45 | b41e27c | 3 | PASS | simulation successful |
| 2026-10-07 22:46 | b41e27c | 4 | PASS | exit 0; REST pipeline 100% HTTP 200 |
| 2026-10-07 22:47 | b41e27c | 5 | PASS | E2E suite completed |
| 2026-10-07 22:47 | b41e27c | 6 | PASS | 7644 label files, 21111 boxes, 100% valid |
| 2026-10-07 22:48 | b41e27c | 7 | PASS | 5/5 cycles, 25/25 HTTP 200 |
| 2026-10-07 22:49 | b41e27c | 8b | FLAKE (harness) | returned `{}` with exit 0: driver slept 2.5 s instead of waiting for page load. Fixed in 87426d3 |
| 2026-10-07 22:52 | b41e27c | 12 | PASS | 300/300 HTTP 200, 0 bad JSON, p50 115 ms, p95 157 ms |
| 2026-10-07 23:05 | 87426d3 | 8 | PASS | 24/24 HTTP 200, 5/5 cycles, 1 timer, 0 console errors |
| 2026-10-07 23:05 | 87426d3 | 8b | PASS ×3 | 55/55, 42/42, 55/55 HTTP 200; liveBeforeReload true; 5/5 cycles each |
| 2026-10-07 23:05 | 87426d3 | 9 | PASS | 23/23 frames BOTTLE 92.6%, box on bbox edge, alignment 0/0/0/0 (synthetic camera) |
| 2026-10-07 23:05 | 87426d3 | 10 | PASS | "Camera Occupied / Blocked" while held; Try Again → 10/10 HTTP 200, 1 timer |
| 2026-10-07 23:05 | 87426d3 | 11 | PASS | ended track → Standby, 0 timers, no frames; restart live, 7 HTTP 200 |
| 2026-10-07 23:05 | 87426d3 | 13 | PASS | Pi (simulated) ONLINE #0011; Upload 6/6 samples, no errors |
| 2026-10-07 23:28 | d927712 | 0–7 | PASS | API 13/13; Pi streaming 10/10 (sim); hw sim; real detection; laptop E2E; dataset 100% valid; restart cycles 25/25 now on fixed frame (BOTTLE 92.4%) |
| 2026-10-07 23:35 | d927712 | 8 | PASS | 8/8 HTTP 200, 5/5 cycles (only 8 frames: on battery at 13%, CPU throttled — see Phase 2 latency analysis) |
| 2026-10-07 23:35 | d927712 | 8b | PASS | 36/36 HTTP 200, 5/5 cycles |
| 2026-10-07 23:35 | d927712 | 9 | PASS | 8/8 frames BOTTLE 92.6%, alignment exact (synthetic camera) |
| 2026-10-07 23:35 | d927712 | 10, 11 | PASS | Try Again 9/9 HTTP 200; ended-track recovery |
| 2026-10-07 23:45 | d927712 | 12 | PASS | 300/300 HTTP 200, p50 107 ms |
| 2026-10-07 23:45 | d927712 | 13 | PASS | Pi ONLINE (sim); Upload 6/6. Printed count for sample 4 was stale (3 vs actual 4, verified via API and baseline code): the script reads the badge after a fixed 2.5 s |
| 2026-10-07 23:45 | d927712 | extra | PASS | fixed scripts run as written (fastapi client HTTP 200; evaluate_models / inspect on scratch_frame BOTTLE 94.6%); API suite 13/13 with AQUASCAN_SAVE_FRAMES=1 |
| 2026-10-10 12:25 | 09b3fc2 | 0–13 | PASS | API 13/13; Pi sim 10/10; hw sim; real detection; E2E; dataset; restart 25/25; flow 58/58 + 62/62 (avg 89/98 ms, AC); render 28/28; Try Again 12/12; ended-track; soak 300/300; Upload 6/6 counts match API |
| 2026-10-10 12:25 | 09b3fc2 | 14, 15, 17 | PASS | geometry ≤0.3 px (4 viewports); mirror flipped correctly (2 viewports); webcam conf fixed 0.2, Upload sends slider 0.3 |
| 2026-10-10 12:25 | 09b3fc2 | hardware | PASS | real Pi Plan A 30/30 HTTP 200; Plan B on Pi bottle 92.4% / plastic 81.5%; collector off (887 debug frames before = after) |
