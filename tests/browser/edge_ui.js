// Regression test #19: Raspberry Pi tab in edge mode (no camera needed).
// Posts edge frames (frame + Pi detections) to /api/edge/ingest from the page, then checks the dashboard:
// "Edge inference (on Pi)" label with the Pi's fps/ms/temp, boxes drawn from the Pi's detections, the stale
// indicator ("OFFLINE — last frame N s ago") after > 3 s without frames, and "Laptop inference" for a Plan A frame.
const sleep = ms => new Promise(r => setTimeout(r, ms));
const $ = id => document.getElementById(id);
const frame = await (await fetch('/api/samples/sample_1/image')).blob();   // a dataset image (no people)
const W = 640, H = 360;                                                      // sample_1 size
const BOX = [200, 100, 360, 220];                                            // synthetic Pi detection
const postEdge = (frameId) => {
  const fd = new FormData();
  fd.append('file', frame, 'frame.jpg');
  fd.append('meta', JSON.stringify({ device_id: 'pi-ui-test', camera_id: 'cam-ui', frame_id: frameId, captured_at: Date.now() / 1000,
    imgsz: 416, conf: 0.25, inference_ms: 106.4, fps: 9.04, cpu_temp_c: 61.3,
    detections: [{ class: 'bottle', confidence: 0.91, bbox: BOX }] }));
  return fetch('/api/edge/ingest', { method: 'POST', body: fd }).then(r => r.status);
};
const out = {};
out.ingestStatus = await postEdge(1);
document.querySelector('.source-btn[data-source="rpi"]').click();
// keep frames fresh while checking the live view
let fid = 2; const feeder = setInterval(() => postEdge(fid++), 500);
await sleep(2500);
const c = $('detection-canvas'), ctx = c.getContext('2d');
const midY = Math.round((BOX[1] + BOX[3]) / 2);
let edgeHit = false;
const img = ctx.getImageData(0, 0, c.width, c.height).data;
for (let y = midY - 5; y <= midY + 5 && !edgeHit; y++) for (let x = BOX[2] - 4; x <= BOX[2] + 4; x++) {
  const k = 4 * (y * c.width + x); if (img[k + 3] > 120 && img[k] < 90 && img[k + 1] > 180 && img[k + 2] > 150) { edgeHit = true; break; }
}
const farEmpty = ![...ctx.getImageData(560, 300, 20, 20).data].some(v => v);
out.live = {
  status: $('rpi-val-status').textContent, inference: $('rpi-val-inference').textContent, device: $('rpi-val-device').textContent,
  badge: $('detection-count-badge').textContent, statLatency: $('stat-latency').textContent, statFps: $('stat-fps').textContent,
  canvas: `${c.width}x${c.height}`, boxPixelsAtBboxEdge: edgeHit, emptyAwayFromBox: farEmpty,
};
clearInterval(feeder);
await sleep(4500);                                   // > 3 s with no new edge frame
out.stale = { status: $('rpi-val-status').textContent, className: $('rpi-val-status').className };
// Plan A frame (laptop inference via /api/predict) must be labelled "Laptop inference"
const fd = new FormData();
fd.append('file', frame, 'frame.jpg'); fd.append('source', 'raspberry_pi'); fd.append('device_id', 'aquascan-pi5');
fd.append('camera_id', 'ov5647'); fd.append('frame_id', '1'); fd.append('conf_threshold', '0.25');
out.planAStatus = (await fetch('/api/predict', { method: 'POST', body: fd })).status;
await sleep(1500);
out.planA = { status: $('rpi-val-status').textContent, inference: $('rpi-val-inference').textContent };
document.querySelector('.source-btn[data-source="webcam"]').click();
out.PASS = out.ingestStatus === 200
  && out.live.status === 'ONLINE' && out.live.inference === 'Edge inference (on Pi) · 9.0 FPS · 106 ms · 61.3 °C'
  && out.live.device === 'pi-ui-test' && out.live.badge === '1 Objects' && out.live.statLatency === '106.4 ms' && out.live.statFps === '9.04 FPS'
  && out.live.canvas === `${W}x${H}` && out.live.boxPixelsAtBboxEdge && out.live.emptyAwayFromBox
  && /^OFFLINE — last frame \d+ s ago$/.test(out.stale.status) && out.stale.className.includes('offline')
  && out.planAStatus === 200 && out.planA.status === 'ONLINE' && out.planA.inference === 'Laptop inference';
out.errors = window.__aq ? window.__aq.errors : [];
return out;
