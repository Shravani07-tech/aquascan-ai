// Static-object box-positioning test (real webcam). Start the camera, wait for a response with >=1
// detection that passes the UI filter, freeze further requests, and return the exact frame sent plus its
// boxes and the on-screen geometry. cdp.mjs (SHOT=...) then screenshots the page with the overlay frozen.
const sleep = ms => new Promise(r => setTimeout(r, ms));
const $ = id => document.getElementById(id);
const WAIT_MS = Number(window.__STATIC_WAIT_MS || 60000);
$('btn-start-webcam').click();
const t0 = Date.now();
let hit = null;
while (Date.now() - t0 < WAIT_MS && !hit) {
  await sleep(100);
  const eff = Math.min(Number($('conf-slider').value) / 100, 0.20); // app.js webcam filter
  hit = window.__st.sent.find(s => (s.response.detections || []).some(d => d.confidence >= eff)) || null;
}
if (!hit) { const n = window.__st.sent.length; $('btn-stop-webcam').click(); return { found: false, framesSent: n }; }
window.__st.freeze = true;           // overlay now stays on this result
await sleep(400);                    // let renderDetections() draw it
const dataUrl = await new Promise(r => { const fr = new FileReader(); fr.onload = () => r(fr.result); fr.readAsDataURL(hit.blob); });
const v = $('webcam-video'), c = $('detection-canvas');
const vr = v.getBoundingClientRect(), cr = c.getBoundingClientRect();
const eff = Math.min(Number($('conf-slider').value) / 100, 0.20);
return {
  found: true, frameDataUrl: dataUrl, dpr: devicePixelRatio,
  frameSize: [v.videoWidth, v.videoHeight],
  detections: hit.response.detections.filter(d => d.confidence >= eff).map(d => ({ class: d.class, confidence: d.confidence, bbox: d.bbox })),
  videoRect: [vr.left, vr.top, vr.width, vr.height], canvasRect: [cr.left, cr.top, cr.width, cr.height],
  badge: $('detection-count-badge').textContent, framesSent: window.__st.sent.length, errors: window.__aq.errors,
};
