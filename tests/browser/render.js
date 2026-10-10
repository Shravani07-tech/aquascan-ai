// Start camera via UI, let frames flow, then verify rendered box pixels on the overlay canvas
const sleep = ms => new Promise(r => setTimeout(r, ms));
const $ = id => document.getElementById(id);
$('btn-start-webcam').click();
await sleep(9000);
const c = $('detection-canvas'), v = $('webcam-video');
const ctx = c.getContext('2d');
const last = window.__aq.predicts.slice(-1)[0];
const box = last && last.bbox && last.bbox[0];
// sample the canvas along the expected box edge (left edge x≈x1, mid-height) and far away (should be empty)
let edgeHit = null, farEmpty = null;
if (box) {
  const [x1, y1, x2, y2] = box; const midY = Math.round((y1 + y2) / 2);
  const strip = ctx.getImageData(Math.max(0, Math.round(x2) - 3), midY, 6, 1).data; // right edge
  edgeHit = [...strip].some((x, i) => i % 4 === 3 && x > 0);
  farEmpty = ![...ctx.getImageData(500, 50, 10, 10).data].some(x => x);
}
const vr = v.getBoundingClientRect(), cr = c.getBoundingClientRect();
return {
  frames: window.__aq.predicts.length, http200: window.__aq.predicts.filter(p => p.status === 200).length,
  lastDetections: last && last.dets, lastBbox: box,
  bottleFramesAllPredicts: window.__aq.predicts.filter(p => (p.dets || []).some(d => d.startsWith('bottle'))).length,
  badge: $('detection-count-badge').textContent, bottleCount: $('count-bottle').textContent,
  canvasHasBoxPixelsAtBboxEdge: edgeHit, canvasEmptyAwayFromBox: farEmpty,
  canvasAlign: { dx: cr.left - vr.left, dy: cr.top - vr.top, dw: cr.width - vr.width, dh: cr.height - vr.height },
  mirrored: getComputedStyle(v).transform,
  webcamTimers: [...window.__aq.intervals.values()].filter(t => t.name === 'captureAndPredictWebcamFrame').length,
  errors: window.__aq.errors,
};
