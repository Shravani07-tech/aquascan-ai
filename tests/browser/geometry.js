// Box-positioning geometry check (laptop webcam mode).
// Measures where a box in model/frame coordinates is DRAWN (overlay canvas) versus where that point
// actually IS in the displayed video picture (object-fit: contain inside the <video> element box).
// Reports the worst-case error in CSS pixels over the frame corners. 0 = boxes land exactly on the object.
const sleep = ms => new Promise(r => setTimeout(r, ms));
const $ = id => document.getElementById(id);
$('btn-start-webcam').click();
for (let i = 0; i < 60 && !(window.__aq && window.__aq.predicts.length >= 2); i++) await sleep(250);
await sleep(300);
const v = $('webcam-video'), c = $('detection-canvas');
const vr = v.getBoundingClientRect(), cr = c.getBoundingClientRect();
const vw = v.videoWidth, vh = v.videoHeight;
// Picture area inside the video element under object-fit: contain (content box = client box, no padding)
const cw = v.clientWidth, ch = v.clientHeight;
const fit = getComputedStyle(v).objectFit;
const s = fit === 'cover' ? Math.max(cw / vw, ch / vh) : Math.min(cw / vw, ch / vh);
const picW = vw * s, picH = vh * s, picX = vr.left + (cw - picW) / 2, picY = vr.top + (ch - picH) / 2;
// Overlay mapping: canvas backing store is (c.width x c.height) stretched over its CSS rect
const drawn = (x, y) => [cr.left + x * cr.width / c.width, cr.top + y * cr.height / c.height];
const truth = (x, y) => [picX + x * picW / vw, picY + y * picH / vh];
let maxErr = 0;
for (const [x, y] of [[0, 0], [vw, 0], [0, vh], [vw, vh], [vw / 2, vh / 2]]) {
  const [dx, dy] = drawn(x, y), [tx, ty] = truth(x, y);
  maxErr = Math.max(maxErr, Math.abs(dx - tx), Math.abs(dy - ty));
}
const out = {
  viewport: `${innerWidth}x${innerHeight}@${devicePixelRatio}`,
  camera: `${vw}x${vh}`, canvasBacking: `${c.width}x${c.height}`, objectFit: fit,
  videoBox: `${Math.round(vr.width)}x${Math.round(vr.height)}`, picture: `${Math.round(picW)}x${Math.round(picH)}`,
  letterboxPx: [Math.round((cw - picW) / 2), Math.round((ch - picH) / 2)],
  canvasRect: `${Math.round(cr.width)}x${Math.round(cr.height)}`,
  maxBoxErrorPx: +maxErr.toFixed(1),
  PASS: maxErr <= 1.5,
  frames: window.__aq.predicts.length, errors: window.__aq.errors,
};
$('btn-stop-webcam').click();
return out;
