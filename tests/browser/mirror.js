// Mirrored-video regression test (synthetic camera = scratch_frame.jpg, bottle at the right of the frame).
// Mirrors the <video> with CSS (scaleX(-1)), then reads the overlay canvas pixels the app actually drew and
// checks the bottle box was flipped to match the mirrored picture. Also measures overlay-vs-picture error
// for the mirrored mapping. PASS = box drawn at the mirrored x-range (within 6 px) and nothing at the unflipped one.
const sleep = ms => new Promise(r => setTimeout(r, ms));
const $ = id => document.getElementById(id);
const v = $('webcam-video'), c = $('detection-canvas');
v.style.transform = 'scaleX(-1)';
$('btn-start-webcam').click();
let res = null;
for (let i = 0; i < 120 && !res; i++) {
  await sleep(250);
  const last = window.__aq.predicts.slice(-1)[0];
  if (last && last.dets && last.dets.some(d => d.startsWith('bottle'))) res = last;
}
if (!res) { $('btn-stop-webcam').click(); return { PASS: false, reason: 'no bottle detection', frames: window.__aq.predicts.length }; }
await sleep(400);
const vw = v.videoWidth, vh = v.videoHeight;
const bi = res.dets.findIndex(d => d.startsWith('bottle'));
const [x1, y1, x2, y2] = res.bbox[bi];
const ctx = c.getContext('2d');
const liveTransform = getComputedStyle(v).transform; // read while the video is visible (hidden => "none")
// Scan the box's vertical band of the overlay for the bottle colour (#00f5d4). The stroke has a glow
// (shadowBlur), so use a tolerant colour/alpha test over all rows of the band, not a single row.
const img = ctx.getImageData(0, 0, c.width, c.height).data;
const xs = [];
for (let y = Math.round(y1) + 4; y <= Math.round(y2) - 4; y++) {
  for (let x = 0; x < c.width; x++) {
    const k = 4 * (y * c.width + x), r = img[k], g = img[k + 1], b = img[k + 2], a = img[k + 3];
    if (a > 120 && r < 90 && g > 180 && b > 150) xs.push(x);
  }
}
const found = xs.length ? [Math.min(...xs), Math.max(...xs)] : null;
const mirroredExpected = [vw - x2, vw - x1];
const near = (p, q) => Math.abs(p - q) <= 6;
const flipped = !!found && near(found[0], mirroredExpected[0]) && near(found[1], mirroredExpected[1]);
const atUnflipped = xs.some(x => x >= x1 + 6 && x <= x2 - 6 && x > mirroredExpected[1] + 6);
// Geometry: a frame point x is displayed (mirrored) at picX + (vw - x) * picW / vw; the overlay draws the app's
// flipped coordinate (vw - x) at cr.left + (vw - x) * cr.width / c.width. Compare for the box corners.
const vr = v.getBoundingClientRect(), cr = c.getBoundingClientRect();
const s = Math.min(v.clientWidth / vw, v.clientHeight / vh), picW = vw * s, picX = vr.left + (v.clientWidth - picW) / 2;
let maxErr = 0;
for (const x of [x1, x2]) maxErr = Math.max(maxErr, Math.abs((cr.left + (vw - x) * cr.width / c.width) - (picX + (vw - x) * picW / vw)));
$('btn-stop-webcam').click();
return {
  videoTransform: liveTransform, bottleBboxFrame: [x1, y1, x2, y2].map(Math.round),
  overlayBottleXRange: found, mirroredExpected: mirroredExpected.map(Math.round), drawnAtUnflippedPosition: atUnflipped,
  mirroredGeometryErrorPx: +maxErr.toFixed(1),
  PASS: flipped && !atUnflipped && maxErr <= 1.5, errors: window.__aq.errors,
};
