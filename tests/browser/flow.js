// End-to-end browser flow through the real UI buttons.
// MODE=live: start and keep running (for screenshot). Otherwise: start, run, then N rapid stop/start cycles, then stop.
const sleep = ms => new Promise(r => setTimeout(r, ms));
const $ = id => document.getElementById(id);
const visible = el => el && !el.classList.contains('hidden');
const webcamTimers = () => [...window.__aq.intervals.values()].filter(t => t.name === 'captureAndPredictWebcamFrame').length;
const snap = () => {
  const v = $('webcam-video'), c = $('detection-canvas');
  const tr = v.srcObject ? v.srcObject.getVideoTracks()[0] : null;
  const vr = v.getBoundingClientRect(), cr = c.getBoundingClientRect();
  return {
    status: $('webcam-status-text').textContent,
    errorShown: visible($('state-error')) ? $('error-title').textContent : null,
    videoPlaying: !!v.srcObject && !v.paused, dims: `${v.videoWidth}x${v.videoHeight}`,
    track: tr ? `${tr.label} ${tr.readyState}` : null,
    webcamTimers: webcamTimers(),
    badge: $('detection-count-badge').textContent,
    bottleCount: $('count-bottle').textContent,
    canvasAlign: { dx: Math.round(cr.left - vr.left), dy: Math.round(cr.top - vr.top), dw: Math.round(cr.width - vr.width), dh: Math.round(cr.height - vr.height), canvasPx: `${c.width}x${c.height}` },
  };
};
const summarize = (arr) => ({
  n: arr.length, http200: arr.filter(p => p.status === 200).length,
  withDetections: arr.filter(p => p.count > 0).length,
  bottleFrames: arr.filter(p => (p.dets || []).some(d => d.startsWith('bottle'))).length,
  avgMs: Math.round(arr.reduce((a, p) => a + (p.ms || 0), 0) / Math.max(1, arr.length)),
  sample: arr.slice(-4).map(p => p.dets.join(',') || '(none)'),
});
const out = {};
$('btn-start-webcam').click();
await sleep(7000);
out.afterStart = snap();
out.afterStartFrames = summarize(window.__aq.predicts);
if (window.__MODE === 'live') return out;

out.cycles = [];
for (let i = 1; i <= Number(window.__CYCLES || 5); i++) {
  const before = window.__aq.predicts.length;
  $('btn-stop-webcam').click();
  const stopped = { videoNull: !$('webcam-video').srcObject, timers: webcamTimers(), canvasCleared: !$('detection-canvas').getContext('2d').getImageData(0, 0, $('detection-canvas').width || 1, $('detection-canvas').height || 1).data.some(x => x) };
  $('btn-start-webcam').click(); // immediate restart, no delay (worst case)
  await sleep(4000);
  out.cycles.push({ cycle: i, stopped, running: snap(), frames: summarize(window.__aq.predicts.slice(before)) });
}
// Try Again path from error state is stopWebcam()+startWebcam(); covered by immediate restart above.
$('btn-stop-webcam').click();
await sleep(500);
const v = $('webcam-video');
out.finalStop = { videoSrcNull: !v.srcObject, webcamTimers: webcamTimers(), status: $('webcam-status-text').textContent, framesAfterStop: (() => { const n = window.__aq.predicts.length; return n; })() };
await sleep(1500);
out.finalStop.noFramesAfterStop = window.__aq.predicts.length === out.finalStop.framesAfterStop;
out.total = summarize(window.__aq.predicts);
out.pageErrors = window.__aq.errors;
return out;
