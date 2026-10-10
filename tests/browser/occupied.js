// Camera is held by OpenCV when this starts; released ~12s after the holder began.
const sleep = ms => new Promise(r => setTimeout(r, ms));
const $ = id => document.getElementById(id);
const out = {};
$('btn-start-webcam').click();
await sleep(3000);
out.whileHeld = {
  status: $('webcam-status-text').textContent,
  errorVisible: !$('state-error').classList.contains('hidden'),
  errorTitle: $('error-title').textContent, errorMessage: $('error-message').textContent,
  timers: [...window.__aq.intervals.values()].filter(t => t.name === 'captureAndPredictWebcamFrame').length,
};
// wait until holder releases the camera
await sleep(14000);
const before = window.__aq.predicts.length;
$('btn-error-retry').click();
await sleep(5000);
const v = $('webcam-video');
out.afterTryAgain = {
  status: $('webcam-status-text').textContent,
  errorVisible: !$('state-error').classList.contains('hidden'),
  playing: !!v.srcObject && !v.paused, dims: `${v.videoWidth}x${v.videoHeight}`,
  frames: window.__aq.predicts.length - before,
  http200: window.__aq.predicts.slice(before).filter(p => p.status === 200).length,
  timers: [...window.__aq.intervals.values()].filter(t => t.name === 'captureAndPredictWebcamFrame').length,
};
$('btn-stop-webcam').click();
out.errors = window.__aq.errors;
return out;
