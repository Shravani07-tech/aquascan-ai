const sleep = ms => new Promise(r => setTimeout(r, ms));
const $ = id => document.getElementById(id);
const timers = () => [...window.__aq.intervals.values()].filter(t => t.name === 'captureAndPredictWebcamFrame').length;
$('btn-start-webcam').click(); await sleep(4000);
const tr = $('webcam-video').srcObject.getVideoTracks()[0];
const before = { status: $('webcam-status-text').textContent, readyState: tr.readyState, timers: timers() };
tr.stop(); // simulate the OS killing the camera (readyState -> 'ended')
await sleep(1500);
const after = { trackReadyState: tr.readyState, status: $('webcam-status-text').textContent, videoSrcNull: !$('webcam-video').srcObject, timers: timers(), startButtonVisible: !$('btn-start-webcam').classList.contains('hidden') };
const n = window.__aq.predicts.length; await sleep(1500);
after.noFramesAfterEnded = window.__aq.predicts.length === n;
$('btn-start-webcam').click(); await sleep(4000);
const restarted = { status: $('webcam-status-text').textContent, readyState: $('webcam-video').srcObject?.getVideoTracks()[0].readyState, timers: timers(), newFrames200: window.__aq.predicts.slice(n).filter(p => p.status === 200).length };
$('btn-stop-webcam').click();
return { before, after, restarted, errors: window.__aq.errors };
