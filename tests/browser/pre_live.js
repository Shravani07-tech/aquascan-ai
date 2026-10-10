// Start the camera and leave it LIVE, so the page is reloaded mid-stream.
document.getElementById('btn-start-webcam').click();
await new Promise(r => setTimeout(r, 4000));
const v = document.getElementById('webcam-video');
return { liveBeforeReload: !!v.srcObject && !v.paused, frames: window.__aq.predicts.length, http200: window.__aq.predicts.filter(p => p.status === 200).length };
