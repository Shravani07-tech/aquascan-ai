// Raspberry Pi panel + Upload/Demo check through the real UI (no camera needed).
// Run while hardware\rpi_client.py --backend file is streaming, so the Pi panel has live frames.
const sleep = ms => new Promise(r => setTimeout(r, ms));
const $ = id => document.getElementById(id);
await sleep(1500);
document.querySelector('.source-btn[data-source="rpi"]').click(); await sleep(3000);
const rpi = { status: $('rpi-val-status').textContent, device: $('rpi-val-device').textContent, frame: $('rpi-val-frame').textContent, imgShown: !$('rpi-image').classList.contains('hidden') };
document.querySelector('.source-btn[data-source="upload"]').click(); await sleep(1500);

// Observe completed /api/predict responses so each count is read only after its analysis finishes
// (a fixed delay read stale counts when inference was slow).
const done = [];
const _fetch = window.fetch;
window.fetch = async (u, o) => {
  const r = await _fetch(u, o);
  if (String(u).includes('/api/predict')) r.clone().json().then(j => done.push(j)).catch(() => done.push(null));
  return r;
};
const waitFor = async (cond, ms) => { const t = Date.now(); while (!cond() && Date.now() - t < ms) await sleep(50); return cond(); };

const samples = document.querySelectorAll('.sample-item'); const perSample = [];
for (let i = 0; i < samples.length; i++) {
  samples[i].click(); await sleep(200);
  const before = done.length;
  $('btn-analyze').click();
  const finished = await waitFor(() => done.length > before, 30000);
  await sleep(150); // let renderDetections() update the badge
  const api = finished ? done[done.length - 1] : null;
  const badge = $('detection-count-badge').textContent;
  perSample.push({
    badge, apiCount: api ? api.count : null, finished,
    badgeMatchesApi: !!api && badge === `${api.count} Objects`,
    error: !$('state-error').classList.contains('hidden'),
  });
}
window.fetch = _fetch;
document.querySelector('.source-btn[data-source="webcam"]').click();
return {
  rpi, uploadSamples: samples.length, perSample,
  PASS: rpi.status === 'ONLINE' && rpi.imgShown && samples.length > 0
    && perSample.every(s => s.finished && s.badgeMatchesApi && !s.error),
  errors: window.__aq ? window.__aq.errors : [],
};
