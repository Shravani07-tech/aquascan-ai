// Probe: enumerate cameras, then try the app's exact constraints and each device individually.
const out = {};
const tryGum = async (c) => {
  try {
    const s = await navigator.mediaDevices.getUserMedia(c);
    const t = s.getVideoTracks()[0];
    const r = { ok: true, label: t.label, settings: t.getSettings() };
    s.getTracks().forEach(x => x.stop());
    await new Promise(r => setTimeout(r, 400));
    return r;
  } catch (e) { return { ok: false, err: `${e.name}: ${e.message}` }; }
};
// 1. app's exact primary constraints (what Start Camera calls)
out.appPrimary = await tryGum({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }, audio: false });
// 2. app's fallback
out.appFallback = await tryGum({ video: true, audio: false });
// 3. enumerate (labels now visible after a grant)
const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
out.devices = devs.map((d, i) => ({ index: i, label: d.label, id: d.deviceId.slice(0, 10) }));
// 4. each device explicitly
out.perDevice = [];
for (const d of devs) {
  out.perDevice.push({ label: d.label, result: await tryGum({ video: { deviceId: { exact: d.deviceId }, width: { ideal: 640 }, height: { ideal: 480 } }, audio: false }) });
}
return out;
