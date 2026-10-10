// Instrumentation installed before app.js runs (observes only; does not alter app behavior).
window.__aq = { predicts: [], intervals: new Map(), errors: [] };
const _si = window.setInterval, _ci = window.clearInterval;
window.setInterval = function (fn, ms, ...a) { const id = _si.call(window, fn, ms, ...a); window.__aq.intervals.set(id, { name: fn && fn.name, ms }); return id; };
window.clearInterval = function (id) { window.__aq.intervals.delete(id); return _ci.call(window, id); };
const _f = window.fetch;
window.fetch = async function (u, o) {
  const t0 = performance.now(); const r = await _f.call(window, u, o);
  if (String(u).includes('/api/predict')) {
    r.clone().json().then(j => window.__aq.predicts.push({
      status: r.status, ms: Math.round(performance.now() - t0), count: j.count,
      dets: (j.detections || []).map(d => `${d.class_name || d.label || d.class}:${(d.confidence * 100).toFixed(1)}`),
      bbox: (j.detections || []).map(d => d.bbox)
    })).catch(() => window.__aq.predicts.push({ status: r.status, parseFail: true }));
  }
  return r;
};
window.addEventListener('error', e => window.__aq.errors.push(String(e.message)));
window.addEventListener('unhandledrejection', e => window.__aq.errors.push('unhandled: ' + String(e.reason)));
