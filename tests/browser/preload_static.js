// Static-object test instrumentation (test only). Records the exact JPEG sent with each /api/predict
// request and its response; once window.__freeze is set, further /api/predict requests never resolve,
// so the overlay stays on the frozen result while the live video keeps playing.
window.__aq = { predicts: [], intervals: new Map(), errors: [] };
window.__st = { sent: [], freeze: false };
const _si = window.setInterval, _ci = window.clearInterval;
window.setInterval = function (fn, ms, ...a) { const id = _si.call(window, fn, ms, ...a); window.__aq.intervals.set(id, { name: fn && fn.name, ms }); return id; };
window.clearInterval = function (id) { window.__aq.intervals.delete(id); return _ci.call(window, id); };
const _f = window.fetch;
window.fetch = async function (u, o) {
  if (!String(u).includes('/api/predict')) return _f.call(window, u, o);
  if (window.__st.freeze) return new Promise(() => {});
  const blob = o && o.body && o.body.get ? o.body.get('file') : null;
  const r = await _f.call(window, u, o);
  const j = await r.clone().json().catch(() => null);
  window.__aq.predicts.push({ status: r.status, count: j && j.count });
  if (blob && j) window.__st.sent.push({ blob, response: j, at: performance.now() });
  return r;
};
window.addEventListener('error', e => window.__aq.errors.push(String(e.message)));
