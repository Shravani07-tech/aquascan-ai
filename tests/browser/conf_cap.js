// Confidence-slider behaviour (synthetic camera). Webcam mode: slider disabled, label "20% (fixed in webcam
// mode)", requests sent with conf_threshold 0.2 even if the slider was moved in another mode. Upload/Demo:
// slider enabled, label = slider value, requests sent with the slider value.
const sleep = ms => new Promise(r => setTimeout(r, ms));
const $ = id => document.getElementById(id);
const sentConf = [];
const _f = window.fetch;
window.fetch = async (u, o) => {
  if (String(u).includes('/api/predict') && o && o.body && o.body.get) sentConf.push({ source: o.body.get('source'), conf: o.body.get('conf_threshold') });
  return _f(u, o);
};
const slider = $('conf-slider'), label = $('conf-val');
const setSlider = v => { slider.value = v; slider.dispatchEvent(new Event('input', { bubbles: true })); };
const ui = () => ({ disabled: slider.disabled, label: label.textContent });
const out = {};
out.webcamOnLoad = ui();
document.querySelector('.source-btn[data-source="upload"]').click(); await sleep(300);
out.uploadMode = ui();
setSlider(10); out.uploadAfterSlider10 = ui();
document.querySelector('.source-btn[data-source="webcam"]').click(); await sleep(300);
out.webcamAfterSlider10 = ui();
$('btn-start-webcam').click();
for (let i = 0; i < 80 && !sentConf.some(s => s.source === 'laptop_webcam'); i++) await sleep(250);
await sleep(1500);
$('btn-stop-webcam').click(); await sleep(300);
out.webcamRequestsConf = [...new Set(sentConf.filter(s => s.source === 'laptop_webcam').map(s => s.conf))];
document.querySelector('.source-btn[data-source="upload"]').click(); await sleep(1500);
setSlider(30);
out.uploadAfterSlider30 = ui();
document.querySelectorAll('.sample-item')[0].click(); await sleep(800);
const before = sentConf.length; $('btn-analyze').click();
for (let i = 0; i < 80 && sentConf.length === before; i++) await sleep(250);
out.uploadRequestConf = sentConf.slice(before).map(s => s.conf);
document.querySelector('.source-btn[data-source="webcam"]').click(); await sleep(300);
out.webcamFinal = ui();
out.PASS = out.webcamOnLoad.disabled && out.webcamOnLoad.label === '20% (fixed in webcam mode)'
  && !out.uploadMode.disabled && out.uploadAfterSlider10.label === '10%'
  && out.webcamAfterSlider10.disabled && out.webcamAfterSlider10.label === '20% (fixed in webcam mode)'
  && out.webcamRequestsConf.length === 1 && out.webcamRequestsConf[0] === '0.2'
  && out.uploadAfterSlider30.label === '30%' && out.uploadRequestConf[0] === '0.3'
  && out.webcamFinal.disabled;
out.errors = window.__aq ? window.__aq.errors : [];
return out;
