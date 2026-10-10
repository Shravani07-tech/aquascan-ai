// Minimal CDP driver: launches real Chrome (real camera, auto-granted permission),
// opens a URL, evaluates a JS file in the page, prints result + console logs.
// Usage: node cdp.mjs <url> <script.js> [--headed] [--browser=chrome|edge]
import { spawn } from 'node:child_process';
import { readFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const [url, scriptPath, ...flags] = process.argv.slice(2);
const headed = flags.includes('--headed');
const edge = flags.includes('--browser=edge');
const exe = edge
  ? 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
  : 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const port = 9300 + Math.floor(Math.random() * 500);
const profile = mkdtempSync(join(tmpdir(), 'aqcdp-'));
const args = [
  `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
  '--use-fake-ui-for-media-stream', '--no-first-run', '--no-default-browser-check',
  '--autoplay-policy=no-user-gesture-required',
  ...(process.env.FAKECAM ? ['--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${process.env.FAKECAM}`] : []),
  ...(headed ? [] : ['--headless=new']), '--window-size=1280,900', 'about:blank'
];
const proc = spawn(exe, args, { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));

let targets;
for (let i = 0; i < 150; i++) {
  try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); break; } catch { await sleep(200); }
}
if (!targets) { console.log('NO DEBUG PORT'); proc.kill(); process.exit(1); }
const page = targets.find(t => t.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener('open', r));
let id = 0; const pending = new Map(); const logs = []; let onLoad = null;
// Resolves on the next Page.loadEventFired (or after 15 s), so scripts never run on a half-loaded page.
const nextLoad = () => new Promise(r => { onLoad = r; setTimeout(r, 15000); });
ws.addEventListener('message', ev => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (m.method === 'Page.loadEventFired' && onLoad) { onLoad(); onLoad = null; }
  if (m.method === 'Runtime.consoleAPICalled') {
    logs.push(`[${m.params.type}] ` + m.params.args.map(a => a.value ?? a.description ?? '').join(' '));
  }
  if (m.method === 'Runtime.exceptionThrown') logs.push('[EXCEPTION] ' + JSON.stringify(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text));
});
const send = (method, params = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });

await send('Runtime.enable');
await send('Page.enable');
if (process.env.VIEWPORT) { // "width,height,dpr" — emulate a window size and display scale (e.g. 1275,800,1.5)
  const [width, height, dpr] = process.env.VIEWPORT.split(',').map(Number);
  await send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: dpr || 1, mobile: false });
}
if (process.env.PRELOAD) await send('Page.addScriptToEvaluateOnNewDocument', { source: readFileSync(process.env.PRELOAD, 'utf8') });
let loaded = nextLoad();
await send('Page.navigate', { url });
await loaded; await sleep(1000); // app init (status + samples fetch) after load
if (process.env.PRE_SCRIPT) {
  const pre = await send('Runtime.evaluate', { expression: `(async () => { ${readFileSync(process.env.PRE_SCRIPT, 'utf8')} })()`, awaitPromise: true, returnByValue: true });
  console.log('=== PRE_SCRIPT (before reload) ===');
  console.log(JSON.stringify(pre.result?.result?.value ?? pre.result));
  loaded = nextLoad();
  await send('Page.reload', { ignoreCache: false });
  await loaded; await sleep(1000);
}
const expr = readFileSync(scriptPath, 'utf8');
const res = await send('Runtime.evaluate', { expression: `(async () => { ${expr} })()`, awaitPromise: true, returnByValue: true, timeout: 600000 });
const value = res.result?.result?.value;
console.log('=== RESULT ===');
console.log(JSON.stringify(value ?? res.result ?? res.error, null, 2));
// A harness failure (no value returned) must not look like a pass.
if (value === undefined) { console.log('=== HARNESS ERROR: script returned no value ==='); process.exitCode = 1; }
if (process.env.SHOT) {
  const { writeFileSync } = await import('node:fs');
  const s = await send('Page.captureScreenshot', { format: 'png' });
  writeFileSync(process.env.SHOT, Buffer.from(s.result.data, 'base64'));
  console.log('screenshot ->', process.env.SHOT);
}
if (process.env.SHOW_LOGS) { console.log('=== CONSOLE ==='); logs.forEach(l => console.log(l)); }
console.log('=== CONSOLE ERRORS/EXCEPTIONS ===');
logs.filter(l => l.startsWith('[error]') || l.startsWith('[EXCEPTION]')).forEach(l => console.log(l));
ws.close(); proc.kill(); await sleep(500); process.exit(process.exitCode ?? 0);
