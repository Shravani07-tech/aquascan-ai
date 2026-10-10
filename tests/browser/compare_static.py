"""Compare server boxes on the exact frame vs the browser overlay screenshot (static-object test).

Usage: .venv\\Scripts\\python.exe tests\\browser\\compare_static.py <cdp_output.txt> <screenshot.png> <out.jpg>
Prints, per detection, the box the server returned (frame px) and the box found in the browser
screenshot (converted to frame px), plus the pixel offset. Writes a side-by-side image:
left = server boxes drawn on the exact frame sent; right = browser screenshot cropped to the video.
"""
import base64
import json
import sys

import cv2
import numpy as np

HEX = {"bottle": "#00f5d4", "can": "#00bbf9", "carton": "#f15bb5", "paper": "#fee440", "plastic": "#ff0054"}


def bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


out_txt, shot_path, out_path = sys.argv[1:4]
j, _ = json.JSONDecoder().raw_decode(open(out_txt, encoding="utf-8").read().split("=== RESULT ===")[1].lstrip())
if not j.get("found"):
    sys.exit(f"No detection in time (frames sent: {j.get('framesSent')})")

frame = cv2.imdecode(np.frombuffer(base64.b64decode(j["frameDataUrl"].split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
fw, fh = j["frameSize"]
server = frame.copy()
for d in j["detections"]:
    x1, y1, x2, y2 = map(int, d["bbox"])
    cv2.rectangle(server, (x1, y1), (x2, y2), bgr(HEX.get(d["class"], "#ffffff")), 3)
    cv2.putText(server, f"{d['class']} {d['confidence']*100:.0f}%", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr(HEX.get(d["class"], "#ffffff")), 2)

shot = cv2.imread(shot_path)
dpr = j["dpr"]
L, T, W, H = [v * dpr for v in j["videoRect"]]
crop = shot[int(round(T)):int(round(T + H)), int(round(L)):int(round(L + W))]
crop = cv2.resize(crop, (fw, fh), interpolation=cv2.INTER_AREA)

# Locate each class colour in the screenshot crop (the overlay stroke), compare with the server box
worst = 0.0
for d in j["detections"]:
    col = np.array(bgr(HEX.get(d["class"], "#ffffff")), dtype=int)
    mask = (np.abs(crop.astype(int) - col).sum(axis=2) < 60).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        print(f"{d['class']} {d['confidence']*100:.0f}%: server {d['bbox']} | overlay colour not found in screenshot")
        continue
    # the box outline is the TALLEST component of that colour (the solid label tag is wide but only ~22 px tall)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_HEIGHT]))
    x, y, w, h = stats[k, :4]
    found = [int(x), int(y), int(x + w), int(y + h)]
    sx1, sy1, sx2, sy2 = d["bbox"]
    # compare left/right/bottom edges (top edge includes the label tag drawn above the box)
    err = max(abs(found[0] - sx1), abs(found[2] - sx2), abs(found[3] - sy2))
    worst = max(worst, err)
    print(f"{d['class']} {d['confidence']*100:.0f}%: server box {[round(v) for v in d['bbox']]} | browser overlay box {found} (frame px) | edge offset {err:.0f}px")

print(f"WORST EDGE OFFSET: {worst:.0f} frame px (<= ~4 px = stroke width/antialiasing, i.e. aligned)")
cv2.putText(server, "SERVER boxes on exact frame", (8, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.putText(crop, "BROWSER overlay (screenshot)", (8, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.imwrite(out_path, np.hstack([server, crop]))
print("wrote", out_path)
