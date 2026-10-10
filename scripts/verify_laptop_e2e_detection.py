import os
import sys
import glob
import base64
import json
from PIL import Image
import io

sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient
from backend.main import app

print("======================================================================")
print("   AquaScan AI — Laptop-Only E2E Detection Verification Suite")
print("======================================================================")

client = TestClient(app)

# 1. Verify /api/status
res_status = client.get("/api/status")
assert res_status.status_code == 200
status_data = res_status.json()
print("\n--- 1. Backend & Model Status ---")
print(f"  [+] Status        : {status_data.get('status')}")
print(f"  [+] Model         : {status_data.get('model')} (Fallback: {status_data.get('is_fallback')})")
print(f"  [+] Model Path    : {status_data.get('model_path')}")
print(f"  [+] Classes ({status_data.get('classes_count')}): {status_data.get('classes')}")
assert status_data['is_fallback'] == False
assert status_data['classes_count'] == 5

# Create directory for output visual plots
out_dir = os.path.abspath("outputs/e2e_verification_plots")
os.makedirs(out_dir, exist_ok=True)
print(f"\nVisual detection plots will be saved to: {out_dir}")

# Find test images covering different waste categories
test_imgs = sorted(glob.glob("dataset_clean/floating_waste_v10/test/images/*.jpg"))
print(f"Found {len(test_imgs)} total test set images.")

# Pick 5 diverse test images
sample_candidates = test_imgs[:5]

thresholds = [0.25, 0.15, 0.50]
results_summary = []

for conf in thresholds:
    print(f"\n======================================================================")
    print(f"   TESTING CONFIDENCE THRESHOLD: {conf:.2f}")
    print(f"======================================================================")
    
    threshold_summary = {"conf": conf, "total_detections": 0, "images": []}

    for idx, img_path in enumerate(sample_candidates, 1):
        fname = os.path.basename(img_path)
        with open(img_path, "rb") as f:
            res = client.post(
                "/api/predict",
                files={"file": (fname, f, "image/jpeg")},
                data={"conf_threshold": conf}
            )
        
        assert res.status_code == 200, f"Failed for {fname} at conf={conf}"
        data = res.json()
        assert data["success"] == True

        count = data["count"]
        cats = data["category_counts"]
        lat = data["latency_ms"]["total"]
        fps = data["latency_ms"]["fps"]
        detections = data["detections"]

        # Decode base64 image and save visual plot
        b64_url = data.get("image") or data.get("annotated_image")
        b64_str = b64_url.split(",")[1]
        img_bytes = base64.b64decode(b64_str)

        
        out_img_name = f"det_conf{int(conf*100):02d}_sample{idx}_{fname}"
        out_img_path = os.path.join(out_dir, out_img_name)
        with open(out_img_path, "wb") as f_out:
            f_out.write(img_bytes)

        threshold_summary["total_detections"] += count
        threshold_summary["images"].append({
            "filename": fname,
            "count": count,
            "categories": cats,
            "latency": lat,
            "fps": fps,
            "plot_path": out_img_path,
            "detections": detections
        })

        print(f"  [Image #{idx}] {fname:<45s} | Detections: {count:<2d} | Latency: {lat}ms ({fps} FPS)")
        if count > 0:
            for d_idx, d in enumerate(detections[:3], 1):
                print(f"     └─ Object {d_idx}: [{d['class'].upper()}] Conf: {d['confidence']*100:.1f}% | BBox: {d['bbox']}")

    results_summary.append(threshold_summary)

print("\n======================================================================")
print("   E2E DETECTION SUMMARY & ACCURACY ANALYSIS")
print("======================================================================")
for ts in results_summary:
    c = ts["conf"]
    td = ts["total_detections"]
    print(f"  - Conf Threshold {c:.2f} : {td} total detected objects across 5 sample images.")

print("\n[SUCCESS] Laptop-only end-to-end detection suite completed successfully!")
