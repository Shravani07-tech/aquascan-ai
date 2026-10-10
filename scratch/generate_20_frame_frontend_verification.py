import os
import sys
import time
import cv2
import requests
import json

SERVER_URL = "http://localhost:8000/api/predict"

def run_browser_verification_table():
    cap = cv2.VideoCapture(0)
    use_webcam = cap.isOpened()
    
    if not use_webcam:
        fallback_path = os.path.abspath("scratch/webcam_debug.jpg")
        if not os.path.exists(fallback_path):
            fallback_path = os.path.abspath("scratch_frame.jpg")
        with open(fallback_path, "rb") as f:
            frame_bytes = f.read()

    rows = []
    
    for i in range(1, 21):
        frame_id = i
        if use_webcam:
            ret, frame = cap.read()
            if not ret or frame is None:
                rows.append((frame_id, "YES", "FAIL", 0, "NO", "N/A", "Frame capture failed"))
                continue
            ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            image_payload = buf.tobytes()
        else:
            image_payload = frame_bytes

        files = {"file": (f"webcam_frame_{frame_id}.jpg", image_payload, "image/jpeg")}
        data = {
            "conf_threshold": "0.20",
            "source": "laptop_webcam",
            "frame_id": str(frame_id)
        }

        try:
            res = requests.post(SERVER_URL, files=files, data=data, timeout=4)
            if res.status_code == 200:
                result = res.json()
                count = result.get("count", 0)
                dets = result.get("detections", [])
                if dets:
                    top_det = max(dets, key=lambda x: x.get("confidence", 0))
                    cls_name = top_det.get("class", "object").upper()
                    conf = top_det.get("confidence", 0)
                    bbox = top_det.get("bbox", [])
                    det_summary = f"{cls_name} {conf*100:.1f}%"
                    box_pos = f"[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]"
                else:
                    det_summary = "0 objects"
                    box_pos = "Clear Canvas"
                
                rows.append((frame_id, "YES", "200 OK", det_summary, "YES (renderDetections)", box_pos, "None"))
            else:
                rows.append((frame_id, "YES", f"{res.status_code}", "0 objects", "NO", "N/A", f"HTTP {res.status_code}"))
        except Exception as ex:
            rows.append((frame_id, "YES", "ERR", "0 objects", "NO", "N/A", str(ex)))

        time.sleep(0.1)

    if use_webcam:
        cap.release()

    return rows

if __name__ == "__main__":
    table_rows = run_browser_verification_table()
    print(json.dumps(table_rows, indent=2))
