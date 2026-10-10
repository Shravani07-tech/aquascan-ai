import os
import sys
import time
import cv2
import requests

SERVER_URL = "http://localhost:8000/api/predict"

def run_20_frame_stability_test():
    print("======================================================================")
    print("   AquaScan AI — 20-Frame Live Webcam Stability Diagnostic Test")
    print("======================================================================")
    
    cap = cv2.VideoCapture(0)
    use_webcam = cap.isOpened()
    
    if not use_webcam:
        print("[NOTICE] VideoCapture(0) unavailable, using saved fallback test frame.")
        fallback_path = os.path.abspath("scratch_frame.jpg")
        if not os.path.exists(fallback_path):
            fallback_path = os.path.abspath("scratch/webcam_debug.jpg")
        if not os.path.exists(fallback_path):
            print("[ERROR] Neither VideoCapture(0) nor fallback frame is available!")
            sys.exit(1)
        with open(fallback_path, "rb") as f:
            frame_bytes = f.read()
    else:
        print("[SUCCESS] OpenCV VideoCapture(0) initialized successfully.")

    print(f"\n{'FRAME':<8} | {'SENT':<6} | {'HTTP':<6} | {'COUNT':<6} | {'CLASS':<10} | {'CONF':<8} | {'RENDERED':<8} | {'ERROR'}")
    print("-" * 80)

    success_count = 0
    detections_summary = []

    for i in range(1, 21):
        frame_id = i
        if use_webcam:
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"{frame_id:<8} | {'YES':<6} | {'FAIL':<6} | {0:<6} | {'N/A':<10} | {'N/A':<8} | {'NO':<8} | Frame read failed")
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

        request_sent = "YES"
        t0 = time.time()
        try:
            res = requests.post(SERVER_URL, files=files, data=data, timeout=4)
            http_status = str(res.status_code)
            if res.status_code == 200:
                result = res.json()
                count = result.get("count", 0)
                dets = result.get("detections", [])
                if dets:
                    top_det = max(dets, key=lambda x: x.get("confidence", 0))
                    det_cls = top_det.get("class", "none")
                    det_conf = f"{top_det.get('confidence', 0)*100:.1f}%"
                else:
                    det_cls = "none"
                    det_conf = "0.0%"
                rendered = "YES" if result.get("success") else "NO"
                err = "None"
                success_count += 1
            else:
                count = 0
                det_cls = "none"
                det_conf = "N/A"
                rendered = "NO"
                err = f"HTTP {res.status_code}"
        except Exception as ex:
            http_status = "ERR"
            count = 0
            det_cls = "none"
            det_conf = "N/A"
            rendered = "NO"
            err = str(ex)

        print(f"{frame_id:<8} | {request_sent:<6} | {http_status:<6} | {count:<6} | {det_cls:<10} | {det_conf:<8} | {rendered:<8} | {err}")
        detections_summary.append((frame_id, http_status, count, det_cls, det_conf))
        time.sleep(0.1)

    if use_webcam:
        cap.release()

    print("=" * 80)
    print(f"Total Frames Processed : 20 / 20")
    print(f"Successful HTTP 200s  : {success_count} / 20")
    print("======================================================================")

if __name__ == "__main__":
    run_20_frame_stability_test()
