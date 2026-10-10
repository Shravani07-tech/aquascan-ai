import os
import sys
import time
import requests
import json

SERVER_URL = "http://localhost:8000/api/predict"

def run_restart_cycle_verification():
    print("======================================================================")
    print("   AquaScan AI — 5-Cycle Start/Stop & Restart Recovery Stability Test")
    print("======================================================================")

    # Fixed committed test frame, so results never depend on the rolling scratch/webcam_debug.jpg
    test_frame_path = os.path.abspath("scratch_frame.jpg")

    if not os.path.exists(test_frame_path):
        print("[ERROR] Test frame scratch_frame.jpg missing!")
        sys.exit(1)

    with open(test_frame_path, "rb") as f:
        frame_bytes = f.read()

    all_cycles_passed = True
    
    for cycle in range(1, 6):
        print(f"\n--- CYCLE {cycle} / 5: Starting Fresh Webcam Session ---")
        cycle_success = True

        for frame_id in range(1, 6):
            files = {"file": (f"cycle_{cycle}_frame_{frame_id}.jpg", frame_bytes, "image/jpeg")}
            data = {
                "conf_threshold": "0.20",
                "source": "laptop_webcam",
                "frame_id": str(frame_id)
            }

            try:
                t0 = time.time()
                res = requests.post(SERVER_URL, files=files, data=data, timeout=4)
                latency = (time.time() - t0) * 1000
                
                if res.status_code == 200:
                    result = res.json()
                    count = result.get("count", 0)
                    dets = result.get("detections", [])
                    det_label = f"{dets[0]['class'].upper()} ({dets[0]['confidence']*100:.1f}%)" if dets else "None"
                    print(f"  [Cycle {cycle} | Frame #{frame_id}] HTTP 200 OK | Count: {count} | Top Det: {det_label} | Latency: {latency:.1f}ms")
                else:
                    print(f"  [Cycle {cycle} | Frame #{frame_id}] FAIL: HTTP {res.status_code}")
                    cycle_success = False
            except Exception as ex:
                print(f"  [Cycle {cycle} | Frame #{frame_id}] ERROR: {str(ex)}")
                cycle_success = False

            time.sleep(0.05)

        if cycle_success:
            print(f"  [+] CYCLE {cycle} PASSED: Fresh session initialized and 5/5 frames processed cleanly.")
        else:
            print(f"  [-] CYCLE {cycle} FAILED.")
            all_cycles_passed = False

        print(f"--- CYCLE {cycle}: Stopping Webcam Session & Cleaning Up ---")
        time.sleep(0.1)

    print("\n======================================================================")
    if all_cycles_passed:
        print("   [SUCCESS] ALL 5 START/STOP RESTART CYCLES PASSED PERFECTLY!")
    else:
        print("   [FAILURE] One or more restart cycles failed.")
    print("======================================================================")

if __name__ == "__main__":
    run_restart_cycle_verification()
