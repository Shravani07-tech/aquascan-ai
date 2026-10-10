import uvicorn
import os
import sys
import socket

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.config import HOST, PORT

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

if __name__ == "__main__":
    local_ip = get_local_ip()
    print("=" * 65)
    print("      AquaScan AI — Marine Floating Waste Perception Platform")
    print("=" * 65)
    print(f"Local Development URL : http://localhost:{PORT}")
    print(f"LAN / Raspberry Pi URL: http://{local_ip}:{PORT}")
    print(f"API Documentation     : http://localhost:{PORT}/docs")
    print(f"Server Host           : {HOST}")
    print(f"Server Port           : {PORT}")
    print("=" * 65)
    print("Notice: Raspberry Pi edge clients must connect using the LAN IP:")
    print(f"        http://{local_ip}:{PORT}/api/predict")
    print("=" * 65)
    
    try:
        uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=False)
    except OSError as e:
        if "10048" in str(e) or "address already in use" in str(e).lower():
            print("\n[ERROR] Port conflict detected (WinError 10048):")
            print(f"Port {PORT} is already occupied by another running process.")
            print("To resolve this on Windows, find and stop the existing process:")
            print(f"  cmd > netstat -ano | findstr :{PORT}")
            print("  cmd > taskkill /PID <PID> /F")
            print("Or set a different port via environment variable:")
            print(f"  cmd > set AQUASCAN_PORT=8001 && python run_server.py\n")
        sys.exit(1)

