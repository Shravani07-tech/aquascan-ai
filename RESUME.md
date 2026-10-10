# RESUME — AquaScan AI (SUPERSEDED on 2026-10-10 by FINAL_STATUS.md; kept for history)

**Commit at pause:** see `git log --oneline -1` (the commit that added this file; previous: `fd38fce`).
**Deadline:** everything done by **15:00**; review at 16:00.

## Done
- Plan A on real hardware: Pi 5 + ov5647 → `rpi_client.py --backend picamera2 --rotate 180` → laptop backend →
  dashboard Pi panel with boxes (3.8 fps). PI_SETUP.md written and smoke-tested.
- Fixes: picamera2 red/blue swap (`dc8c6ab`); `--rotate 180` (`01fcc5c`).
- Plan B cancelled; partial venv + pip cache removed from the Pi; Pi venv recreated clean
  (`--system-site-packages`) and Plan A re-verified 5/5 after cleanup.
- FINAL_STATUS.md drafted (Plan B marked "not completed — deferred").
- Laptop backend stopped. Collected frames are in `debug_frames/` (git-ignored): `preview` (74),
  `preview2` (94), `scene_bottle` (67: the invalid ceiling/hand-held run plus a few laptop test frames after #60).

## Left (in this order)
1. **Demo-scene test** (Plan A): preview → aim the Pi camera at the basin (low oblique, ~30°, 20–40 cm above
   the water, upright on the dashboard, no ceiling/people) → "scene ready" → 60 frames bottle (phase
   `scene_bottle2`) → "water only" → 30 frames (phase `scene_water2`) → §17 report, FP confidences and the
   threshold that would remove them (deployed threshold unchanged), 3 clean annotated frames for slides.
2. **Box-positioning fix** for laptop webcam mode + regression test (letterbox / mirror / non-4:3 / DPR 1.5),
   before/after screenshots.
3. **0.20 webcam confidence cap** fix (smallest change), full TESTING.md regression on AC, best.pt hash check,
   `git tag release-review-2026-10-09`, finalise FINAL_STATUS.md.

## First commands when back (laptop PowerShell, in the project folder)
1. Network — must be SSID **Shanky** and an IPv4 on **172.24.21.x** (laptop was 172.24.21.133):
   ```
   netsh wlan show interfaces ; ipconfig | findstr IPv4
   ```
   If the SSID is "SIESGST 2", reconnect to Shanky (and turn off auto-connect for SIESGST 2).
2. Power — charger in; then check it is really charging (Windows reported "Offline" once while plugged):
   ```
   powershell -Command "(Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus).PowerOnline"
   ```
   Must print `True`.
3. Power on the Pi, wait ~60 s, then find it (IP may change after reboot):
   ```
   ssh shanki@172.24.21.132 "hostname -I"
   ```
   If that times out, use the hostname instead (mDNS works for ssh from this laptop):
   ```
   ssh shanki@shanky.local "hostname -I"
   ```
   Use the printed IP in every Pi command (PI_SETUP.md). If both fail, check the phone hotspot's connected
   devices list for the Pi.
4. Say **"resume"** — Claude re-checks all of the above, starts the backend with the frame collector for the
   scene test, and streams a 30 s preview.

## Shutting down / restarting the Pi
- Shutdown needs the Pi password (sudo is not passwordless):
  `ssh -t shanki@172.24.21.132 "sudo shutdown -h now"`, then wait until the green LED stops flashing.
- The Pi venv for Plan A is `~/aquascan/.venv` (system site packages; no pip installs needed).
