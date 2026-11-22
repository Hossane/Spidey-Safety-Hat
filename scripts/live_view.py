"""Live view of the hat camera, with what the tracker sees, in your browser.

    python scripts/live_view.py --host qnxpi-eth

Opens http://localhost:8765. Each box shows the track number, its movement and, for people,
the person-down state:
    green   standing / normal
    amber   wide and low in the frame (could be lying): counts up to "person down"
    red     an incident just fired (NEW, DOWN, LYING, CLOSE, APPROACH)
The camera can only be used by one program, so stop hat_detect.py / the companion first.
Ctrl+C to stop.
"""
import argparse
import io
import json
import subprocess
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, ImageDraw

latest = {"jpg": None, "n": 0}
cond = threading.Condition()

PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>Hat camera</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui;display:flex;flex-direction:column;
align-items:center}img{width:min(100vw,960px);image-rendering:auto}p{margin:8px}</style></head>
<body><img src="/stream"><p>Green: normal &middot; Amber: wide and low (could be lying) &middot;
Red: incident fired</p></body></html>"""


def draw(head, rgb):
    img = Image.frombytes("RGB", (head["w"], head["h"]), rgb).resize((head["w"] * 2, head["h"] * 2))
    d = ImageDraw.Draw(img)
    fired = {s.split()[-1] for s in head["incidents"]}          # "#id"
    for o in head["other"]:
        d.rectangle(o["box"], outline=(120, 160, 255), width=2)
        d.text((o["box"][0] + 4, o["box"][1] + 4), f'{o["name"]} {o["score"]}', fill=(120, 160, 255))
    for tr in head["tracks"]:
        if f'#{tr["id"]}' in fired:
            col = (255, 60, 60)
        elif tr["floor"] and tr["aspect"] <= 0.9:
            col = (255, 180, 0)
        else:
            col = (60, 220, 90)
        x0, y0, x1, y1 = tr["box"]
        d.rectangle([x0, y0, x1, y1], outline=col, width=4)
        lines = [f'#{tr["id"]} {tr["name"]} {tr["score"]}  {tr["motion"]}'
                 + (f'  {tr["ttc"]:.1f}s away' if tr["ttc"] else "")]
        if tr["name"] == "person":
            lines.append(f'shape {tr["aspect"]}  floor {"yes" if tr["floor"] else "no"}  '
                         f'down {tr["down"]}/{tr["need_down"]}  lying {tr["lying_s"]:.0f}/{tr["lying_need"]:.0f}s')
        ty = max(y0 - 30, 2)
        for i, line in enumerate(lines):
            d.rectangle([x0, ty + i * 14, x0 + 7 * len(line), ty + i * 14 + 13], fill=(0, 0, 0))
            d.text((x0 + 2, ty + i * 14), line, fill=col)
    d.rectangle([0, 0, 260, 18], fill=(0, 0, 0))
    d.text((4, 3), f'{head["fps"]} fps   {len(head["tracks"])} tracked', fill=(255, 255, 255))
    if head["incidents"]:
        msg = "  ".join(head["incidents"])
        d.rectangle([0, img.height - 26, img.width, img.height], fill=(180, 0, 0))
        d.text((8, img.height - 20), msg, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return buf.getvalue()


def reader(host, remote_dir):
    cmd = ["ssh", host, f"cd {remote_dir} && exec python3 -u scripts/live_view_pi.py"]
    while True:
        print(f"[view] connecting: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                if not line.startswith(b"{"):
                    continue
                head = json.loads(line)
                rgb = proc.stdout.read(head["w"] * head["h"] * 3)
                if len(rgb) < head["w"] * head["h"] * 3:
                    break
                for s in head["incidents"]:
                    print(f"[track] {s}", flush=True)
                jpg = draw(head, rgb)
                with cond:
                    latest["jpg"], latest["n"] = jpg, latest["n"] + 1
                    cond.notify_all()
        finally:
            proc.kill()
        print("[view] camera stream ended; retrying in 3 s", flush=True)
        time.sleep(3)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path != "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        seen = -1
        try:
            while True:
                with cond:
                    cond.wait_for(lambda: latest["n"] != seen, timeout=10)
                    jpg, seen = latest["jpg"], latest["n"]
                if jpg is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="qnxpi-eth", help="SSH host (default: qnxpi-eth, the cable)")
    ap.add_argument("--remote-dir", default="~/hatdetect")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    threading.Thread(target=reader, args=(args.host, args.remote_dir), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"[view] {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
