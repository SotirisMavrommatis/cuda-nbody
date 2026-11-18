"""Local server for the N-body studio.

Lets the browser run simulations: viewer.html sends a scenario, this server runs
nbody.exe on the GPU and hands back the .frames file to play.

Start it with start_studio.bat (Windows) or:  python server.py
Then open http://127.0.0.1:8765/  (it opens automatically).

Only reachable from this computer (127.0.0.1). Uses only the Python standard library.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

PORT = int(os.environ.get("NBODY_PORT", "8765"))
ROOT = os.path.dirname(os.path.abspath(__file__))
NBODY = os.path.join(ROOT, "nbody.exe" if os.name == "nt" else "nbody")
SCENARIOS = os.path.join(ROOT, "scenarios")
OUT = os.path.join(ROOT, "output", "studio")

# Safety limits so a typo (count=10000000) can't lock up the GPU for hours.
MAX_BODIES = 131072
MAX_STEPS = 100000
MAX_FILE_BYTES = 1_500_000_000
TIMEOUT_SEC = 900
KEEP_RUNS = 6  # older studio runs are deleted

DEFAULT_COUNT = {"galaxy": 4096, "cluster": 2048, "body": 1}
ALLOWED_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}

gpu_lock = threading.Lock()  # one simulation at a time


def estimate(text):
    """Rough size check of a scenario, mirroring the parser in nbody.cu."""
    bodies, steps, save_every = 0, 1500, 6
    for raw in text.splitlines():
        line = re.sub(r"\s*=\s*", "=", raw.split("#", 1)[0]).strip()
        if not line:
            continue
        tokens = line.split()
        if tokens[0] in DEFAULT_COUNT:
            count = DEFAULT_COUNT[tokens[0]]
            for t in tokens[1:]:
                if t.startswith("count=") and tokens[0] != "body":
                    try:
                        count = int(float(t[6:]))
                    except ValueError:
                        pass
            bodies += max(count, 0)
        else:
            for t in tokens:
                key, _, val = t.partition("=")
                try:
                    if key == "steps":
                        steps = int(float(val))
                    elif key == "save_every":
                        save_every = max(1, int(float(val)))
                except ValueError:
                    pass
    frames = steps // max(save_every, 1)
    return bodies, steps, frames, bodies * frames * 12


def safe_name(name):
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", (name or "").strip())[:60].strip("_")
    return name or "my_scenario"


def cleanup_old_runs():
    try:
        runs = sorted(
            (f for f in os.listdir(OUT) if f.endswith((".frames", ".txt"))),
            key=lambda f: os.path.getmtime(os.path.join(OUT, f)),
        )
        stems = []
        for f in reversed(runs):
            s = f.rsplit(".", 1)[0]
            if s not in stems:
                stems.append(s)
        for s in stems[KEEP_RUNS:]:
            for ext in (".frames", ".txt"):
                p = os.path.join(OUT, s + ext)
                if os.path.exists(p):
                    os.remove(p)
    except OSError:
        pass


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    # -- helpers ---------------------------------------------------------
    def log_message(self, fmt, *args):
        if self.command == "POST":
            sys.stderr.write("  %s\n" % (fmt % args))

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def host_ok(self):
        # Blocks other websites from talking to this server (DNS rebinding).
        return self.headers.get("Host", "") in ALLOWED_HOSTS

    def origin_ok(self):
        origin = self.headers.get("Origin")
        return origin is None or urlparse(origin).netloc in ALLOWED_HOSTS

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            raise ValueError("expected JSON")
        length = int(self.headers.get("Content-Length", "0"))
        if length > 200_000:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    # -- routes ----------------------------------------------------------
    def do_GET(self):
        if not self.host_ok():
            return self.send_error(403)
        path = urlparse(self.path).path
        if path == "/":
            self.send_response(302)
            self.send_header("Location", "/viewer.html")
            return self.end_headers()
        if path == "/api/status":
            return self.send_json({"ok": True, "nbody": os.path.exists(NBODY),
                                   "limits": {"bodies": MAX_BODIES, "steps": MAX_STEPS}})
        if path == "/api/scenarios":
            items = []
            if os.path.isdir(SCENARIOS):
                for f in sorted(os.listdir(SCENARIOS)):
                    if f.endswith(".txt"):
                        with open(os.path.join(SCENARIOS, f), encoding="utf-8", errors="replace") as fh:
                            items.append({"name": f[:-4], "text": fh.read()})
            return self.send_json({"scenarios": items})
        return super().do_GET()

    def do_POST(self):
        if not self.host_ok() or not self.origin_ok():
            return self.send_error(403)
        path = urlparse(self.path).path
        try:
            data = self.read_json()
        except (ValueError, json.JSONDecodeError) as e:
            return self.send_json({"ok": False, "error": str(e)}, 400)

        if path == "/api/simulate":
            return self.simulate(data)
        if path == "/api/save":
            return self.save(data)
        return self.send_error(404)

    def simulate(self, data):
        text = str(data.get("text", ""))
        name = safe_name(data.get("name"))
        if not os.path.exists(NBODY):
            return self.send_json({"ok": False, "error": "nbody.exe not found. Start the studio with start_studio.bat so it gets compiled."})

        bodies, steps, frames, size = estimate(text)
        if bodies > MAX_BODIES:
            return self.send_json({"ok": False, "error": f"Too many bodies ({bodies:,}). The limit is {MAX_BODIES:,}."})
        if steps > MAX_STEPS:
            return self.send_json({"ok": False, "error": f"Too many steps ({steps:,}). The limit is {MAX_STEPS:,}."})
        if size > MAX_FILE_BYTES:
            return self.send_json({"ok": False, "error": f"The result would be {size / 1e9:.1f} GB. Raise save_every or lower steps/count."})

        if not gpu_lock.acquire(blocking=False):
            return self.send_json({"ok": False, "error": "Another simulation is still running."})
        try:
            os.makedirs(OUT, exist_ok=True)
            stem = f"{name}_{time.strftime('%H%M%S')}"
            scen_path = os.path.join(OUT, stem + ".txt")
            frames_path = os.path.join(OUT, stem + ".frames")
            with open(scen_path, "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"Simulating '{name}': {bodies:,} bodies, {steps:,} steps ...")
            t0 = time.time()
            try:
                proc = subprocess.run([NBODY, scen_path, frames_path], cwd=ROOT,
                                      capture_output=True, text=True, timeout=TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                return self.send_json({"ok": False, "error": f"Stopped after {TIMEOUT_SEC} s."})
            log = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode != 0 or not os.path.exists(frames_path):
                return self.send_json({"ok": False, "error": "The simulation failed.", "log": log})
            print(f"  done in {time.time() - t0:.1f} s")
            cleanup_old_runs()
            rel = os.path.relpath(frames_path, ROOT).replace(os.sep, "/")
            return self.send_json({"ok": True, "log": log, "frames": "/" + rel, "name": name})
        finally:
            gpu_lock.release()

    def save(self, data):
        name = safe_name(data.get("name"))
        text = str(data.get("text", ""))
        os.makedirs(SCENARIOS, exist_ok=True)
        path = os.path.join(SCENARIOS, name + ".txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return self.send_json({"ok": True, "path": f"scenarios/{name}.txt"})


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"N-body studio running at {url}")
    print("Keep this window open while you use the studio. Close it (or press Ctrl+C) to stop.")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
