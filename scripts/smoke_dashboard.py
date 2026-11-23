"""Cross-platform dashboard startup smoke test.

Starts Streamlit headless on a saved run, waits for the health endpoint, fetches the page,
and shuts the server down. Uses only the standard library plus the project environment, so it
behaves the same on macOS, Linux, and Windows.

    uv run python scripts/smoke_dashboard.py --output-dir outputs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--run", default="latest")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from portfolio_engine.artifacts import find_run

    run = find_run(Path(args.output_dir), args.run).root.resolve()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "app" / "dashboard.py"),
        "--server.headless",
        "true",
        "--server.port",
        str(args.port),
        "--browser.gatherUsageStats",
        "false",
        "--theme.base",
        "light",
        "--",
        "--run",
        str(run),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    base = f"http://127.0.0.1:{args.port}"
    t0 = time.time()
    ok = False
    try:
        while time.time() - t0 < args.timeout:
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"{base}/_stcore/health", timeout=5) as r:
                    if r.status == 200 and r.read().decode().strip() == "ok":
                        with urllib.request.urlopen(base, timeout=10) as page:
                            ok = page.status == 200
                        break
            except OSError:
                time.sleep(1.0)
    finally:
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
    print(f"run: {run}")
    print(f"dashboard health: {'ok' if ok else 'FAILED'} after {time.time() - t0:.1f}s")
    if not ok:
        print(out[-4000:])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
