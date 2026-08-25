"""Rebuild the dashboard: run the demo (regenerates trace.json), then bake it
into a self-contained frontend/index.html (openable by double-click, no
server needed). Run this after changing demo/level.py or the planner.

    PYTHONPATH=. python3 frontend/build.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    subprocess.run([sys.executable, os.path.join(ROOT, "demo", "run_demo.py")],
                    check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": ROOT})

    trace = json.load(open(os.path.join(HERE, "trace.json")))
    tpl = open(os.path.join(HERE, "index_template.html")).read()
    html = tpl.replace("__TRACE_JSON__", json.dumps(trace))
    out = os.path.join(HERE, "index.html")
    open(out, "w").write(html)
    print(f"Wrote {out} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
