"""Durable remote jobs: explicit commands, logs, PID and exit-status files."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAFE_CPUS = frozenset(range(24, 32))


def bind_safe_cpus():
    """Remote-5080: logical 24-31 map to physical cores 16-23."""
    available = os.sched_getaffinity(0)
    selected = available & SAFE_CPUS
    if not selected:
        raise RuntimeError("No approved CPUs available; refusing an unbound job")
    os.sched_setaffinity(0, selected)
    actual = os.sched_getaffinity(0)
    if actual != selected:
        raise RuntimeError("CPU affinity verification failed")
    return sorted(actual)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Use a simple job name")
    folder = ROOT / "jobs"
    folder.mkdir(exist_ok=True)
    status_path = folder / f"{args.name}.json"
    log_path = folder / f"{args.name}.log"
    if not args.worker:
        if status_path.exists():
            print(status_path.read_text())
            return
        command = args.command
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            raise ValueError("A command is required")
        cpu_affinity = bind_safe_cpus()
        status_path.write_text(json.dumps({"state": "starting", "command": command, "started": time.time(), "cpu_affinity": cpu_affinity}))
        with log_path.open("ab") as log:
            subprocess.Popen([sys.executable, __file__, "--worker", args.name, "--", *command],
                             cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        print(status_path.read_text())
        return
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    status = {"state": "running", "pid": os.getpid(), "command": command, "started": time.time()}
    status_path.write_text(json.dumps(status, indent=2))
    try:
        status["cpu_affinity"] = bind_safe_cpus()
        status_path.write_text(json.dumps(status, indent=2))
        environment = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
        result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        status.update(state="complete" if result.returncode == 0 else "failed", returncode=result.returncode, ended=time.time())
    except Exception as exc:
        status.update(state="failed", error=repr(exc), ended=time.time())
    status_path.write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
