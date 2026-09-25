"""Provenance manifest: hash the inputs + code state that produced a result."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def write_manifest(path: str, inputs: list):
    entry = {
        "python": sys.version.split()[0],
        "git_rev": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "inputs": {
            str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in inputs
            if Path(p).exists()
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entry, f, indent=2)
