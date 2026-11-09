"""Verify the snapshot without importing model code."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT / "reproducibility/SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    checked = 0
    for item in manifest["files"]:
        path = ROOT / item["published_path"]
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Manifest path escapes repository")
        data = path.read_bytes()
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Snapshot mismatch: " + item["published_path"])
        checked += 1
    python_count = json_count = 0
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        if path.suffix == ".py":
            compile(path.read_bytes(), str(path), "exec")
            python_count += 1
        elif path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8-sig"))
            json_count += 1
    print(json.dumps({"passed": True, "snapshot_files": checked,
                      "python_files": python_count, "json_files": json_count,
                      "model_execution": False, "training": False}, indent=2))

if __name__ == "__main__":
    main()
