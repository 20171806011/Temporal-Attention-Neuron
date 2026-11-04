#!/usr/bin/env python3
"""Verify archive/source byte identity. Does not run research experiments."""
import hashlib
import json
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    sources = json.loads((here / "SOURCE_MANIFEST.json").read_text())
    archive = json.loads((here / "ARCHIVE_MANIFEST.json").read_text())
    checked = 0
    for base, entries in ((root, sources["sources"]), (here, archive["files"])):
        for entry in entries:
            contents = (base / entry["path"]).read_bytes()
            assert len(contents) == entry["size_bytes"], entry["path"]
            assert hashlib.sha256(contents).hexdigest() == entry["sha256"], entry["path"]
            checked += 1
    for entry in sources["byte_identical_copies"]:
        assert (here / entry["archive_path"]).read_bytes() == (
            root / entry["source_path"]
        ).read_bytes(), entry["archive_path"]
        checked += 1
    print(json.dumps({"status": "PASS", "identity_checks": checked,
                      "source_commit": sources["source_commit"],
                      "scope": "Archive/source hashes and snapshot byte identity; no experiment replay."}))


if __name__ == "__main__":
    main()
