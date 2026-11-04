#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"unsafe archive member: {member.name}")
        # Every resolved member was checked above; omit the newer ``filter``
        # argument so the downloader remains compatible with Python 3.10/3.11.
        bundle.extractall(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the versioned JevGNSS data and checkpoint archive")
    parser.add_argument("--destination", type=Path, default=ROOT / ".assets")
    parser.add_argument("--archive", type=Path, help="Local release archive to unpack")
    args = parser.parse_args()
    metadata = json.loads((ROOT / "configs" / "paper_v1.json").read_text())["assets"]
    archive_meta = json.loads((ROOT / "expected" / "checksums.json").read_text()).get("archive", {})
    with tempfile.TemporaryDirectory(prefix="jevgnss-assets-") as temporary:
        local = Path(temporary) / "assets.tar.gz"
        if args.archive:
            shutil.copyfile(args.archive, local)
        else:
            url = metadata.get("dataset_url", "")
            location = f" The public dataset is at {url}." if url else ""
            raise RuntimeError(
                "This Dropbox link is not a direct archive download."
                + location
                + " Pass --archive to unpack the local release bundle."
            )
        if archive_meta.get("sha256") and sha256(local) != archive_meta["sha256"]:
            raise ValueError("release archive checksum mismatch")
        if args.destination.exists():
            raise FileExistsError(f"destination already exists: {args.destination}")
        args.destination.mkdir(parents=True)
        safe_extract(local, args.destination)
    print(json.dumps({"status": "downloaded", "destination": str(args.destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

