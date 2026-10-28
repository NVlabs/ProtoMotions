# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch the pinned Metatron G1 package from Hugging Face (standard library only)."""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
from http.client import IncompleteRead
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "data/pretrained_models/metatron/huggingface.json"
DESTINATION = ROOT / "data/pretrained_models/metatron/G1"
REPO_ID = "nvidia/Metatron-v1"
FILES = {
    "LICENSE.pdf",
    "MODEL_CARD.md",
    "NOTICE.txt",
    "README.md",
    "config.yaml",
    "experiment_config.py",
    "inference_last.ckpt",
    "provenance.json",
    "resolved_configs.json",
}
RETRIES = 4
BACKOFF_SECONDS = 1.0
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def verify(path, record):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != record["size"]:
        return False
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest() == record["sha256"]


def validate_record(record):
    sha256 = record.get("sha256")
    if (
        type(record.get("size")) is not int
        or record["size"] <= 0
        or not isinstance(sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", sha256)
    ):
        raise ValueError("Invalid artifact size or SHA256")


def _retry_after(error):
    """Seconds requested by a 429/503 ``Retry-After`` header, else ``None``."""
    if isinstance(error, urllib.error.HTTPError):
        header = error.headers.get("Retry-After") if error.headers else None
        if header and header.strip().isdigit():
            return min(int(header.strip()), 60)
    return None


def _fatal(error):
    """An HTTP status we should not retry (for example a 404 mis-pin)."""
    return (
        isinstance(error, urllib.error.HTTPError) and error.code not in RETRYABLE_STATUS
    )


def _download_once(url, target, record):
    """One attempt: stream the body into a temp file, verify it, and hard-link
    it into place. Removes the temp file on success and on failure alike."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, suffix=".partial", delete=False
        ) as stream:
            temporary = Path(stream.name)
            print(f"Downloading {target.name} ({record['size']:,} bytes)", flush=True)
            count = 0
            with urlopen(url, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > record["size"]:
                        raise ValueError(f"Oversized response for {target.name}")
                    stream.write(chunk)
        # http.client returns a short body silently when the server closes
        # early, so check the length before the digest and raise a distinct,
        # retryable error rather than a digest mismatch that reads like a mis-pin.
        if count != record["size"]:
            raise ConnectionError(
                f"connection closed after {count:,} of {record['size']:,} bytes "
                f"for {target.name}; retry"
            )
        if not verify(temporary, record):
            raise ValueError(f"SHA256 mismatch for {target.name}")
        # tempfile creates the .partial with mode 0600 and the hard link shares
        # the inode, so without this every published file would be owner-only
        # whatever the umask. Publish it with the umask-reduced default instead.
        mask = os.umask(0)
        os.umask(mask)
        temporary.chmod(0o666 & ~mask)
        # os.link is atomic and fails if the destination exists, so it cannot
        # clobber a local file (Path.hardlink_to would require Python 3.10).
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download(url, target, record):
    """Publish only verified files; never replace unknown local data or symlinks."""
    validate_record(record)
    if target.exists() or target.is_symlink():
        if verify(target, record):
            print(f"Verified cached {target.name}", flush=True)
            return
        raise FileExistsError(
            f"Refusing to replace {target}; move the whole G1/ directory aside "
            "(or pass --output-dir) and retry"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, RETRIES + 1):
        try:
            _download_once(url, target, record)
            return
        except (
            urllib.error.URLError,
            TimeoutError,
            IncompleteRead,
            ConnectionError,
        ) as error:
            if _fatal(error) or attempt == RETRIES:
                raise
            wait = _retry_after(error)
            if wait is None:
                wait = BACKOFF_SECONDS * 2 ** (attempt - 1)
            print(
                f"Retry {attempt}/{RETRIES} for {target.name} after {error}",
                flush=True,
            )
            time.sleep(wait)


def fetch(destination=DESTINATION, *, configs_only=False, lock_path=LOCK):
    lock = json.loads(lock_path.read_text())
    if lock["repo_id"] != REPO_ID or not re.fullmatch(
        r"[0-9a-f]{40}", lock["revision"]
    ):
        raise ValueError("Expected the official repository and an immutable commit")
    base = f"https://huggingface.co/{REPO_ID}/resolve/{lock['revision']}"
    destination = Path(destination).resolve()
    # Remove stale .partial files from an interrupted earlier run so they cannot
    # accumulate or be mistaken for a published artifact.
    if destination.is_dir():
        for leftover in destination.glob("*.partial"):
            leftover.unlink(missing_ok=True)
    # The functional Hub index is verified against the checked-in lock. It
    # describes actual downloads, not a separate download-counter beacon.
    index_path = destination / "hub_manifest.json"
    download(f"{base}/config.json", index_path, lock["manifest"])
    index = json.loads(index_path.read_text())
    try:
        if index["schema_version"] != 1 or index["repo_id"] != REPO_ID:
            raise ValueError("Unsupported Hub artifact index")
        records = index["packages"]["G1"]["files"]
        paths = {record["path"] for record in records}
    except (KeyError, TypeError) as error:
        raise ValueError("Unsupported Hub artifact index") from error
    if len(records) != len(FILES) or paths != {f"G1/{f}" for f in FILES}:
        raise ValueError("Unexpected files in G1 package")
    for record in records:
        validate_record(record)
    # Include all documentation and legal files even for config-only checks.
    for record in records:
        name = record["path"].split("/")[1]
        if configs_only and name == "inference_last.ckpt":
            continue
        download(f"{base}/{record['path']}", destination / name, record)
    print(f"Verified G1 {'configs' if configs_only else 'package'}: {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    parser.add_argument(
        "--configs-only",
        action="store_true",
        help="Skip weights; configuration tests only",
    )
    args = parser.parse_args()
    try:
        fetch(args.output_dir, configs_only=args.configs_only)
    except urllib.error.HTTPError as error:
        where = getattr(error, "url", None) or getattr(error, "filename", "the Hub")
        print(f"Download failed: {where} returned HTTP {error.code}", file=sys.stderr)
        raise SystemExit(1)
    except urllib.error.URLError as error:
        print(
            f"Download failed: could not reach the Hub ({error.reason})",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
