# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline tests of the native package download and integrity contract."""

import hashlib
import importlib.util
import io
import json
import os
import sys
import urllib.error
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "download_metatron", ROOT / "scripts/download_metatron.py"
)
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


def record(data):
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def hub(tmp_path, monkeypatch):
    payloads = {f"G1/{name}": name.encode() for name in downloader.FILES}
    index = {
        "schema_version": 1,
        "repo_id": downloader.REPO_ID,
        "packages": {
            "G1": {
                "files": [
                    {"path": name, **record(data)} for name, data in payloads.items()
                ]
            }
        },
    }
    payloads["config.json"] = json.dumps(index).encode()
    lock = {
        "repo_id": downloader.REPO_ID,
        "revision": "a" * 40,
        "manifest": record(payloads["config.json"]),
    }
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock))
    requests = []

    def open_url(url, timeout):
        prefix = f"https://huggingface.co/{downloader.REPO_ID}/resolve/{'a' * 40}/"
        assert url.startswith(prefix)
        name = url[len(prefix) :]
        requests.append(name)
        return io.BytesIO(payloads[name])

    monkeypatch.setattr(downloader, "urlopen", open_url)
    return lock_path, payloads, requests


def test_pinned_package_config_only_and_cached_upgrade(tmp_path, hub):
    lock, _, requests = hub
    dest = tmp_path / "G1"
    downloader.fetch(dest, configs_only=True, lock_path=lock)
    assert not (dest / "inference_last.ckpt").exists()
    assert {p.name for p in dest.iterdir()} == (
        downloader.FILES - {"inference_last.ckpt"}
    ) | {"hub_manifest.json"}
    requests.clear()
    downloader.fetch(dest, lock_path=lock)
    assert requests == ["G1/inference_last.ckpt"]
    requests.clear()
    downloader.fetch(dest, lock_path=lock)
    assert requests == []


@pytest.mark.parametrize("bad", [b"short", b"x" * 500])
def test_corrupt_download_not_published(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(downloader, "urlopen", lambda *a, **kw: io.BytesIO(bad))
    monkeypatch.setattr(downloader.time, "sleep", lambda *a: None)
    target = tmp_path / "model"
    # A short body is a retryable truncation (ConnectionError after the retries
    # run dry); an oversized body is a non-retryable ValueError. Neither leaves a
    # published file behind.
    with pytest.raises((ValueError, ConnectionError)):
        downloader.download("https://example.test/model", target, record(b"expected"))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("symlink", [False, True])
def test_unknown_files_are_never_overwritten(tmp_path, symlink):
    original = tmp_path / "original"
    original.write_bytes(b"local data")
    target = tmp_path / "model"
    if symlink:
        target.symlink_to(original)
    else:
        target.write_bytes(b"local data")
    with pytest.raises(FileExistsError):
        downloader.download("https://example.test/model", target, record(b"expected"))
    assert original.read_bytes() == target.read_bytes() == b"local data"


def test_interrupted_response_is_cleaned_up(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise TimeoutError("offline")

    monkeypatch.setattr(downloader, "urlopen", fail)
    monkeypatch.setattr(downloader.time, "sleep", lambda *a: None)
    with pytest.raises(TimeoutError):
        downloader.download(
            "https://example.test/model", tmp_path / "model", record(b"expected")
        )
    assert list(tmp_path.iterdir()) == []


def test_published_file_is_not_owner_only(tmp_path, monkeypatch):
    data = b"payload"
    monkeypatch.setattr(downloader, "urlopen", lambda *a, **kw: io.BytesIO(data))
    target = tmp_path / "model"
    previous = os.umask(0o022)
    try:
        downloader.download("https://example.test/model", target, record(data))
    finally:
        os.umask(previous)
    mode = target.stat().st_mode & 0o777
    assert mode != 0o600, "the shared temp-file inode must not stay owner-only"
    assert mode == 0o644


def test_short_body_then_full_body_succeeds_on_retry(tmp_path, monkeypatch):
    data = b"the full Metatron payload"
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(data[:4] if len(calls) == 1 else data)

    monkeypatch.setattr(downloader, "urlopen", flaky)
    monkeypatch.setattr(downloader.time, "sleep", lambda *a: None)
    target = tmp_path / "model"
    downloader.download("https://example.test/model", target, record(data))
    assert len(calls) == 2
    assert target.read_bytes() == data


def test_rate_limit_retry_after_is_honoured(tmp_path, monkeypatch):
    data = b"payload"
    slept = []
    monkeypatch.setattr(downloader.time, "sleep", lambda seconds: slept.append(seconds))
    calls = []

    def rate_limited(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                "https://example.test/model",
                429,
                "Too Many Requests",
                {"Retry-After": "7"},
                None,
            )
        return io.BytesIO(data)

    monkeypatch.setattr(downloader, "urlopen", rate_limited)
    target = tmp_path / "model"
    downloader.download("https://example.test/model", target, record(data))
    assert slept == [7]
    assert target.read_bytes() == data


def test_persistent_failure_prints_one_line(tmp_path, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise urllib.error.URLError("name resolution failed")

    monkeypatch.setattr(downloader, "fetch", boom)
    monkeypatch.setattr(
        sys, "argv", ["download_metatron.py", "--output-dir", str(tmp_path / "G1")]
    )
    with pytest.raises(SystemExit) as exit_info:
        downloader.main()
    assert exit_info.value.code == 1
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1
    assert "could not reach the Hub" in lines[0]


def test_path_traversal_in_index_is_rejected(tmp_path, hub):
    lock, payloads, requests = hub
    index = json.loads(payloads["config.json"])
    index["packages"]["G1"]["files"][0]["path"] = "G1/../../escape"
    payloads["config.json"] = json.dumps(index).encode()
    metadata = json.loads(lock.read_text())
    metadata["manifest"] = record(payloads["config.json"])
    lock.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Unexpected files"):
        downloader.fetch(tmp_path / "G1", lock_path=lock)
    assert requests == ["config.json"]
    assert not (tmp_path / "escape").exists()


def test_malformed_index_shape_is_rejected(tmp_path, hub):
    lock, payloads, _ = hub
    payloads["config.json"] = json.dumps(
        {"schema_version": 1, "repo_id": downloader.REPO_ID, "packages": {}}
    ).encode()
    metadata = json.loads(lock.read_text())
    metadata["manifest"] = record(payloads["config.json"])
    lock.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Unsupported Hub artifact index"):
        downloader.fetch(tmp_path / "G1", lock_path=lock)


def test_checked_in_lock_is_immutable_and_official():
    lock = json.loads(downloader.LOCK.read_text())
    assert lock["repo_id"] == "nvidia/Metatron-v1"
    assert downloader.re.fullmatch(r"[0-9a-f]{40}", lock["revision"])
    downloader.validate_record(lock["manifest"])
