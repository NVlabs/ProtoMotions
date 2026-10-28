# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "g1_bundle", ROOT / "scripts/install_g1_racer.py"
)
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


def game_archive(name="browser_game.pyc"):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(name, b"fixture")
    return stream.getvalue()


def fixture(directory, extra=None, *, release_status="integration-fixture"):
    files = {name: b"fixture" for name in bundle.REQUIRED}
    files["game-runtime.zip"] = game_archive()
    source_hash = "1" * 64
    files["manifest.json"] = json.dumps(
        {"release_status": release_status, "artifact_sha256": source_hash}
    ).encode()
    files["policy/network.json"] = json.dumps(
        {
            "url": (
                "https://huggingface.co/nvidia/Metatron-v1/resolve/"
                + "a" * 40
                + "/racer/network.onnx"
            ),
            "sha256": "2" * 64,
            "source_sha256": source_hash,
            "bytes": 396370481,
        }
    ).encode()
    files["licenses/third-party/manifest.json"] = b'{"files":{}}'
    files.update(extra or {})
    with zipfile.ZipFile(directory / "game.zip", "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    info = {
        "release_status": release_status,
        "sha256": hashlib.sha256((directory / "game.zip").read_bytes()).hexdigest(),
        "files": {name: len(content) for name, content in files.items()},
        "policy_source_sha256": source_hash,
    }
    (directory / "bundle.json").write_text(json.dumps(info))
    return directory


def test_non_public_release_requires_explicit_opt_in(tmp_path):
    fixture(tmp_path)
    with pytest.raises(ValueError, match="not marked public-release"):
        bundle.verify(tmp_path)
    bundle.verify(tmp_path, allow_fixture=True)


def test_public_release_installs_without_opt_in(tmp_path):
    fixture(tmp_path, release_status="public-release")
    destination = tmp_path / "site"
    bundle.install(tmp_path, destination)
    assert (destination / "index.html").is_file()


def test_public_release_rejects_mismatched_manifest(tmp_path):
    fixture(
        tmp_path,
        release_status="public-release",
        extra={
            "manifest.json": json.dumps(
                {"release_status": "integration-fixture", "artifact_sha256": "1" * 64}
            ).encode()
        },
    )
    with pytest.raises(ValueError, match="release statuses disagree"):
        bundle.verify(tmp_path)


@pytest.mark.parametrize(
    "name", ["../escape.js", "/absolute.js", "a/../b.js", "a\\b.js", ".hidden"]
)
def test_reject_path_traversal(name):
    with pytest.raises(ValueError):
        bundle.safe_path(name)


@pytest.mark.parametrize(
    "name", ["src/main.js", "game.py", "main.ts", "app.js.map", "package.json"]
)
def test_reject_application_source(tmp_path, name):
    fixture(tmp_path, {name: b"source"})
    with pytest.raises(ValueError):
        bundle.verify(tmp_path, allow_fixture=True)


def test_reject_source_hidden_in_game_archive(tmp_path):
    fixture(tmp_path, {"game-runtime.zip": game_archive("browser_game.py")})
    with pytest.raises(ValueError, match="game member"):
        bundle.verify(tmp_path, allow_fixture=True)


def test_reject_sourcemap_reference(tmp_path):
    fixture(tmp_path, {"app.js": b"//# sourceMappingURL=app.js.map"})
    with pytest.raises(ValueError, match="Source map"):
        bundle.verify(tmp_path, allow_fixture=True)


def test_reject_observation_debug_information(tmp_path):
    fixture(tmp_path, {"observations.wasm": b"fixture.debug_info"})
    with pytest.raises(ValueError, match="debug/source maps"):
        bundle.verify(tmp_path, allow_fixture=True)


def test_checksum_and_existing_destination_are_protected(tmp_path):
    fixture(tmp_path)
    with pytest.raises(FileExistsError):
        bundle.install(tmp_path, tmp_path, allow_fixture=True)
    with (tmp_path / "game.zip").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        bundle.verify(tmp_path, allow_fixture=True)


def test_install_into_new_directory(tmp_path):
    fixture(tmp_path)
    destination = tmp_path / "site"
    bundle.install(tmp_path, destination, allow_fixture=True)
    assert (destination / "index.html").is_file()


def test_checked_in_metadata_is_compiled_only():
    info = bundle.metadata()
    # release_status is either public-release or integration-fixture; the
    # installer's metadata() already rejects any other value.
    assert info["release_status"] in {"public-release", "integration-fixture"}
    assert bundle.REQUIRED <= info["files"].keys()
    assert sum(info["files"].values()) <= 550_000_000
    assert not any(
        Path(name).suffix in {".py", ".ts", ".map"} for name in info["files"]
    )
    assert (
        "model_wasm_fused.onnx" not in info["files"]
    )  # One weight payload for both providers.


def test_docs_link_and_public_hosting_gate():
    assert "user_guide/g1_racer" in (ROOT / "docs/source/index.rst").read_text()
    workflow = (ROOT / ".github/workflows/deploy-docs.yml").read_text()
    assert "steps.racer.outputs.ready == 'true'" in workflow
    assert "--allow-fixture" not in workflow
    # The preview workflow (.github/workflows/g1-racer.yml) is dev-only and the
    # release projection excludes it, so this shipped test must not read it (it
    # would FileNotFoundError in the public tree). Its assertions live with the
    # other development-repository projection checks, which are excluded from the
    # release.


@pytest.mark.parametrize(
    "changes",
    [
        {
            "url": "https://huggingface.co/nvidia/Metatron-v1/resolve/"
            "main/racer/network.onnx"
        },
        {"url": "https://example.org/network.onnx"},
        {"source_sha256": "3" * 64},
        {"sha256": "invalid"},
        {"bytes": -1},
    ],
)
def test_reject_unpinned_or_mismatched_network(tmp_path, changes):
    fixture(tmp_path)
    with zipfile.ZipFile(tmp_path / "game.zip") as archive:
        remote = json.loads(archive.read("policy/network.json"))
    remote.update(changes)
    fixture(tmp_path, {"policy/network.json": json.dumps(remote).encode()})
    with pytest.raises(ValueError, match="pinned policy"):
        bundle.verify(tmp_path, allow_fixture=True)


def test_reject_duplicate_hosted_weights(tmp_path):
    fixture(tmp_path, {"network.onnx": b"weights"})
    with pytest.raises(ValueError, match="must not also be bundled"):
        bundle.verify(tmp_path, allow_fixture=True)


@pytest.mark.needs_lfs
def test_real_bundle_verifies_when_materialised():
    """When the Git LFS payload is present, the checked-in bundle must pass the
    full installer verification over the real archive -- the synthetic fixtures
    above never exercise the shipped members, inventory or checksums.

    Marked ``needs_lfs`` so the no-payload CPU jobs deselect it; the LFS jobs and
    the preview workflow (after its ``git lfs pull``) run it on the real file."""
    archive = ROOT / "demos/g1-racer/game.zip"
    with archive.open("rb") as stream:
        if stream.read(64).startswith(b"version https://git-lfs"):
            pytest.skip("game.zip is a Git LFS pointer; run after `git lfs pull`")
    bundle.verify()
