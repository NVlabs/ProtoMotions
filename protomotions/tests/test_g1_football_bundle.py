# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only bundle contract tests; synthetic graphs are not policy evaluations."""

import copy
import importlib.util
import io
import json
import stat
import sys
import warnings
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "football_bundle", ROOT / "scripts/install_g1_football.py"
)
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)
SIDE = f"{bundle.PERCEPTION}.manifest.json"
INNER = f"{bundle.PERCEPTION}.zip"


def encoded(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()


def zipped(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return stream.getvalue()


def set_perception(files, manifest, payloads):
    manifest = copy.deepcopy(manifest)
    manifest.pop("archive", None)
    manifest["inventorySha256"] = bundle.sha256(encoded(manifest["inventory"]))
    files[INNER] = zipped({**payloads, "manifest.json": encoded(manifest)})
    manifest["archive"] = {
        "file": "keeper-perception-cp313.zip",
        "bytes": len(files[INNER]),
        "sha256": bundle.sha256(files[INNER]),
    }
    files[SIDE] = encoded(manifest)


def edit_json(files, name, edit):
    value = json.loads(files[name])
    edit(value)
    files[name] = encoded(value)


@pytest.fixture
def assets():
    files = {name: b"fixture" for name in bundle.REQUIRED}
    files.update(
        {
            "assets/app-compiled.js": b"console.log('fixture')",
            "assets/robot.stl": b"mesh",
        }
    )
    source_hash = "1" * 64
    policy = {
        "release_status": "integration-fixture",
        "single_file_artifact_sha256": source_hash,
        "artifact_sha256": bundle.sha256(files["policy/model.onnx"]),
        "artifact_bytes": len(files["policy/model.onnx"]),
        "external_data": {
            "path": "weights.bin",
            "sha256": bundle.WEIGHTS_SHA256,
            "bytes": bundle.WEIGHTS_BYTES,
            "download": {"url": bundle.WEIGHTS_URL, "sha256": bundle.WEIGHTS_SHA256},
        },
    }
    for key, name in (
        ("network_graph", "network.onnx"),
        ("observations_graph", "observations.onnx"),
    ):
        data = files["policy/" + name]
        policy[key] = {"path": name, "bytes": len(data), "sha256": bundle.sha256(data)}
    policy["network_graph"]["source_sha256"] = source_hash
    files["policy/manifest.json"] = encoded(policy)
    packages = {}
    records = {}
    for name in ("numpy", "opencv-python"):
        filename = name + "-fixture.whl"
        data = zipped({name + "/__init__.py": b"# upstream dependency"})
        files["pyodide/" + filename] = data
        packages[name] = {
            "file": filename,
            "version": "fixture",
            "sha256": bundle.sha256(data),
        }
        records[name] = {
            "file_name": filename,
            "version": "fixture",
            "sha256": bundle.sha256(data),
        }
    files["pyodide/python_stdlib.zip"] = zipped({"os.pyc": b"upstream"})
    files["pyodide/pyodide-lock.json"] = encoded(
        {
            "info": {
                "python": "3.13.2",
                "abi_version": "2025_0",
                "arch": "wasm32",
                "version": "0.28.0.dev0",
            },
            "packages": records,
        }
    )
    bytecode = bytes.fromhex("f30d0d0a01000000") + b"12345678fixture"
    payloads = {
        "keeper_perception.pyc": bytecode,
        "results/ball_model.json": b"{}",
        "notices/LICENSE.txt": b"notice",
    }
    inventory = [
        {
            "path": path,
            "bytes": len(data),
            "sha256": bundle.sha256(data),
            "kind": "bytecode"
            if path.endswith(".pyc")
            else "notice"
            if path.startswith("notices/")
            else "model-data",
        }
        for path, data in payloads.items()
    ]
    set_perception(
        files,
        {
            "schema": 1,
            "format": "sourceless-cpython-bytecode-zip",
            "runtime": dict(bundle.RUNTIME),
            "entrypoint": "keeper_perception",
            "extractDir": "/football",
            "compile": {"invalidationMode": "UNCHECKED_HASH"},
            "inventory": inventory,
            "notices": ["notices/LICENSE.txt"],
            "dependencies": {
                "packages": packages,
                "pyodideLockSha256": bundle.sha256(files["pyodide/pyodide-lock.json"]),
            },
        },
        payloads,
    )
    files["policy/LICENSE.pdf"] = bundle.MODEL_LICENSE.read_bytes()
    files["policy/NOTICE.txt"] = (
        b"Licensed by NVIDIA Corporation under the NVIDIA Open Model License"
    )
    files["licenses/third-party/example.txt"] = b"upstream notice"
    files["licenses/third-party/manifest.json"] = encoded(
        {
            "files": {
                "example.txt": bundle.sha256(files["licenses/third-party/example.txt"]),
            }
        }
    )
    return files


def write_bundle(directory, files, *, status="integration-fixture"):
    (directory / "game.zip").write_bytes(zipped(files))
    info = {
        "release_status": status,
        "sha256": bundle.sha256((directory / "game.zip").read_bytes()),
        "files": {name: len(data) for name, data in files.items()},
        "policy_source_sha256": "1" * 64,
    }
    (directory / "bundle.json").write_bytes(encoded(info))
    return directory


def verify_fixture(tmp_path, assets):
    return bundle.verify(write_bundle(tmp_path, assets), allow_fixture=True)


def test_fixture_opt_in_and_install(tmp_path, assets):
    write_bundle(tmp_path, assets)
    with pytest.raises(ValueError, match="not marked public-release"):
        bundle.verify(tmp_path)
    destination = tmp_path / "site"
    bundle.install(tmp_path, destination, allow_fixture=True)
    assert (destination / "policy/model.onnx").read_bytes() == assets[
        "policy/model.onnx"
    ]
    assert not (destination / "policy/weights.bin").exists()
    with pytest.raises(FileExistsError):
        bundle.install(tmp_path, destination, allow_fixture=True)


def test_public_status_and_install(tmp_path, assets, monkeypatch, capsys):
    edit_json(
        assets,
        "policy/manifest.json",
        lambda m: m.update(release_status="public-release"),
    )
    write_bundle(tmp_path, assets, status="public-release")
    bundle.install(tmp_path, tmp_path / "site")
    monkeypatch.setattr(
        bundle, "metadata", lambda: {"release_status": "public-release"}
    )
    monkeypatch.setattr(sys, "argv", ["install_g1_football.py", "--status"])
    bundle.main()
    assert capsys.readouterr().out.strip() == "true"
    monkeypatch.setattr(
        bundle, "metadata", lambda: {"release_status": "integration-fixture"}
    )
    bundle.main()
    assert capsys.readouterr().out.strip() == "false"


def test_public_metadata_cannot_promote_an_embedded_fixture(tmp_path, assets):
    # --status is a cheap pre-fetch metadata gate. Installation must still check
    # the archive's independent policy status before writing any public files.
    write_bundle(tmp_path, assets, status="public-release")
    for allow_fixture in (False, True):
        with pytest.raises(ValueError, match="release statuses disagree"):
            bundle.install(tmp_path, tmp_path / "site", allow_fixture=allow_fixture)
    assert not (tmp_path / "site").exists()
    assert bundle.metadata(tmp_path)["release_status"] == "public-release"


def test_unknown_release_status_fails_closed(tmp_path, assets):
    write_bundle(tmp_path, assets, status="approved-looking-typo")
    with pytest.raises(ValueError, match="Unknown release status"):
        bundle.metadata(tmp_path)
    with pytest.raises(ValueError, match="Unknown release status"):
        bundle.verify(tmp_path, allow_fixture=True)


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "../escape.js",
        "/absolute.js",
        "a/../b.js",
        "a\\b.js",
        "a//b.js",
        "a/./b.js",
        ".hidden",
        "C:a.js",
        "nul\x00.js",
    ],
)
def test_path_safety(name):
    with pytest.raises(ValueError):
        bundle.safe_path(name)


@pytest.mark.parametrize(
    "name",
    [
        "src/main.js",
        "game.py",
        "app.ts",
        "app.js.map",
        "app.css.map",
        "package.json",
        "tests/a.js",
        "private_source/notes.py",
        "policy/weights.bin",
        "weights.bin",
        "full.onnx",
        "policy/full.onnx",
        "payload.zip",
        "README.md",
    ],
)
def test_reject_non_runtime_files(tmp_path, assets, name):
    assets[name] = b"not allowed"
    with pytest.raises(ValueError, match="Non-runtime"):
        verify_fixture(tmp_path, assets)


@pytest.mark.parametrize(
    "payload",
    [
        b"//# sourceMappingURL=data:source",
        b"/*# sourceMappingURL=app.map */",
        b"const KEEPER_PERCEPTION_PY = `source`;",
        b"const x = `\ndef secret():\n pass`; ",
    ],
)
def test_reject_embedded_source(tmp_path, assets, payload):
    assets["assets/app-compiled.js"] = payload
    with pytest.raises(ValueError, match="Source map|Python source"):
        verify_fixture(tmp_path, assets)


def test_checksum_and_inventory(tmp_path, assets):
    write_bundle(tmp_path, assets)
    (tmp_path / "game.zip").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        bundle.verify(tmp_path, allow_fixture=True)
    assets.pop("isolation-worker.js")
    with pytest.raises(ValueError, match="inventory"):
        verify_fixture(tmp_path, assets)


@pytest.mark.parametrize(
    "field,value",
    [
        ("release_status", "public-release"),
        ("single_file_artifact_sha256", "2" * 64),
        ("artifact_sha256", "2" * 64),
        ("artifact_bytes", 123),
    ],
)
def test_policy_contract(tmp_path, assets, field, value):
    edit_json(assets, "policy/manifest.json", lambda m: m.update({field: value}))
    with pytest.raises(ValueError):
        verify_fixture(tmp_path, assets)


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", "elsewhere.bin"),
        ("bytes", 1),
        ("sha256", "2" * 64),
        (
            "download",
            {"url": "https://example.org/weights.bin", "sha256": bundle.WEIGHTS_SHA256},
        ),
    ],
)
def test_remote_weight_pin(tmp_path, assets, field, value):
    edit_json(
        assets,
        "policy/manifest.json",
        lambda m: m["external_data"].update({field: value}),
    )
    with pytest.raises(ValueError, match="pinned policy"):
        verify_fixture(tmp_path, assets)


def test_graph_size_limit(tmp_path, assets, monkeypatch):
    monkeypatch.setattr(bundle, "MAX_GRAPH_BYTES", 1)
    with pytest.raises(ValueError, match="full models"):
        verify_fixture(tmp_path, assets)


@pytest.mark.parametrize("field", ["runtime", "inventorySha256", "archive"])
def test_perception_sidecar(tmp_path, assets, field):
    edit_json(
        assets,
        SIDE,
        lambda m: m.update({field: {} if field != "inventorySha256" else "bad"}),
    )
    with pytest.raises(ValueError):
        verify_fixture(tmp_path, assets)


@pytest.mark.parametrize(
    "name,payload,kind",
    [
        ("game.py", b"source", "bytecode"),
        ("nested.zip", b"PK", "model-data"),
        ("notices/bomb.zip", b"PK", "notice"),
        ("bad.pyc", b"wrong ABI" * 3, "bytecode"),
    ],
)
def test_perception_compiled_only(tmp_path, assets, name, payload, kind):
    manifest = json.loads(assets[SIDE])
    with zipfile.ZipFile(io.BytesIO(assets[INNER])) as archive:
        payloads = {
            n: archive.read(n) for n in archive.namelist() if n != "manifest.json"
        }
    payloads[name] = payload
    manifest["inventory"].append(
        {
            "path": name,
            "kind": kind,
            "bytes": len(payload),
            "sha256": bundle.sha256(payload),
        }
    )
    set_perception(assets, manifest, payloads)
    with pytest.raises(ValueError, match="perception member|bytecode ABI"):
        verify_fixture(tmp_path, assets)


def test_perception_notice_hash(tmp_path, assets):
    manifest = json.loads(assets[SIDE])
    with zipfile.ZipFile(io.BytesIO(assets[INNER])) as archive:
        payloads = {
            n: archive.read(n) for n in archive.namelist() if n != "manifest.json"
        }
    payloads["notices/LICENSE.txt"] = b"tampered"
    set_perception(assets, manifest, payloads)
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_fixture(tmp_path, assets)


@pytest.mark.parametrize(
    "name",
    [
        "policy/LICENSE.pdf",
        "policy/NOTICE.txt",
        "licenses/third-party/example.txt",
        "pyodide/numpy-fixture.whl",
        "pyodide/pyodide-lock.json",
    ],
)
def test_dependency_and_legal_integrity(tmp_path, assets, name):
    assets[name] = b"tampered"
    with pytest.raises(ValueError):
        verify_fixture(tmp_path, assets)


def test_unlisted_runtime_archive(tmp_path, assets):
    assets["pyodide/extra.whl"] = zipped({"x": b"x"})
    with pytest.raises(ValueError, match="Unexpected nested runtime"):
        verify_fixture(tmp_path, assets)


@pytest.mark.parametrize("attack", ["duplicate", "symlink", "traversal", "directory"])
def test_archive_member_safety(attack):
    stream = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(stream, "w") as archive:
            entry = zipfile.ZipInfo("asset.js")
            if attack == "symlink":
                entry.create_system = 3
                entry.external_attr = (stat.S_IFLNK | 0o777) << 16
            elif attack == "traversal":
                entry.filename = "../asset.js"
            elif attack == "directory":
                entry.filename = "folder/"
            archive.writestr(entry, b"fixture")
            if attack == "duplicate":
                archive.writestr("asset.js", b"second")
    with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as archive:
        with pytest.raises(ValueError):
            bundle.members(archive, 1000)


def test_nested_expansion_bounded_before_read(tmp_path, assets, monkeypatch):
    monkeypatch.setattr(bundle, "MAX_PERCEPTION_BYTES", 1)
    with pytest.raises(ValueError, match="expansion"):
        verify_fixture(tmp_path, assets)


def test_nested_runtime_cannot_hide_another_zip(tmp_path, assets):
    assets["pyodide/python_stdlib.zip"] = zipped({"bomb.zip": zipped({"x": b"x"})})
    with pytest.raises(ValueError, match="nested archive inside"):
        verify_fixture(tmp_path, assets)


@pytest.mark.needs_lfs
def test_real_compiled_bundle():
    path = bundle.BUNDLE / "game.zip"
    if not path.exists():
        pytest.skip("Compiled football bundle has not been supplied")
    with path.open("rb") as stream:
        if stream.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
            pytest.skip("Compiled football Git LFS payload not fetched")
    bundle.verify(allow_fixture=True)
