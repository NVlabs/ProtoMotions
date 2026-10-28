# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release and archive contracts, using tiny fixtures instead of model downloads."""

import hashlib
import json
import runpy
import stat
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INSTALL = runpy.run_path(str(ROOT / "scripts/install_metatron_demo.py"))


def write_bundle(directory, files, status="public-release", symlink=None):
    with zipfile.ZipFile(directory / "demo.zip", "w") as archive:
        for name, content in files.items():
            info = zipfile.ZipInfo(name)
            if name == symlink:
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, content)
    metadata = {
        "release_status": status,
        "sha256": hashlib.sha256((directory / "demo.zip").read_bytes()).hexdigest(),
        "files": {name: len(content) for name, content in files.items()},
    }
    (directory / "bundle.json").write_text(json.dumps(metadata))


@pytest.fixture
def files():
    values = dict.fromkeys(INSTALL["REQUIRED_FILES"], b"test")
    values["assets/policy/manifest.json"] = json.dumps(
        {
            "release_status": "public-release",
            "artifact_sha256": "1" * 64,
            "model_download": {
                "url": "https://huggingface.co/nvidia/Metatron-v1/resolve/"
                + "a" * 40
                + "/demo/model.onnx",
                "sha256": "1" * 64,
            },
        }
    ).encode()
    values["licenses/third-party/manifest.json"] = json.dumps(
        {"files": {"example/LICENSE": hashlib.sha256(b"notice").hexdigest()}}
    ).encode()
    values["licenses/third-party/example/LICENSE"] = b"notice"
    return values


def test_public_bundle_installs(tmp_path, files):
    write_bundle(tmp_path, files)
    target = tmp_path / "site"
    INSTALL["install_bundle"](tmp_path, target)
    assert (target / "index.html").read_bytes() == b"test"
    with pytest.raises(FileExistsError):
        INSTALL["install_bundle"](tmp_path, target)


def test_fixture_requires_explicit_preview(tmp_path, files):
    policy = json.loads(files["assets/policy/manifest.json"])
    policy["release_status"] = "integration-fixture"
    files["assets/policy/manifest.json"] = json.dumps(policy).encode()
    write_bundle(tmp_path, files, "integration-fixture")
    with pytest.raises(ValueError, match="not approved"):
        INSTALL["verify_bundle"](tmp_path)
    INSTALL["verify_bundle"](tmp_path, allow_fixture=True)


def test_release_flag_alone_cannot_publish_fixture(tmp_path, files):
    files["assets/policy/manifest.json"] = b'{"release_status":"integration-fixture"}'
    write_bundle(tmp_path, files)
    with pytest.raises(ValueError, match="statuses disagree"):
        INSTALL["verify_bundle"](tmp_path)


@pytest.mark.parametrize(
    "path",
    [
        "../outside.js",
        "/outside.js",
        "assets/../outside.js",
        "assets\\file.js",
        "assets/stream:ads.js",
        "src/main.js",
        "assets/main.ts",
        "assets/main.js.map",
        "package.json",
        "node_modules/package/index.js",
        ".git/config",
        "licenses/source.py",
    ],
)
def test_rejects_sources_and_unsafe_paths(tmp_path, files, path):
    files[path] = b"test"
    write_bundle(tmp_path, files)
    with pytest.raises(ValueError):
        INSTALL["verify_bundle"](tmp_path)


def test_rejects_inline_source_maps(tmp_path, files):
    files["assets/main.js"] = b"//# sourceMappingURL=data:application/json;base64,e30="
    write_bundle(tmp_path, files)
    with pytest.raises(ValueError, match="Source map"):
        INSTALL["verify_bundle"](tmp_path)


def test_rejects_symlinks(tmp_path, files):
    write_bundle(tmp_path, files, symlink="index.html")
    with pytest.raises(ValueError, match="regular asset"):
        INSTALL["verify_bundle"](tmp_path)


def test_rejects_tampered_archive(tmp_path, files):
    write_bundle(tmp_path, files)
    with (tmp_path / "demo.zip").open("ab") as stream:
        stream.write(b"unexpected bytes")
    with pytest.raises(ValueError, match="checksum mismatch"):
        INSTALL["verify_bundle"](tmp_path)


@pytest.mark.parametrize(
    "notice", ["assets/policy/LICENSE.pdf", "assets/recovery/NOTICE.txt"]
)
def test_rejects_missing_notice(tmp_path, files, notice):
    del files[notice]
    write_bundle(tmp_path, files)
    with pytest.raises(ValueError, match="Missing runtime files/notices"):
        INSTALL["verify_bundle"](tmp_path)


def test_rejects_changed_third_party_notice(tmp_path, files):
    files["licenses/third-party/example/LICENSE"] = b"wrong"
    write_bundle(tmp_path, files)
    with pytest.raises(ValueError, match="notice checksum"):
        INSTALL["verify_bundle"](tmp_path)


def test_rejects_manifest_naming_missing_member(tmp_path, files):
    files["licenses/third-party/manifest.json"] = json.dumps(
        {"files": {"absent/LICENSE": hashlib.sha256(b"x").hexdigest()}}
    ).encode()
    write_bundle(tmp_path, files)
    with pytest.raises(ValueError, match="missing member"):
        INSTALL["verify_bundle"](tmp_path)


def test_inventory_contains_runtime_only():
    metadata = INSTALL["read_metadata"](ROOT / "demos/metatron")
    for path in metadata["files"]:
        INSTALL["validate_path"](path)
    assert INSTALL["REQUIRED_FILES"] <= metadata["files"].keys()


@pytest.mark.parametrize(
    "change",
    [
        {
            "url": "https://huggingface.co/nvidia/Metatron-v1/resolve/main/demo/model.onnx"
        },
        {"url": "https://example.org/model.onnx"},
        {
            "url": "https://huggingface.co/nvidia/Metatron-v1/resolve/"
            + "a" * 40
            + "/racer/network.onnx"
        },
        {"sha256": "2" * 64},
        {"sha256": "invalid"},
    ],
)
def test_reject_unpinned_or_mismatched_download(tmp_path, files, change):
    policy = json.loads(files["assets/policy/manifest.json"])
    policy["model_download"].update(change)
    files["assets/policy/manifest.json"] = json.dumps(policy).encode()
    write_bundle(tmp_path, files)
    with pytest.raises(ValueError, match="pinned policy"):
        INSTALL["verify_bundle"](tmp_path)


def test_reject_remote_compression_contract(tmp_path, files):
    policy = json.loads(files["assets/policy/manifest.json"])
    policy["model_storage"] = {"encoding": "byte-shuffle-gzip-v1"}
    files["assets/policy/manifest.json"] = json.dumps(policy).encode()
    write_bundle(tmp_path, files)
    with pytest.raises(ValueError, match="pinned policy"):
        INSTALL["verify_bundle"](tmp_path)


@pytest.mark.parametrize(
    "member, match",
    [
        # A non-asset suffix is refused by validate_path before the member loop
        # reaches the bundled-weights name check.
        ("assets/policy/model.onnx.lossless", "Not a runtime asset or notice"),
        # An allowed suffix passes validate_path, so the bundled-weights check
        # itself must still reject anything under the hosted weights' name.
        ("assets/policy/model.onnx.json", "must not also be bundled"),
    ],
)
def test_reject_bundled_hosted_weights(tmp_path, files, member, match):
    files[member] = b"weights"
    write_bundle(tmp_path, files)
    with pytest.raises(ValueError, match=match):
        INSTALL["verify_bundle"](tmp_path)


@pytest.mark.parametrize("download", [None, "https://example.org/model.onnx"])
def test_reject_non_dict_model_download(tmp_path, files, download):
    policy = json.loads(files["assets/policy/manifest.json"])
    if download is None:
        del policy["model_download"]
    else:
        policy["model_download"] = download
    files["assets/policy/manifest.json"] = json.dumps(policy).encode()
    write_bundle(tmp_path, files)
    with pytest.raises(ValueError, match="pinned policy"):
        INSTALL["verify_bundle"](tmp_path)
