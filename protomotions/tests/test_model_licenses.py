# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import runpy
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[2]
POLICY = runpy.run_path(str(ROOT / "scripts/check_model_licenses.py"))
MODEL = "data/pretrained_models/test/model"


@pytest.fixture
def release(tmp_path):
    (tmp_path / "legal").mkdir()
    shutil.copyfile(ROOT / "legal/NVIDIA-Open-Model-License.pdf",
                    tmp_path / "legal/NVIDIA-Open-Model-License.pdf")
    (tmp_path / "legal/model-license-exceptions.json").write_text("{}")
    model = tmp_path / MODEL
    model.mkdir(parents=True)
    shutil.copyfile(tmp_path / "legal/NVIDIA-Open-Model-License.pdf", model / "LICENSE.pdf")
    (model / "NOTICE.txt").write_text(POLICY["NOTICE"])
    (model / "last.ckpt").write_text("LFS pointer; no payload needed")
    (model / "MODEL_CARD.md").write_text(
        "# Test model\n\n## License\n\nModel license: NVIDIA Open Model License\n\n"
        "[License](LICENSE.pdf) and [Notice](NOTICE.txt)\n")
    return tmp_path


def test_published_models_comply():
    assert POLICY["check"](ROOT) == []


def test_complete_bundle_passes(release):
    assert POLICY["check"](release) == []


@pytest.mark.parametrize("filename", ["MODEL_CARD.md", "LICENSE.pdf", "NOTICE.txt"])
def test_missing_required_file_fails(release, filename):
    (release / MODEL / filename).unlink()
    assert POLICY["check"](release)


@pytest.mark.parametrize("filename", ["LICENSE.pdf", "NOTICE.txt"])
def test_altered_license_or_notice_fails(release, filename):
    (release / MODEL / filename).write_text("Apache-2.0")
    assert POLICY["check"](release)


@pytest.mark.parametrize("suffix", [".ckpt", ".onnx", ".safetensors", ".pth", ".pt", ".engine"])
def test_uncarded_release_is_not_skipped(release, suffix):
    orphan = release / "data/pretrained_models/new/export_only"
    orphan.mkdir(parents=True)
    (orphan / ("weights" + suffix)).write_text("pointer")
    assert any("new/export_only" in e for e in POLICY["check"](release))


def test_nested_exports_use_enclosing_card(release):
    export = release / MODEL / "compiled_models"
    export.mkdir()
    (export / "policy.onnx").write_text("pointer")
    assert POLICY["check"](release) == []


def test_explicit_exception_requires_matching_card_and_license(release):
    entry = {"license": "Example third-party license", "license_file": "LICENSE.txt",
             "reason": "Upstream model has separate terms"}
    (release / "legal/model-license-exceptions.json").write_text(json.dumps({MODEL: entry}))
    assert POLICY["check"](release)
    (release / MODEL / "MODEL_CARD.md").write_text(
        "## License\nModel license: Example third-party license\n"
        "License exception: Upstream model has separate terms\n"
        "[License](LICENSE.txt) and [Notice](NOTICE.txt)\n")
    (release / MODEL / "LICENSE.txt").write_text("Example full license")
    (release / MODEL / "NOTICE.txt").write_text("Example upstream attribution")
    assert POLICY["check"](release) == []


@pytest.mark.parametrize("entry", [
    {}, {"license": "Other", "reason": "", "license_file": "LICENSE.txt"},
    {"license": "Other", "reason": "Upstream", "license_file": "../LICENSE.txt"},
    {"license": "Other", "reason": "Upstream", "license_file": "MODEL_CARD.md"},
])
def test_malformed_exception_fails(release, entry):
    (release / "legal/model-license-exceptions.json").write_text(json.dumps({MODEL: entry}))
    assert POLICY["check"](release)


def test_stale_exception_fails(release):
    (release / "legal/model-license-exceptions.json").write_text('{"removed/model": {}}')
    assert any("Stale" in e for e in POLICY["check"](release))


def test_canonical_license_cannot_drift(release):
    (release / "legal/NVIDIA-Open-Model-License.pdf").write_text("Replacement")
    assert any("canonical" in e for e in POLICY["check"](release))


def test_code_license_stays_apache():
    assert "Apache License" in (ROOT / "LICENSE.md").read_text()
    assert "SPDX-License-Identifier: Apache-2.0" in (ROOT / ".license-header.txt").read_text()


def test_public_ci_checks_model_licenses_without_lfs():
    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/public-ci.yml").read_text())
    steps = workflow["jobs"]["model-licenses"]["steps"]
    assert any(step.get("run") == "python scripts/check_model_licenses.py" for step in steps)
    checkout = next(step for step in steps if step.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["lfs"] is False
