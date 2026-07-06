# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_ROOT = REPO_ROOT / "data/pretrained_models"
CATALOG_PATH = REPO_ROOT / "docs/source/getting_started/pretrained_models.rst"
QUICKSTART_PATH = REPO_ROOT / "docs/source/getting_started/quickstart.rst"
GPC_GUIDE_PATH = REPO_ROOT / "docs/source/user_guide/gpc.rst"
EXPERIMENTS_GUIDE_PATH = REPO_ROOT / "docs/source/user_guide/experiments.rst"
README_PATH = REPO_ROOT / "README.md"

REQUIRED_SECTIONS = (
    "Overview",
    "Intended Use",
    "Training",
    "Inputs and Outputs",
    "Artifacts",
    "Runtime Support",
    "Limitations",
    "Provenance",
)
PRIVATE_PATTERNS = (
    r"/lustre(?:/|\b)",
    r"/home(?:/|\b)",
    r"\bportfolios\b",
    r"\bnvr_",
    r"\bslurmrank\b",
    r"\b\d{6}(?:_[a-z0-9]+){4,}(?:\.pt)?\b",
    r"\bexp-20\d{6}_\d{6}\b",
)
G1_DEPLOY_DIR = PRETRAINED_ROOT / "motion_tracker/g1-bones-deploy"
SOMA_CONTINUOUS_DIR = PRETRAINED_ROOT / "motion_tracker/soma-bones"


def _model_dirs() -> List[Path]:
    return sorted({checkpoint.parent for checkpoint in PRETRAINED_ROOT.glob("*/*/*.ckpt")})


def test_every_pretrained_model_directory_has_a_model_card():
    missing_cards = [
        str(model_dir.relative_to(REPO_ROOT))
        for model_dir in _model_dirs()
        if not (model_dir / "MODEL_CARD.md").is_file()
    ]

    assert missing_cards == []


def test_model_cards_are_public_safe_and_describe_support():
    violations = []

    for model_dir in _model_dirs():
        card = model_dir / "MODEL_CARD.md"
        if not card.is_file():
            continue
        text = card.read_text()
        relative_card = str(card.relative_to(REPO_ROOT))

        for section in REQUIRED_SECTIONS:
            if f"## {section}" not in text:
                violations.append(f"{relative_card}: missing {section!r} section")
        for pattern in PRIVATE_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                violations.append(f"{relative_card}: contains private pattern {pattern!r}")

        if "`last.ckpt`" not in text:
            violations.append(f"{relative_card}: does not describe last.ckpt")
        if model_dir == SOMA_CONTINUOUS_DIR:
            if "fine-tuned in **IsaacLab**" not in text:
                violations.append(
                    f"{relative_card}: does not name the IsaacLab fine-tune"
                )
        elif "Training simulator: **IsaacLab**" not in text:
            violations.append(f"{relative_card}: does not name IsaacLab training")

        if model_dir == G1_DEPLOY_DIR:
            if "Simulator expectation: **Expected to transfer**" not in text:
                violations.append(f"{relative_card}: missing G1 transfer expectation")
        elif "Simulator expectation: **Training simulator only**" not in text:
            violations.append(f"{relative_card}: missing simulator-specific expectation")

    assert violations == []


def test_public_pretrained_docs_do_not_expose_private_provenance():
    violations = []

    for path in (CATALOG_PATH, QUICKSTART_PATH):
        text = path.read_text()
        relative_path = str(path.relative_to(REPO_ROOT))
        for pattern in PRIVATE_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                violations.append(
                    f"{relative_path}: contains private pattern {pattern!r}"
                )

    assert violations == []


def test_pretrained_model_catalog_links_every_card():
    assert CATALOG_PATH.is_file()
    catalog = CATALOG_PATH.read_text()
    missing_links = []

    for model_dir in _model_dirs():
        relative_card = (model_dir / "MODEL_CARD.md").relative_to(REPO_ROOT)
        github_url = (
            "https://github.com/NVlabs/ProtoMotions/blob/main/"
            f"{relative_card.as_posix()}"
        )
        if github_url not in catalog:
            missing_links.append(github_url)

    assert missing_links == []


def test_quickstart_uses_the_model_catalog_and_existing_paths():
    quickstart = QUICKSTART_PATH.read_text()

    assert ":doc:`pretrained_models`" in quickstart
    assert "soma-bones-deploy" not in quickstart
    assert "masked_mimic/g1" not in quickstart
    assert "motion_tracker/soma-bones/last.ckpt" in quickstart

    referenced_checkpoints = re.findall(
        r"data/pretrained_models/[A-Za-z0-9_./-]+\.ckpt", quickstart
    )
    missing_checkpoints = [
        checkpoint
        for checkpoint in referenced_checkpoints
        if not (REPO_ROOT / checkpoint).is_file()
    ]
    assert missing_checkpoints == []


def test_gpc_docs_reference_shipped_assets_and_current_entry_points():
    gpc_guide = GPC_GUIDE_PATH.read_text()
    experiments_guide = EXPERIMENTS_GUIDE_PATH.read_text()
    quickstart = QUICKSTART_PATH.read_text()
    readme = README_PATH.read_text()

    assert "examples/experiments/mimic/fsq.py" in gpc_guide
    assert "results/tracker_gpc_soma23/last.ckpt" not in gpc_guide
    assert (
        "data/pretrained_models/motion_tracker/soma_bones_fsq/"
        "inference_last.ckpt"
    ) in gpc_guide
    referenced_experiments = re.findall(
        r"examples/experiments/[A-Za-z0-9_./-]+\.py", gpc_guide
    )
    missing_experiments = [
        experiment
        for experiment in referenced_experiments
        if not (REPO_ROOT / experiment).is_file()
    ]
    assert missing_experiments == []
    assert (
        "agent.pretrained_modules.prior.checkpoint_path=/path/to/prior/last.ckpt"
        in gpc_guide
    )

    required_assets = (
        REPO_ROOT / "data/motion_for_trackers/crouch_soma23.pt",
        REPO_ROOT / "docs/source/_static/gpc_prior_unconditional.mp4",
        REPO_ROOT / "docs/source/_static/gpc_location_peft.mp4",
    )
    missing_assets = [
        str(path.relative_to(REPO_ROOT))
        for path in required_assets
        if not path.is_file()
    ]
    assert missing_assets == []
    assert "gpc_prior_unconditional.mp4" in experiments_guide
    assert "gpc_location_peft.mp4" in experiments_guide
    assert "data/motion_for_trackers/crouch_soma23.pt" in quickstart
    assert "GPC and PEFT guide" in readme
    assert "protomotions/data/assets/mjcf/" in readme
    assert "protomotions/data/robots/" not in readme


def test_gpc_and_discrete_latent_modules_are_in_the_api_reference():
    agents_index = (
        REPO_ROOT / "docs/source/api_reference/protomotions.agents.rst"
    ).read_text()
    common_index = (
        REPO_ROOT / "docs/source/api_reference/protomotions.agents.common.rst"
    ).read_text()

    assert "protomotions.agents.peft" in agents_index
    for module in (
        "autoencoder",
        "autoregressive",
        "discrete_latent",
        "fsq",
        "fsq_config",
        "latent",
        "pretrained",
        "supervision",
    ):
        reference = REPO_ROOT / (
            "docs/source/api_reference/protomotions.agents.common."
            f"{module}.rst"
        )
        assert reference.is_file()
        assert f"protomotions.agents.common.{module}" in common_index

    assert (
        REPO_ROOT / "docs/source/api_reference/protomotions.agents.peft.rst"
    ).is_file()
