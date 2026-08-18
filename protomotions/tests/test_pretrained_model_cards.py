# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path
from typing import List

import pytest
import torch
import yaml


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
SOMA_FSQ_DIR = PRETRAINED_ROOT / "motion_tracker/soma_bones_fsq"
SOMA_GPC_PRIOR_DIR = PRETRAINED_ROOT / "gpc_prior/soma_bones"


def _model_dirs() -> List[Path]:
    """Every published model directory: one that ships a checkpoint or a card.

    Keyed on the union of ``*.ckpt`` and ``MODEL_CARD.md``. The checkpoints are
    committed via Git LFS, so globbing for them matches. Keying on the card
    alone would make ``test_every_pretrained_model_directory_has_a_model_card``
    a tautology -- it would check that card-bearing directories have a card --
    and would let the public-safety scan below skip any directory that ships a
    checkpoint but no card. The union keeps both honest.
    """
    dirs = {p.parent for p in PRETRAINED_ROOT.glob("*/*/*.ckpt")}
    dirs |= {p.parent for p in PRETRAINED_ROOT.glob("*/*/MODEL_CARD.md")}
    return sorted(dirs)


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
    assert "motion_tracker/soma-bones/last_lab.ckpt" in quickstart

    referenced_checkpoints = re.findall(
        r"data/pretrained_models/[A-Za-z0-9_./-]+\.ckpt", quickstart
    )
    assert referenced_checkpoints, "quickstart should reference at least one checkpoint"

    # Checked against the model directory each path names, not the exact
    # checkpoint file: most checkpoints are committed, but some (the IsaacLab
    # fine-tune last_lab.ckpt) are fetched separately and are not in the tree.
    # What must hold either way is that the docs cannot drift onto a model
    # directory that does not exist.
    missing_models = [
        checkpoint
        for checkpoint in referenced_checkpoints
        if not (REPO_ROOT / checkpoint).parent.joinpath("MODEL_CARD.md").is_file()
    ]
    assert missing_models == []


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
    assert "examples/experiments/gpc/target_prior_peft.py" in gpc_guide
    assert (
        "agent.pretrained_modules.prior.checkpoint_path="
        "data/pretrained_models/gpc_prior/soma_bones/inference_last.ckpt"
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


@pytest.mark.needs_lfs
def test_soma_gpc_artifacts_use_current_config_contracts():
    tracker_config = torch.load(
        SOMA_FSQ_DIR / "resolved_configs.pt", weights_only=False
    )
    prior_config = torch.load(
        SOMA_GPC_PRIOR_DIR / "resolved_configs.pt", weights_only=False
    )
    prior_inference_config = torch.load(
        SOMA_GPC_PRIOR_DIR / "resolved_configs_inference.pt",
        weights_only=False,
    )

    assert tracker_config["agent"].model.actor.in_keys == [
        "max_coords_obs", "mimic_target_poses"
    ]
    assert prior_config["agent"].model.prior.context_encoder.in_keys == [
        "max_coords_obs"
    ]
    assert prior_config["agent"].model.latent_decoder.checkpoint_path == (
        "data/pretrained_models/motion_tracker/soma_bones_fsq/"
        "inference_last.ckpt"
    )
    assert prior_inference_config["agent"].model.latent_decoder.checkpoint_path == ""
    assert (
        prior_inference_config["agent"].model.latent_decoder.module_config is not None
    )


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


@pytest.mark.needs_lfs
def test_pretrained_configs_preserve_training_robot_friction():
    half_friction_model_dirs = (
        PRETRAINED_ROOT / "gpc_prior/soma_bones",
        PRETRAINED_ROOT / "masked_mimic/smpl",
        PRETRAINED_ROOT / "motion_tracker/smpl-terrains",
        PRETRAINED_ROOT / "motion_tracker/smpl",
        PRETRAINED_ROOT / "motion_tracker/soma-bones",
        PRETRAINED_ROOT / "motion_tracker/soma_bones_fsq",
    )
    violations = []

    # A model listed here may not be present in every checkout -- checkpoints and
    # their configs are sometimes fetched separately. Loading unconditionally
    # would turn an absent model into a FileNotFoundError that reads like a
    # broken test rather than a missing artifact, so skip what is not there.
    present = [d for d in half_friction_model_dirs if (d / "resolved_configs.pt").is_file()]
    assert present, "no pretrained configs found to check"

    for model_dir in present:
        training_path = model_dir / "resolved_configs.pt"
        training_configs = torch.load(
            training_path, map_location="cpu", weights_only=False
        )
        domain_randomization = getattr(
            training_configs["simulator"], "domain_randomization", None
        )
        friction_randomization = (
            getattr(domain_randomization, "friction", None)
            if domain_randomization is not None
            else None
        )
        if friction_randomization is not None:
            relative_path = training_path.relative_to(REPO_ROOT)
            violations.append(f"{relative_path}: unexpectedly randomizes friction")

        for config_name in ("resolved_configs.pt", "resolved_configs_inference.pt"):
            config_path = model_dir / config_name
            configs = torch.load(config_path, map_location="cpu", weights_only=False)
            actual_friction = getattr(
                configs["simulator"], "default_robot_friction", None
            )
            if actual_friction != 0.5:
                relative_path = config_path.relative_to(REPO_ROOT)
                violations.append(
                    f"{relative_path}: expected robot friction 0.5, "
                    f"got {actual_friction}"
                )

            yaml_path = config_path.with_suffix(".yaml")
            readable_configs = yaml.safe_load(yaml_path.read_text())
            readable_friction = readable_configs["simulator"].get(
                "default_robot_friction"
            )
            if readable_friction != 0.5:
                relative_path = yaml_path.relative_to(REPO_ROOT)
                violations.append(
                    f"{relative_path}: expected robot friction 0.5, "
                    f"got {readable_friction}"
                )

    assert violations == []
