# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from setuptools import find_namespace_packages


REPO_ROOT = Path(__file__).resolve().parents[2]
API_REFERENCE_DIR = REPO_ROOT / "docs/source/api_reference"
LEGAL_NOTICE = (
    REPO_ROOT / "legal/THIRD-PARTY SOFTWARE NOTICES AND ASSET LICENSES - ProtoMotions.txt"
)
CANONICAL_SPDX = [
    "# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
    "# SPDX-License-Identifier: Apache-2.0",
]


EXPECTED_API_REFERENCE_FILES = [
    "protomotions.agents.common.autoencoder.rst",
    "protomotions.agents.common.autoregressive.rst",
    "protomotions.agents.common.discrete_latent.rst",
    "protomotions.agents.common.fsq.rst",
    "protomotions.agents.common.fsq_config.rst",
    "protomotions.agents.common.latent.rst",
    "protomotions.agents.common.pretrained.rst",
    "protomotions.agents.common.supervision.rst",
    "protomotions.agents.peft.rst",
    "protomotions.agents.peft.actor.rst",
    "protomotions.agents.peft.adapters.rst",
    "protomotions.agents.peft.config.rst",
    "protomotions.agents.peft.model.rst",
    "protomotions.agents.peft.prior_agent.rst",
    "protomotions.agents.peft.prior_amp_agent.rst",
    "protomotions.agents.peft.prior_amp_config.rst",
    "protomotions.agents.peft.prior_amp_model.rst",
    "protomotions.agents.peft.prior_config.rst",
    "protomotions.agents.peft.prior_setup.rst",
    "protomotions.agents.peft.prior_with_peft.rst",
    "protomotions.agents.peft.sft_agent.rst",
    "protomotions.agents.peft.sft_model.rst",
    "protomotions.agents.peft.utils.rst",
    "protomotions.agents.peft.utils.adapter_state.rst",
    "protomotions.agents.peft.utils.frozen_prior_checkpoint.rst",
    "protomotions.agents.peft.utils.frozen_prior_contract.rst",
    "protomotions.agents.peft.utils.model_state.rst",
]


def test_gpc_peft_modules_are_in_api_reference_toctrees():
    agents_toctree = (API_REFERENCE_DIR / "protomotions.agents.rst").read_text()
    common_toctree = (API_REFERENCE_DIR / "protomotions.agents.common.rst").read_text()

    assert "protomotions.agents.peft" in agents_toctree
    for module in [
        "protomotions.agents.common.autoencoder",
        "protomotions.agents.common.autoregressive",
        "protomotions.agents.common.discrete_latent",
        "protomotions.agents.common.fsq",
        "protomotions.agents.common.fsq_config",
        "protomotions.agents.common.latent",
        "protomotions.agents.common.pretrained",
        "protomotions.agents.common.supervision",
    ]:
        assert module in common_toctree

    missing = [
        filename
        for filename in EXPECTED_API_REFERENCE_FILES
        if not (API_REFERENCE_DIR / filename).exists()
    ]
    assert missing == []


def test_release_legal_metadata_is_current():
    conf_text = (REPO_ROOT / "docs/source/conf.py").read_text()
    create_video_lines = (REPO_ROOT / "scripts/create_video.sh").read_text().splitlines()
    legal_text = LEGAL_NOTICE.read_text()

    assert 'copyright = "2025-2026, NVIDIA CORPORATION & AFFILIATES"' in conf_text
    assert 'author = "NVIDIA CORPORATION & AFFILIATES"' in conf_text
    assert "ProtoMotions Developers" not in conf_text
    assert create_video_lines[1:3] == CANONICAL_SPDX
    assert "- protomotions/data/assets/mesh/smpl/" in legal_text


def test_release_docs_do_not_reference_removed_public_surfaces():
    public_text_paths = [
        REPO_ROOT / "README.md",
        *sorted((REPO_ROOT / "docs/source").rglob("*.rst")),
    ]
    public_text = "\n".join(path.read_text() for path in public_text_paths)

    stale_fragments = [
        "ContextRouter",
        "protomotions.envs.context_router",
        "protomotions/envs/managers/",
        "protomotions/eval_agent.py",
        "examples/experiments/steering_mlp.py",
        "protomotions/agents/add/agent.py",
        "protomotions/data/robots/",
        "GPC Prior and PEFT",
        "arch.png",
        "data/pretrained_models/gpc_prior",
        "TODO: Add videos",
    ]
    missing = [fragment for fragment in stale_fragments if fragment in public_text]
    assert missing == []

    deploy_yaml = (
        REPO_ROOT
        / "data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.yaml"
    ).read_text()
    assert "exps/exp-" not in deploy_yaml


def test_quickstart_pretrained_table_matches_shipped_release_artifacts():
    quickstart = (REPO_ROOT / "docs/source/getting_started/quickstart.rst").read_text()

    assert "SOMA GPC prior" in quickstart
    assert "Releasing soon" in quickstart
    assert "SOMA BONES-SEED FSQ" in quickstart
    assert "data/pretrained_models/motion_tracker/soma_bones_fsq/inference_last.ckpt" in quickstart
    assert "   * - Vaulting" not in quickstart
    assert "   * - MaskedMimic G1" not in quickstart


def test_setup_discovers_protomotions_subpackages():
    setup_text = (REPO_ROOT / "setup.py").read_text()
    discovered_packages = set(find_namespace_packages(where=REPO_ROOT))

    assert "find_namespace_packages" in setup_text
    assert "protomotions.agents" in discovered_packages
    assert "protomotions.envs" in discovered_packages
    assert "protomotions.simulator" in discovered_packages
    assert 'packages=["protomotions"]' not in setup_text


def test_component_factories_public_exports_exist():
    import protomotions.envs.component_factories as component_factories

    missing = [
        name
        for name in component_factories.__all__
        if not hasattr(component_factories, name)
    ]
    assert missing == []


def test_public_release_surfaces_do_not_reference_nonportable_infrastructure():
    terrain_material = (
        REPO_ROOT / "protomotions/data/assets/usd/terrain_material.usda"
    ).read_text()

    assert not (REPO_ROOT / "Dockerfile.isaaclab").exists()
    assert "omniverse" + "://" not in terrain_material


def test_g1_deployment_docs_match_public_script_contract():
    guide = (
        REPO_ROOT / "docs/source/tutorials/workflows/g1_deployment.rst"
    ).read_text()
    requirements = (REPO_ROOT / "requirements_mujoco.txt").read_text()

    assert "--random-heading" not in guide
    assert "--explicit-pd" not in guide
    assert "onnxruntime" in requirements
    assert "pyyaml" in requirements.lower()
    assert "torch" in guide
