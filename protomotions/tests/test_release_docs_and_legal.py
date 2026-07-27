# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 and earlier
    import tomli as tomllib


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


def test_pyproject_discovers_only_protomotions_code_packages():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    package_finder = pyproject["tool"]["setuptools"]["packages"]["find"]
    package_data = pyproject["tool"]["setuptools"]["package-data"]["protomotions"]

    assert package_finder["include"] == ["protomotions", "protomotions.*"]
    assert package_finder["namespaces"] is False
    assert "protomotions.tests" in package_finder["exclude"]
    assert "data/assets/**/*" in package_data


def test_setup_discovers_protomotions_subpackages():
    """Actually run package discovery, not just string-match the TOML.

    With ``namespaces = false`` a directory without ``__init__.py`` is silently
    dropped from the wheel, so a purely declarative check cannot catch a
    subpackage that stops shipping. This executes the same finder setuptools
    uses and asserts the real result.
    """

    from setuptools import find_packages

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    package_finder = pyproject["tool"]["setuptools"]["packages"]["find"]
    discovered = set(
        find_packages(
            where=str(REPO_ROOT),
            include=package_finder["include"],
            exclude=package_finder["exclude"],
        )
    )

    for expected in (
        "protomotions.agents",
        "protomotions.components",
        "protomotions.envs",
        "protomotions.robot_configs",
        "protomotions.simulator",
        "protomotions.simulator.isaacgym",
        "protomotions.simulator.isaaclab",
        "protomotions.simulator.mujoco",
        "protomotions.simulator.newton",
        "protomotions.utils",
    ):
        assert expected in discovered, f"{expected} would not ship in the wheel"

    assert "protomotions.tests" not in discovered


def test_every_protomotions_subpackage_has_init_so_it_ships():
    """Guard against a new module directory silently missing from the wheel."""

    from setuptools import find_packages

    code_dirs = set()
    for path in (REPO_ROOT / "protomotions").rglob("*.py"):
        relative = path.relative_to(REPO_ROOT)
        parts = relative.parts[:-1]
        # `data/` holds package-data (assets plus the odd asset-generation
        # helper script); it is shipped as data, not imported as a package.
        if "tests" in parts or "data" in parts:
            continue
        if any(part.startswith(".") for part in parts):
            continue
        code_dirs.add(".".join(parts))

    discovered = set(
        find_packages(
            where=str(REPO_ROOT),
            include=["protomotions", "protomotions.*"],
            exclude=["protomotions.tests", "protomotions.tests.*"],
        )
    )
    missing = sorted(code_dirs - discovered)
    assert missing == [], (
        "these directories contain modules but would not ship in the wheel; "
        f"add an __init__.py: {missing}"
    )


def test_smpl_assets_are_excluded_from_built_distributions():
    """SMPL/SMPL-H assets are not Apache-2.0 (legal notice section 4)."""

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    excluded = pyproject["tool"]["setuptools"]["exclude-package-data"]["protomotions"]
    for pattern in (
        "data/assets/mesh/smpl/**/*",
        "data/assets/mjcf/smpl*.xml",
        "data/assets/usd/smpl*.usda",
    ):
        assert pattern in excluded, f"SMPL carve-out missing: {pattern}"


def test_core_dependency_bounds_preserve_preconfigured_environments():
    """Core bounds must not replace preselected NumPy or Torch builds."""

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    dependencies = pyproject["project"]["dependencies"]
    numpy_pin = next(dep for dep in dependencies if dep.startswith("numpy"))
    torch_pin = next(dep for dep in dependencies if dep.startswith("torch"))
    assert "<" not in numpy_pin
    assert torch_pin == "torch>=2.2"


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
