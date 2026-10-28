# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shipped docs and model cards must not name retired bundles, checkpoints or dirs.

v2 ships one config bundle, ``resolved_configs.json`` (a tagged JSON tree loaded
through ``load_experiment_configs``): there is no ``resolved_configs.pt``, no YAML
sidecar, and no separate ``resolved_configs_inference`` bundle. Shipped pretrained
checkpoints are ``inference_last.ckpt`` (and ``inference_last_lab.ckpt``), not the
training-output ``last.ckpt``. Four pretrained directories were removed. This
greps the public docs and the shipped model cards for those retired names so a
stale page or card fails.

Training-output ``last.ckpt`` paths (``results/<run>/last.ckpt``) stay legal; only
a pretrained-model path or a card's own artifact list may not name the retired
checkpoint.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_SOURCE = REPO_ROOT / "docs/source"
PRETRAINED = REPO_ROOT / "data/pretrained_models"

# Retired names with no legitimate use anywhere in scope.
RETIRED_NAMES = (
    "resolved_configs.pt",
    "resolved_configs_inference",  # .pt or .yaml
    "resolved_configs.yaml",
    "pretrained_models/masked_mimic/smpl",
    "pretrained_models/motion_tracker/smpl",  # covers smpl and smpl-terrains
    "soma_bones_fsq_amp_muon",
)
# A shipped pretrained-dir checkpoint is inference_last*.ckpt; the bare name under
# a pretrained_models path is retired (a results/<run>/last.ckpt output is not).
_SHIPPED_OLD_CKPT = re.compile(
    r"pretrained_models/\S*?(?<!inference_)last(?:_lab)?\.ckpt"
)
# A model card names its own directory's artifact, so a bare checkpoint there is
# retired wherever it appears.
_CARD_OLD_CKPT = re.compile(r"(?<!inference_)\blast(?:_lab)?\.ckpt")


def _doc_files():
    files = sorted(DOCS_SOURCE.rglob("*.rst")) + sorted(DOCS_SOURCE.rglob("*.md"))
    files += [REPO_ROOT / "README.md", REPO_ROOT / "CLAUDE.md"]
    return [p for p in files if p.is_file()]


def _card_files():
    return sorted(PRETRAINED.rglob("MODEL_CARD.md"))


def _offences(path, ckpt_regex):
    text = path.read_text()
    hits = [name for name in RETIRED_NAMES if name in text]
    if ckpt_regex.search(text):
        hits.append("the retired last.ckpt name")
    return [f"{path.relative_to(REPO_ROOT)}: names {hit}" for hit in hits]


def test_public_docs_and_cards_do_not_name_retired_bundles_or_checkpoints():
    offences = []
    for path in _doc_files():
        offences += _offences(path, _SHIPPED_OLD_CKPT)
    for path in _card_files():
        offences += _offences(path, _CARD_OLD_CKPT)
    assert not offences, "retired names in shipped docs/cards:\n" + "\n".join(offences)
