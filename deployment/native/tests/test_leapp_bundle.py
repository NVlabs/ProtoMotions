# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LEAPP bundle export and deployment-side resolution (no checkpoint needed)."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from examples.policy_artifact import (
    bundle_model_path,
    resolve_policy_artifact,
    validate_leapp_bundle,
)

REPO = Path(__file__).resolve().parents[1]
JOINTS = ["left_knee_joint", "right_knee_joint", "waist_yaw_joint"]
KEYS = [
    "noisy.dof_pos",
    "noisy.dof_vel",
    "raw_action_history",
    "masked_mimic.velocity_command_mask",
]
NAMES = [key.replace(".", "_") for key in KEYS]
OUTPUTS = ["action", "stiffness", "damping", "raw_action"]


class _ToyWrapper(torch.nn.Module):
    """Raw-context policy I/O with constant gains."""

    def __init__(self) -> None:
        super().__init__()
        self.policy = torch.nn.Linear(12, 3)
        self.register_buffer("kp", torch.tensor([40.0, 40.0, 20.0]))
        self.register_buffer("kd", torch.tensor([2.0, 2.0, 1.0]))

    def forward(self, dof_pos, dof_vel, raw_action_history, mask):
        features = torch.cat((dof_pos, dof_vel, raw_action_history.flatten(1)), dim=-1)
        raw_action = self.policy(features) * mask.to(features.dtype)
        batch = dof_pos.shape[0]
        return (
            0.5 * raw_action,
            self.kp.expand(batch, -1),
            self.kd.expand(batch, -1),
            raw_action,
        )


def _sample(seed: int):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn((1, 3), generator=generator),
        torch.randn((1, 3), generator=generator),
        torch.randn((1, 2, 3), generator=generator),
        torch.ones((1, 1), dtype=torch.bool),
    )


@unittest.skipUnless(importlib.util.find_spec("leapp"), "leapp is not installed")
class LeappBundleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from tools.leapp_export import export_leapp_bundle

        torch.manual_seed(0)
        cls.wrapper = _ToyWrapper().eval()
        cls.tmp = tempfile.TemporaryDirectory()
        samples = [_sample(seed) for seed in (1, 2)]
        cls.bundle = export_leapp_bundle(
            cls.wrapper, KEYS, NAMES, samples, Path(cls.tmp.name), "toy", JOINTS
        )
        # The raw-context manifest fields accompanying the graph.
        cls.manifest = {
            "input_names": NAMES,
            "input_shapes": [list(value.shape[1:]) for value in samples[0]],
            "input_dtypes": [
                str(value.dtype).removeprefix("torch.") for value in samples[0]
            ],
            "output_names": OUTPUTS,
            "input_keys": KEYS,
            "robot_contract": {"joint_names": JOINTS},
            "leapp_bundle": cls.bundle.name,
            "artifact_sha256": hashlib.sha256(
                bundle_model_path(cls.bundle).read_bytes()
            ).hexdigest(),
        }
        (cls.bundle.parent / "manifest.json").write_text(json.dumps(cls.manifest))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_yaml_contract_and_leapp_runtime(self) -> None:
        import yaml
        from leapp import InferenceManager

        node = yaml.safe_load(self.bundle.read_text())["models"]["policy"]
        inputs = {item["name"]: item for item in node["inputs"]}
        outputs = {item["name"]: item for item in node["outputs"]}
        self.assertEqual(list(inputs), NAMES)
        self.assertEqual(inputs["noisy_dof_pos"]["kind"], "state/joint/position")
        self.assertEqual(inputs["noisy_dof_pos"]["protomotions_key"], "noisy.dof_pos")
        self.assertEqual(set(outputs), set(OUTPUTS))
        self.assertEqual(outputs["action"]["kind"], "target/joint/position")
        self.assertEqual(outputs["stiffness"]["kind"], "kp")

        fresh = _sample(7)  # not seen while tracing
        actual = InferenceManager(str(self.bundle)).run_policy(
            {f"policy/{name}": value for name, value in zip(NAMES, fresh)}
        )
        with torch.no_grad():
            expected = self.wrapper(*fresh)
        for name, reference in zip(OUTPUTS, expected):
            torch.testing.assert_close(
                torch.as_tensor(actual[f"policy/{name}"]).float(),
                reference,
                rtol=1e-4,
                atol=1e-5,
            )

    def test_deployment_resolves_bundle(self) -> None:
        model, manifest = resolve_policy_artifact(self.bundle)
        self.assertEqual(model, bundle_model_path(self.bundle))
        self.assertEqual(manifest, self.bundle.parent / "manifest.json")

        for change in (
            lambda m: m.update(leapp_bundle="other.yaml"),
            lambda m: m["input_names"].reverse(),
            lambda m: m["input_shapes"].__setitem__(2, [3, 3]),
            lambda m: m.update(output_names=["action"]),
            lambda m: m["input_shapes"].pop(),
            lambda m: m["robot_contract"]["joint_names"].append("extra_joint"),
            lambda m: m["input_dtypes"].__setitem__(1, "float64"),
        ):
            manifest = copy.deepcopy(self.manifest)
            change(manifest)
            with self.assertRaises(ValueError):
                validate_leapp_bundle(self.bundle, manifest)

    def test_existing_onnx_export_preserves_model_and_runtime_outputs(self) -> None:
        from leapp import InferenceManager
        from tools.leapp_export import export_onnx_bundle

        with tempfile.TemporaryDirectory() as tmp:
            source = bundle_model_path(self.bundle)
            bundle = export_onnx_bundle(source, self.manifest, Path(tmp) / "bundle")
            self.assertEqual(
                bundle_model_path(bundle).read_bytes(), source.read_bytes()
            )
            self.assertEqual(
                resolve_policy_artifact(bundle)[0], bundle_model_path(bundle)
            )
            fresh = _sample(19)
            actual = InferenceManager(str(bundle)).run_policy(
                {f"policy/{name}": value for name, value in zip(NAMES, fresh)}
            )
            with torch.no_grad():
                expected = self.wrapper(*fresh)
            for name, reference in zip(OUTPUTS, expected):
                torch.testing.assert_close(
                    torch.as_tensor(actual[f"policy/{name}"]).float(), reference
                )
            with self.assertRaises(FileExistsError):
                export_onnx_bundle(source, self.manifest, bundle.parent)

    def test_existing_onnx_export_rejects_mismatched_graph_metadata(self) -> None:
        from tools.leapp_export import export_onnx_bundle

        for change in (
            lambda m: m.update(artifact_sha256="bad"),
            lambda m: m["input_shapes"].__setitem__(0, [4]),
            lambda m: m["robot_contract"]["joint_names"].append("extra_joint"),
            lambda m: m["input_names"].reverse(),
        ):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                metadata = copy.deepcopy(self.manifest)
                change(metadata)
                output = Path(tmp) / "bundle"
                with self.assertRaises(ValueError):
                    export_onnx_bundle(bundle_model_path(self.bundle), metadata, output)
                self.assertFalse(output.exists())

    def test_existing_onnx_export_rejects_external_weights(self) -> None:
        import onnx
        from tools.leapp_export import export_onnx_bundle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = onnx.load(bundle_model_path(self.bundle))
            source = root / "external.onnx"
            onnx.save_model(
                model,
                source,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="weights.bin",
                size_threshold=0,
            )
            metadata = copy.deepcopy(self.manifest)
            metadata["artifact_sha256"] = hashlib.sha256(
                source.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(ValueError, "self-contained"):
                export_onnx_bundle(source, metadata, root / "bundle")
            self.assertFalse((root / "bundle").exists())

    def test_native_preparation_accepts_bundle(self) -> None:
        # The toy contract is not a native one: resolving the bundle must succeed
        # and preparation stop at the contract check.
        with tempfile.TemporaryDirectory() as out:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.prepare_metatron_v1_policy",
                    "--model",
                    str(self.bundle),
                    "--output",
                    out,
                    "--recovery-plan",
                    "unused-recovery.json",
                ],
                cwd=REPO,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertIn("contract", result.stderr)
        self.assertNotIn("FileNotFoundError", result.stderr)

    def test_torchscript_helpers_need_pytorch_jit_disabled(self) -> None:
        # Direct wrapper tracing requires PYTORCH_JIT=0 before importing torch.
        script = (
            "import tempfile, torch\n"
            "from pathlib import Path\n"
            "from tools.leapp_export import export_leapp_bundle\n"
            "@torch.jit.script\n"
            "def double(x: torch.Tensor) -> torch.Tensor:\n"
            "    return 2.0 * x\n"
            "class W(torch.nn.Module):\n"
            "    def forward(self, x):\n"
            "        a = double(x)\n"
            "        return a, torch.ones_like(x), torch.ones_like(x), a\n"
            "export_leapp_bundle(W(), ['noisy.dof_pos'], ['noisy_dof_pos'],\n"
            "    [(torch.randn(1, 3),), (torch.randn(1, 3),)], Path(tempfile.mkdtemp()), 'jit', 'abc')\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "export_scripted.py"  # TorchScript needs a source file
            path.write_text(script)

            def run(jit: str) -> subprocess.CompletedProcess:
                env = dict(
                    os.environ,
                    PYTORCH_JIT=jit,
                    PYTHONPATH=os.pathsep.join(
                        [str(REPO), os.environ.get("PYTHONPATH", "")]
                    ),
                )
                return subprocess.run(
                    [sys.executable, str(path)],
                    cwd=REPO,
                    check=False,
                    env=env,
                    capture_output=True,
                    text=True,
                )

            scripted = run("1")
            self.assertNotEqual(scripted.returncode, 0)
            self.assertIn(
                "action outputs must be computed from the inputs", scripted.stderr
            )
            self.assertEqual(run("0").returncode, 0)


class BareOnnxArtifactTest(unittest.TestCase):
    def test_defaults_to_json_sidecar(self) -> None:
        self.assertEqual(
            resolve_policy_artifact(Path("/a/policy.onnx")),
            (Path("/a/policy.onnx"), Path("/a/policy.json")),
        )

    def test_rejects_inconsistent_bundle_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "policy.onnx"
            model.write_bytes(b"contract-only fixture")
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            manifest = {
                "leapp_bundle": "policy.yaml",
                "artifact_sha256": digest,
                "input_names": ["sensor"],
                "input_keys": ["noisy.dof_pos"],
                "input_shapes": [[3]],
                "input_dtypes": ["float32"],
                "output_names": OUTPUTS,
                "robot_contract": {"joint_names": JOINTS},
            }
            node = {
                "parameters": {"model_path": "policy.onnx", "sha256sum": digest},
                "inputs": [
                    {
                        "name": "sensor",
                        "shape": [1, 3],
                        "dtype": "float32",
                        "protomotions_key": "noisy.dof_pos",
                    }
                ],
                "outputs": [
                    {"name": name, "shape": [1, 3], "dtype": "float32"}
                    for name in OUTPUTS
                ],
            }
            bundle = root / "policy.yaml"
            bundle.write_text(yaml.safe_dump({"models": {"policy": node}}))
            validate_leapp_bundle(bundle, manifest)
            for change in (
                lambda n: n["inputs"][0].update(shape=[2, 3]),
                lambda n: n["inputs"][0].update(protomotions_key="wrong.context"),
                lambda n: n["outputs"][0].update(shape=[1, 4]),
                lambda n: n["outputs"][0].update(dtype="float64"),
                lambda n: n["outputs"].append(copy.deepcopy(n["outputs"][0])),
                lambda n: n["parameters"].update(sha256sum="wrong"),
                lambda n: n["parameters"].update(model_path="../policy.onnx"),
            ):
                with self.subTest(change=change):
                    changed = copy.deepcopy(node)
                    change(changed)
                    bundle.write_text(yaml.safe_dump({"models": {"policy": changed}}))
                    with self.assertRaises(ValueError):
                        validate_leapp_bundle(bundle, manifest)


if __name__ == "__main__":
    unittest.main()
