# Metatron v1 — G1

The complete native package is hosted at
[nvidia/Metatron-v1/G1](https://huggingface.co/nvidia/Metatron-v1/tree/main/G1).
It includes the inference checkpoint, resolved configuration, reconstruction
recipe, [model card](https://huggingface.co/nvidia/Metatron-v1/blob/main/G1/MODEL_CARD.md),
NVIDIA Open Model License and attribution notice. No SOMA policy is included.

From the ProtoMotions repository root with MuJoCo dependencies installed:

```bash
python scripts/download_metatron.py
python protomotions/inference_agent.py --simulator mujoco --num-envs 1 \
  --checkpoint data/pretrained_models/metatron/G1/inference_last.ckpt \
  --inference-config examples/inference/metatron.py
```

`huggingface.json` pins the Hub commit and artifact-index checksum. The helper
needs only Python's standard library, downloads anonymously, verifies every
file's size/SHA256 and reuses verified files. About 460 MB is needed. It refuses
to overwrite unknown files: move a mismatched package aside before retrying.
Downloads go into the ignored `G1/` directory; do not commit them back to Git LFS.
Use `--output-dir /path/to/G1` for a different destination and adjust the
checkpoint argument. Keep all sidecars with the checkpoint.

`--configs-only` downloads everything except weights for configuration tests.
For the full native playback test, download the weights too and install MuJoCo.
Tests never download models implicitly. `experiment_config.py` is a Python file
that ProtoMotions imports and runs when it loads the checkpoint, so treat the Hub
package as trusted executable input; it is safe because the helper publishes only
the files recorded in `huggingface.json` by commit and checksum. Review changes
before updating the pinned revision.

The playback config starts the robot idle; pressing `X` in the viewer cycles
four goal modes (idle, walk to a target location, follow a velocity command,
reach with the right hand), and scene inputs are fed empty. Although the weights
retain scene conditioning, generic
native re-export does not recreate the browser scene pipeline. The separate
`demo/model.onnx` and `racer/network.onnx` remain on the same Hub repository and
require their paired browser runtimes. Their paths and bytes are unchanged.

Model artifacts use the NVIDIA Open Model License; source code is Apache-2.0.
Keep `LICENSE.pdf` and `NOTICE.txt` with redistributed model artifacts. Training
motions, recovery clips, robot meshes and furniture retain separate terms and
are not included in the Hub model package.
