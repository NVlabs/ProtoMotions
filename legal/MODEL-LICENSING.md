# Model release licensing

ProtoMotions source code remains under [Apache-2.0](../LICENSE.md).
NVIDIA-provided pretrained model weights, checkpoints, serialized model
configurations and exported models are released under the **NVIDIA Open Model
License**, unless a model card explicitly specifies an approved exception.
This is a release policy for models distributed by this project, not a license
imposed on every model that someone trains using the Apache-licensed code.

The bundled [agreement](NVIDIA-Open-Model-License.pdf) is the unmodified
[official NVIDIA PDF](https://www.nvidia.com/content/dam/en-zz/Solutions/license-agreements/enterprise-software/nvidia-open-model-license-agreements-24-10-2025.pdf),
version October 24, 2025. The [official license page](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
provides an HTML version. Source files, including model architectures and
experiment scripts, keep their Apache SPDX headers. Third-party datasets,
robot assets and other separately licensed components retain their own terms.
This policy does not revoke rights previously granted under another license.

## Publishing a model

Keep each model under `data/pretrained_models/<family>/<model>/` and include:

1. A `MODEL_CARD.md` with a `## License` section containing
   `Model license: NVIDIA Open Model License`.
2. A byte-for-byte copy of `legal/NVIDIA-Open-Model-License.pdf` named
   `LICENSE.pdf` beside that card.
3. A `NOTICE.txt` containing the attribution:
   `Licensed by NVIDIA Corporation under the NVIDIA Open Model License`.
4. Links to `LICENSE.pdf` and `NOTICE.txt` from the card's License section.
   Keep the card's training-data provenance and applicable third-party notices.

Include the license, notice and model card in standalone downloads and
exported-model packages too; a weights-only or ONNX-only upload is not a complete
release. Preserve applicable upstream terms when distributing fine-tuned models,
adapters or derivatives; do not infer dataset redistribution rights from the
model license.

Run `python scripts/check_model_licenses.py` before publishing. CI runs this
without loading checkpoints or downloading LFS payloads, and the release
projection checks the actual outgoing tree.

## Explicit exceptions

Never silently relabel third-party weights. A maintainer-reviewed exception
must specify the alternative license and its rationale in both the card and
`legal/model-license-exceptions.json`. The JSON maps the model directory
(relative to the repository) to `license`, `license_file`, and `reason`.
The card uses `Model license: <license>` and
`License exception: <reason>` in its License section. Bundle the full alternative
license under the specified filename and a nonempty `NOTICE.txt`, linking both
from that section. No current public model has an exception.
