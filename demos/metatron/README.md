# Metatron Compiled Demo

This directory distributes a prebuilt browser demo. The application's source is
not published: this repository carries only its built output, not the TypeScript
project, development tooling or source maps. Browsers still download executable
JavaScript, WASM, the policy and motion assets; these are inspectable. This
packaging is not code secrecy or DRM.

`demo.zip` is tracked with Git LFS. `bundle.json` records its SHA256,
release status and complete file inventory. The bundle is the
static output of the separately maintained application. GitHub Pages verifies
and extracts it; it does not rebuild the application or install npm packages.

The FP32 unified scene/recovery policy is fetched anonymously from
[Hugging Face — demo](https://huggingface.co/nvidia/Metatron-v1/tree/main/demo),
not from Pages. The packaged manifest pins an immutable revision and SHA256;
the browser verifies downloaded and cached bytes before inference. No account
or token is needed. The archive contains the app, robot/motion assets and
notices, but no neural weights. The `racer/network.onnx` export in the same
Hugging Face repository, used by a separate racing demo, is batched for six
robots and is not interchangeable with `demo/model.onnx`.

## Preview

From the repository root:

```bash
git lfs pull --include="demos/metatron/demo.zip"
python scripts/install_metatron_demo.py --out /tmp/metatron-preview
python -m http.server 4188 --bind 127.0.0.1 --directory /tmp/metatron-preview
```

Open `http://127.0.0.1:4188/?task=ik`. All paths, including the threading service
worker scope, are relative, so the same bundle works under the documentation
site's `demos/metatron/` path. Serve over HTTPS or localhost, not `file://`.

## Licenses And Assets

Packaging does not change licenses. The archive retains the Apache-2.0 runtime
notice, pinned third-party notices, Unitree robot notice, NVIDIA Open Model
License, model card, and motion-use notices. The model license does not grant
rights to the motion datasets. Included motion/style previews are demonstration
assets, not a dataset release; see `motion-notice.html` inside the demo.
The recovery plan has its own `assets/recovery/NOTICE.txt` recording its source
and demo-only terms. It is not covered by the motion-preview notice.

The scene furniture is procedurally generated, not the original dataset meshes.
The current robot uses the standard Unitree head mesh. The bundle's readable
inventory includes these assets and all notices so reviewers can inspect what
will be served.
