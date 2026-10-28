Metatron Browser Demo
=====================

The Metatron demo runs a G1 policy and MuJoCo simulation in the browser. Its
tasks include steering, location, path following, motion tracking, scene
interaction and interactive IK.

Open `the demo <../demos/metatron/?task=ik>`_. No Python installation is needed.
The first visit downloads the unified scene/recovery policy from
`Hugging Face <https://huggingface.co/nvidia/Metatron-v1/tree/main/demo>`_
and the remaining assets from the site. No account or token is needed. The
manifest pins an immutable revision and SHA256; verified model bytes are cached
when browser storage permits. Speed
depends on browser and device; WebGPU is used where supported, with a WASM path
for other devices. Mobile performance is not guaranteed by desktop emulation.

Distribution
------------

The repository carries the compiled static bundle, not the application's
TypeScript project or source maps. GitHub Pages verifies and extracts that
bundle beside the Sphinx documentation; it does not compile the application.
Executable JavaScript, WASM and model assets must still be downloaded by the
browser and remain inspectable.

Local preview instructions are in ``demos/metatron/README.md``. The bundle
checksum, file inventory and release status are recorded in
``demos/metatron/bundle.json``.

Usage And Notices
-----------------

The demo is a simulated research interface, not a hardware controller or safety
system. Task completion, precise IK and recovery are not guaranteed for arbitrary
goals or fallen states.

Code, third-party runtime libraries, robot assets, policy weights and motion
assets retain their respective licenses and notices in the bundle. Demo motions
are not licensed for reuse through the code or model license; consult the
in-demo motion notice and the original rights holders.
