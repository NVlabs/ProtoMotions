G1 Racer
========

Six simulated Unitree G1 robots race through one MuJoCo world in your browser,
all driven by a single batched Metatron policy. It is a simulation game with
movement assists, not a real-robot controller.

.. note::

   The bundled motion assets — the jump reference, the gait cycles and the
   recovery pose plan — are demo-only data, included so the game runs; they carry
   no standalone redistribution or reuse rights. Publishing the compiled game
   grants no additional rights to them.

Playing
-------

Open `G1 Racer <../demos/g1-racer/>`_ and click **Start race**. No Python,
account or install is needed. On the first visit the page downloads about
403 MB of policy weights from Hugging Face. After the first verified download the
browser keeps a copy in its own site storage, so later visits in the same browser
normally start without downloading again; a different browser, cleared site data,
or the browser evicting the copy downloads it again, and a stored copy is
re-verified before use. WebGPU gives the best experience; a slower CPU/WASM path
runs where WebGPU is unavailable.

You control **Green**, marked by an overhead arrow; the other five robots are
AI. After a short 3-2-1-GO countdown, use **WASD** or the **arrow keys** to
drive, **Space** to jump, **Q** to throw a held item, **R** to respawn in place
(a three-second penalty), and **P** to pause. Drive over floating **?** boxes to
pick up a throwable Ball or Oil. Speed-arrow pads boost you, and clean jumps and
gap crossings grant a brief speed bonus. A fallen robot tries to get up on its
own and is rescued back onto the track if it stays stuck. Leaving the road runs
you into physical gravel, and marked zones lower gravity.

To keep the race fair, the game smooths each robot's measured speed over about
two seconds and never lets a robot be asked to go more than one metre per second
above that smoothed speed. It only limits the request; it never brakes, steers
or pushes a robot directly.

How it runs
-----------

The policy weights are fetched from
`Hugging Face <https://huggingface.co/nvidia/Metatron-v1/tree/6d5932084bf06055569c5ed39af2707ee20cf770/racer>`_
at a pinned, immutable revision and verified by SHA256 before use, with no login
or token. The compiled observation code travels inside the game bundle. The browser
loads these assets and initialises its device-specific runtime at startup; it
does not export or rebuild anything.

Licensing
---------

The game's own code (compiled JavaScript, WebAssembly and CPython bytecode) is
licensed under Apache-2.0. The policy weights are not bundled; they download from
Hugging Face under the NVIDIA Open Model License. The bundled motion assets are
demo-only data, as noted above. Browser code and weights remain downloadable and
inspectable; this is not encryption or DRM.

See ``demos/g1-racer/README.md`` to run the game locally. Maintainers of the
development repository have a separate note on how the bundle is rebuilt and
verified.
