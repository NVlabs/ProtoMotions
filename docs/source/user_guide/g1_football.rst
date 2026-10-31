G1 Football
===========

Take five penalty shots against a simulated Unitree G1 goalkeeper driven by
Metatron. MuJoCo physics, the keeper-mounted camera and ball perception run in
your browser. This is a simulation game, not a real-robot controller.

.. note::

   The compiled bundle is a public release (``release_status: public-release``).
   Its evidence is a bounded gameplay audit of 24 paired kicks and Chromium
   touch emulation; it is not a general save-rate estimate or a physical
   iPhone/Safari qualification. This is a simulation game, not a real-robot
   controller.

Playing
-------

The demo is deployed at ``demos/g1-football/`` beside the
documentation. The page loads the pitch, vision runtime and Metatron in sequence
and warms up the keeper in the background. The opening panel shows progress.
Click **Start game** to play immediately when ready, or to queue the start while
loading finishes. Readiness alone does not enable aiming or shooting.
The keeper inset is labeled **READY · PRESS START** while waiting. If loading
fails, **Reload to retry** restarts the page rather than partially restarting
the runtime.

Use **W / S** to aim higher or lower and **A / D** to aim left or right.
Hold **Space** to charge power, then release to shoot. On a touch screen,
drag the aim pad and hold/release the charge button. Each round has five shots;
use **Play again** after the round finishes. The keeper-camera panel shows the
camera feed used by the ball detector, not a prerecorded animation.

Runtime and download
--------------------

Opening the page downloads approximately 403 MB of unchanged FP32 policy weights
from `Metatron on Hugging Face
<https://huggingface.co/nvidia/Metatron-v1/tree/167355ae5630ebaece752545516da5217bf15d04/demo>`_.
The immutable revision and SHA256 checks are recorded in
``policy/manifest.json``. No account or token is needed. Small external-data
graphs ship with the compiled application; the large weights do not.

Verified weights can be reused from browser storage when quota permits. Cached
weights are checked again before use. Clearing site data, storage eviction or
insufficient quota can require another download. The mobile loading path uses
sliced cache verification and Blob URLs to reduce temporary memory copies,
but the model and its runtime still require substantial memory.

Automatic provider selection uses a suitable WebGPU adapter when available and
falls back to CPU/WASM. Software GPU adapters are not selected automatically.
Use ``?provider=wasm`` to select CPU inference explicitly. On a static host, a
service worker scoped to this demo adds cross-origin-isolation headers and may
reload the page once. Use HTTPS or localhost, not ``file://``.

Known limits
------------

Touch controls and mobile memory safeguards are implemented, but an actual
iPhone/Safari device has not been qualified. Browser preview CI checks hosting,
local perception, isolation and the background-preload request. It explicitly
intercepts policy initialization to avoid downloading weights; it does not
establish model readiness, mobile compatibility, gameplay quality or hardware
WebGPU performance.

The keeper sees only its own camera. Wide low or mid shots (a camera-derived
crossing at least 1.2 m to the side) trigger a goalkeeper dive: the keeper
shuffles and drops toward the ball line as early as its body allows, holds the
low pose until the ball arrives, then hands over to the native recovery. The
dive follows a reference clip derived from one BONES-SEED motion, credited in
the bundle's ``NOTICE.txt``. Nearer shots keep the native timed save, which
paired camera tests showed is stronger there; high corners (ball above 0.9 m at
the keeper) are not dived at because the keeper has no jumping reference. Over
24 paired kicks with identical camera noise, the dive saved 21 where the native
save made 16: all six 1.5 m medium-pace kicks became saves, two far-post slow
kicks were lost, near kicks were identical. Getting up after a dive is slower
than after the native save.
If the detector finds no plausible ball in its existing search window, it
retries with a smaller image-closing kernel. This avoids merging a small ball
with nearby white pitch markings. The retry uses the same camera pixels,
shape checks and uncertainty thresholds; it does not receive the physical
ball position or bypass trajectory confidence.
The keeper can still miss high or very wide shots and lose balance.
No public hardware-WebGPU speed claim is made. Performance depends on the
browser and device; inference can run slower than real time.

Licensing
---------

Applicable ProtoMotions-authored software retains Apache-2.0. This does not
relicense supplied game or perception components or grant rights beyond their
supplied notices. NVIDIA-provided exported models and weights are under the
NVIDIA Open Model License; release packages must include the model card, full
license and attribution notice. Runtimes and robot assets retain their own
terms. See the bundle's ``NOTICE.txt`` and ``licenses/`` directory. Compiled
browser assets remain downloadable and inspectable; compilation is not
encryption or DRM.

See ``demos/g1-football/README.md`` for local install instructions, the
evidence behind the release status and the dive reference's provenance.
