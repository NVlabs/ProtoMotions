# G1 Football

A five-shot penalty game against a simulated Unitree G1 keeper driven by
Metatron, with MuJoCo physics and a live keeper-mounted camera and ball detector.
This is a simulation game, not a real-robot controller.

## Release status

The compiled bundle is a **public release** (`release_status: public-release`
in `bundle.json` and the policy manifest). The installer accepts it without
`--allow-fixture`, and `--status` returns `true` for the Pages release gate.
The evidence behind this status is a bounded gameplay audit (24 paired kicks
across 8 kick types, Chromium touch emulation for phones); it is not a general
save-rate estimate or a physical iPhone/Safari qualification.

## Controls

The page loads and warms up the keeper in the background. Click **Start game**
to take five shots; if loading is unfinished, your click queues the start.
**W / S** aim higher/lower; **A / D** aim left/right. Hold **Space**
to charge and release to shoot. Touch controls provide a drag-to-aim pad and a
hold/release charge button. Select **Play again** after the round. The keeper
camera displays the live image used by perception.

The centered Start panel shows download and warmup progress, then a ready message.
The keeper inset is labeled **READY · PRESS START** while its prepared frame waits
for play. If initialization fails, **Reload to retry** reloads the whole page;
it does not start a second runtime alongside a partially initialized one.
Loading does not begin play: aiming and charging require both readiness and your
Start click. The power bar cycles from minimum to maximum and
back every 2.4 seconds; release selects that instant's power. Switching away
cancels the hold without shooting. The same controls work with CPU/WASM inference.
After each replay, the main camera returns to the ball for the next shot.
The ProtoMotions and Metatron header links open the public repository and project
page. Stadium chairs form solid-color NVIDIA eye mosaics; these are display-only
and are excluded from the keeper's sensor images and physics.

Opening the page automatically downloads approximately 403 MB from the immutable
[Metatron revision](https://huggingface.co/nvidia/Metatron-v1/tree/167355ae5630ebaece752545516da5217bf15d04/demo)
pinned by `policy/manifest.json`, without a token. SHA256-verified weights are
cached when storage quota permits and re-verified on reuse. Scene setup, vision
initialization and policy loading are sequenced to reduce simultaneous allocations
on mobile. Eviction or cleared
site data may require another download. The ZIP includes only the small
external-data graphs, never the large weight file.

The WebGPU path also loads approximately 95 MB of runtime and scene assets
(less for WASM-only inference). Browser caching reduces repeat downloads;
a cold load still needs both assets and weights.

## Local fixture preview

Download and install the checked-in candidate:

```bash
git lfs pull --include="demos/g1-football/game.zip"
python scripts/install_g1_football.py --allow-fixture --out /tmp/football-site/demos/g1-football
python -m http.server 4188 --bind 127.0.0.1 --directory /tmp/football-site
```

Open `http://127.0.0.1:4188/demos/g1-football/`. Use localhost or HTTPS, not a
`file://` URL. The demo-scoped service worker enables cross-origin isolation on
ordinary static hosting and may reload once. Automatic provider selection
prefers a suitable WebGPU adapter and rejects known software adapters; use
`?provider=wasm` for CPU inference. If GPU session initialization fails after
allocating its runtime, the app asks you to reload with `?provider=wasm` instead
of allocating another large runtime in the same worker. The mobile path disables
weight prepacking, uses one WASM thread and reduces cached-weight copying with
sliced hashing and Blob URLs. This does not remove the model's substantial
memory requirement.

The preview-only browser check intercepts the background policy-start message.
It verifies that the page requests preloading, but deliberately does not download
weights or qualify model initialization/gameplay:

```bash
python -m pip install playwright==1.55.0
python -m playwright install chromium
python scripts/check_g1_football_browser.py /tmp/football-site
```

Releases install without `--allow-fixture` (the flag is still accepted). Public
Pages deployment is restricted to the canonical repository and requires
`--status` to return `true`, which this bundle does.

## Known limits and licenses

- Actual iPhone/Safari devices have not been qualified; mobile controls and
  memory safeguards alone are not proof of compatibility. Chromium touch
  emulation passes in portrait and landscape: the pitch strip, keeper camera
  panel, start panel and thumb controls no longer overlap, and a charge pressed
  while the keeper camera is still preparing the next shot is accepted (its
  release is queued and fires as soon as the game is ready).
- The current keeper controller can miss high/lateral shots or lose balance.
  Two fresh camera estimates with clear left/right margins can trigger a bounded
  preparatory step before the crossing location is precise. The target remains
  capped at 0.65 m with the same slew limit. Timed save targets still require the
  original final trajectory commitment; neither stage receives hidden ball state.
  Paired camera tests showed earlier lateral motion on three of four side shots,
  with both center saves preserved, but all four near-post shots still scored.
  Wide low or mid shots trigger a goalkeeper dive driven by a reference clip, a
  derived excerpt of one BONES-SEED motion credited in NOTICE.txt and inside
  keeper-dive-clip.json. The dive uses only the camera-derived crossing: it
  shuffles and drops toward the ball line as early as the body allows, holds
  there until the ball arrives, then hands over to the native recovery. Shots
  within about 1.2 m of the keeper keep the native save, which paired camera
  tests showed is the stronger option there. Over 24 paired kicks (8 kick types,
  3 camera-noise seeds, same seeds for both keepers) the dive saved 21 where the
  native save made 16: all six 1.5 m medium-pace kicks became saves, two far-post
  slow kicks were lost, near kicks were identical. Getting back up after a dive
  can be slower than after the native save. High corners (ball above 0.9 m at the
  keeper) are not dived at; the keeper has no jumping reference. Experimental stances and velocity cues are not bundled.
- Small-ball detection retries the existing search window with less aggressive
  image closing when the original detector finds no plausible candidate. This
  helps avoid nearby white pitch markings being joined to the ball and rejected
  as a non-circular shape. Normal detections take the original path; camera noise,
  shape/volume checks, trajectory confidence and modeled latency are unchanged.
  The upstream detector is packaged unchanged, with a separate browser retry
  subclass; no simulator ball position or shot velocity enters perception.
  In a six-shot camera/physics check, the affected high-left shot was detected
  100 ms earlier and entered its timed save 300 ms earlier. The other five shots
  retained their timings and outcomes. The high-left shot still scored, and the
  earlier save tilted the keeper more; this is not a save-rate or no-fall claim.
- A committed-state query takes priority over the next queued camera analysis.
  This removes an avoidable wall-time wait on CPU/WASM; it does not shorten the
  modeled 60 ms sensor latency or change the final trajectory-confidence gate.
- Preview CI is a hosting/startup check, not gameplay or performance validation.
  There is no public hardware-WebGPU speed claim.
- Applicable ProtoMotions-authored software retains Apache-2.0. This does not
  relicense supplied game or perception components or grant rights beyond their
  supplied notices. NVIDIA-provided exported models and weights use the NVIDIA
  Open Model License. Include the model card, full license and attribution
  notice with the model package; runtime and robot-asset terms remain separate.
  See `NOTICE.txt` and `licenses/` in the bundle and
  [the repository model-licensing policy](../../legal/MODEL-LICENSING.md).

Only compiled application assets are packaged; no development sources or source
maps. Compilation does not make browser-delivered code or weights secret.
