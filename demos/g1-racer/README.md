# G1 Racer

Six simulated Unitree G1 robots race through one MuJoCo world in your browser,
all driven by a single batched Metatron policy. It is a simulation game with
movement assists, not a real-robot controller and not a claim about real
hardware.

## Play

Open the hosted page and click **Start race**. No Python, account or install is
needed. On the first visit the page downloads about 403 MB of policy weights
from Hugging Face. After the first verified download the browser keeps a copy in
its own site storage, so later visits in the same browser normally start without
downloading again. A different browser, cleared site data, or the browser
evicting the copy downloads it again; a stored copy is re-verified before use.
WebGPU gives the best experience; a slower CPU/WASM path runs where WebGPU is
absent. On static hosts without cross-origin-isolation headers the game
registers a service worker that adds them and reloads once, so the CPU path
runs on several WASM threads instead of one; `?threads=1` disables that.

You control **Green**, marked by an overhead arrow; the other five robots are
AI. After a short 3-2-1-GO countdown:

- **WASD** or **arrow keys** — drive
- **Space** — jump
- **Q** — throw your held item
- **R** — respawn in place (a three-second penalty applies)
- **P** — pause

Drive over floating **?** boxes to pick up a throwable Ball or Oil; the HUD
shows what you are holding. Speed-arrow pads give a boost, and clean jumps and
gap crossings grant a brief speed bonus. If your robot falls it tries to get up
on its own, and is rescued back onto the track if it stays stuck. Leaving the
road runs you into physical gravel, and marked zones lower gravity.

To keep the race fair, the game smooths each robot's measured speed over about
two seconds and never lets a robot be asked to go more than one metre per second
above that smoothed speed. It only limits the request; it never brakes, steers
or pushes a robot directly.

## Running it locally

```bash
git lfs pull --include="demos/g1-racer/game.zip"
python scripts/install_g1_racer.py --out /tmp/g1-racer
python -m http.server 4188 --bind 127.0.0.1 --directory /tmp/g1-racer
```

Open `http://127.0.0.1:4188/` in a WebGPU-capable browser. Serve over localhost
or HTTPS, not a `file://` URL. Add `?watch` to watch an AI-only race, or
`?provider=wasm` to force CPU inference.

## Licensing

- The game's own code (compiled JavaScript, WebAssembly and CPython bytecode) is
  licensed under **Apache-2.0**.
- The policy weights are **not** bundled here. The game downloads them from
  [Hugging Face](https://huggingface.co/nvidia/Metatron-v1) under the **NVIDIA
  Open Model License**, pinned to an exact revision and verified by SHA256
  before use (`policy/network.json`).
- The bundled motion assets — the jump reference, the gait cycles and the
  recovery pose plan — are **demo-only** data: included so the game runs, with no
  standalone redistribution or reuse rights. See `motions/NOTICE.txt`.
- Third-party browser-runtime packages keep their own upstream licenses; see
  `NOTICE.txt` and `licenses/`.

Maintainers of the development repository have a separate note on how the
`game.zip` bundle is rebuilt, verified, promoted and held back.
