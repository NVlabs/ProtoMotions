# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test the compiled game under a static-host subpath (no build tools)."""

import argparse
import json
import threading
import time
from contextlib import ExitStack
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Stream a bounded snapshot while the page is responsive. Failure reporting
# must not make another evaluate() call into a potentially stalled renderer.
OBSERVE_DIAGNOSTICS = """(() => {
    let lastFrame = null, workerError = null, lastPublish = -Infinity;
    const deliveries = [], longTasks = [];
    const append = (rows, value) => { rows.push(value); if (rows.length > 16) rows.shift(); };
    const publish = () => {
        const at = performance.now();
        if (at - lastPublish < 1000) return;
        lastPublish = at;
        window.__racerReport({at, lastFrame, workerError,
            error: window.kartError ?? null, deliveries, longTasks,
            visibility: document.visibilityState, focused: document.hasFocus(),
            active: document.activeElement?.tagName + ':' + document.activeElement?.id
        }).catch(() => {});
    };
    const NativeWorker = window.Worker;
    window.Worker = class extends NativeWorker {
        constructor(...args) {
            super(...args);
            this.addEventListener('message', ({data: m}) => {
                if (m.type === 'frame' && m.state) {
                    const s = m.state, receivedAt = performance.now();
                    lastFrame = structuredClone({receivedAt, timings: m.timings,
                        state: {time: s.time, race_time: s.race_time,
                            controlled: s.controlled, positions: s.positions,
                            respawn_countdown: s.respawn_countdown}});
                    append(deliveries, {receivedAt, ticks: m.timings.ticks, wall: m.timings.wall});
                }
                if (m.type === 'error') workerError = String(m.message).slice(0, 2000);
                publish();
            });
        }
    };
    if (globalThis.PerformanceObserver) {
        new PerformanceObserver(list => {
            for (const e of list.getEntries()) append(longTasks, {start: e.startTime, duration: e.duration});
        }).observe({type: 'longtask', buffered: true});
    }
    setInterval(publish, 1000);
})()"""

# Observe the existing frame publication, not the pause button (which changes
# before the worker has finished its current inference). Keep only adjacent
# atomic state/tick samples; no runtime or worker protocol changes are needed.
OBSERVE_PAUSE_FRAMES = """() => {
    let status = window.kartStatus;
    let sequence = 0;
    const sample = () => ({sequence: sequence++,
        ticks: status.timings.ticks, state: structuredClone(status.state)});
    const frames = [sample()];
    window.__racerPauseFrames = frames;
    Object.defineProperty(window, 'kartStatus', {
        configurable: true, enumerable: true,
        get: () => status,
        set: value => {
            status = value;
            frames.push(sample());
            if (frames.length > 3) frames.shift();
        }
    });
}"""

WAIT_FOR_RESPAWN = """requestSequence => {
    if (window.kartError) return {error: window.kartError};
    const frames = window.__racerPauseFrames;
    const after = frames.at(-1), before = frames.at(-2);
    // A completed tick increments ticks. The queued respawn publishes a new
    // snapshot WITHOUT incrementing ticks, after the paused loop drains.
    if (!before || after.sequence <= requestSequence ||
        after.ticks !== before.ticks || after.state.respawn_countdown !== -3)
        return false;
    return {before, after};
}"""


def assert_paused_respawn(request, before, after):
    """Separate the allowed final live tick from the stationary reset itself."""
    samples = {"request": request, "before": before, "after": after}
    assert after["sequence"] > request["sequence"], samples
    assert 0 <= after["ticks"] - request["ticks"] <= 1, samples
    assert after["state"]["race_time"] >= request["state"]["race_time"], samples
    assert before["ticks"] == after["ticks"], samples
    assert before["state"]["race_time"] == after["state"]["race_time"], samples
    assert after["state"]["respawn_countdown"] == -3, samples
    for old, new in zip(
        before["state"]["positions"][2][:2], after["state"]["positions"][2][:2]
    ):
        assert abs(old - new) < 1e-6, samples


def main():
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path, help="Contains demos/g1-racer/index.html")
    parser.add_argument("--executable", help="Optional Chromium executable")
    parser.add_argument(
        "--webgpu",
        action="store_true",
        help="Requires hardware GPU; not a CI emulation",
    )
    parser.add_argument("--ticks", type=int, default=20)
    args = parser.parse_args()
    model_url = json.loads(
        (args.site / "demos/g1-racer/policy/network.json").read_text()
    )["url"]
    handler = partial(SimpleHTTPRequestHandler, directory=str(args.site.resolve()))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    errors, missing, messages, requests = [], [], [], []
    diagnostic = {}

    def remember_diagnostic(_source, value):
        diagnostic.clear()
        diagnostic.update(value)
        diagnostic["received_at_monotonic"] = time.monotonic()

    flags = ["--no-sandbox"]
    if args.webgpu:
        flags += [
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan",
            "--use-angle=vulkan",
            "--disable-vulkan-surface",
        ]
    try:
        with sync_playwright() as playwright, ExitStack() as cleanup:
            browser = playwright.chromium.launch(
                headless=True, executable_path=args.executable, args=flags
            )
            cleanup.callback(browser.close)
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.expose_binding("__racerReport", remember_diagnostic)
            page.add_init_script(OBSERVE_DIAGNOSTICS)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: messages.append(m.text))
            page.on("request", lambda request: requests.append(request.url))
            page.on(
                "response", lambda r: missing.append(r.url) if r.status >= 400 else None
            )
            # Exercise normal automatic WebGPU selection at the regular URL.
            # CPU CI explicitly selects WASM, but gameplay has no EMA opt-in.
            query = "" if args.webgpu else "?provider=wasm"
            page.goto(
                f"http://127.0.0.1:{server.server_port}/demos/g1-racer/" f"{query}"
            )
            page.wait_for_function(
                "window.kartReady || window.kartError", timeout=240_000
            )
            assert not page.evaluate("window.kartError"), page.evaluate(
                "window.kartError"
            )
            assert "G1 Racer" in page.title()
            assert page.locator("#start").inner_text() == "Start race"
            assert page.locator(".start-controls").is_visible()
            assert page.locator(".action-controls kbd").all_text_contents() == [
                "SPACE",
                "Q",
            ]
            assert page.locator(".action-controls strong").all_text_contents() == [
                "Jump",
                "Use item",
            ]
            assert not page.locator("#landing-bonus").is_visible()
            assert page.locator('nav[aria-label="Game mode"]').count() == 0
            page.wait_for_function("window.kartCamera && window.kartStatus")
            assert page.evaluate("window.kartStatus.state.time") == 0
            assert page.evaluate("window.kartStatus.state.mode") == "battle"
            assert page.evaluate("window.kartStatus.state.boxes.length") == 12
            assert page.evaluate("window.kartStatus.timings.ticks") == 0
            assert not page.locator("#countdown").is_visible()
            runtime = page.locator("#runtime").inner_text()
            if args.webgpu:
                assert "WebGPU" in runtime
            else:
                # The bootstrap's service worker makes a static host
                # cross-origin isolated after one reload, so the CPU path gets
                # its WASM threads instead of running on one.
                assert page.evaluate("window.crossOriginIsolated") is True, runtime
                cores = page.evaluate("navigator.hardwareConcurrency")
                threads = int(runtime.split("·")[1].split()[0])
                assert threads == min(12, cores, max(1, cores // 2)), (runtime, cores)
            page.locator("#start").click()
            page.wait_for_function(
                f"window.kartStatus?.timings.ticks >= {args.ticks} || window.kartError",
                timeout=180_000,
                polling=100,
            )
            assert not page.evaluate("window.kartError"), page.evaluate(
                "window.kartError"
            )
            assert page.evaluate("window.kartStatus.state.controlled") == 2
            assert page.evaluate(
                "window.kartStatus.state.speed_ema.length === 6 && "
                "window.kartStatus.state.commands.every((cmd, i) => "
                "cmd.speed <= window.kartStatus.state.speed_ema[i] + 1.000001)"
            ), "Running racers must use the default EMA speed cap"
            assert page.evaluate(
                "window.kartStatus.state.recovering.length === 6 && "
                "window.kartStatus.state.recovering.every(v => typeof v === 'boolean')"
            )
            assert page.evaluate("window.kartEffects.marker")
            assert page.evaluate("window.kartEffects.boxes") > 0
            assert page.evaluate("window.kartLowGravity.zones") > 0
            assert page.evaluate("window.kartLowGravity.motes") > 0
            assert page.evaluate(
                "window.kartStatus.state.gravity_fractions.length === 6"
            )
            assert "Green · YOU" in page.locator("#ranking").inner_text()
            before_tab = page.evaluate("window.kartStatus.timings.ticks")
            page.keyboard.press("Tab")
            page.wait_for_function(
                f"window.kartStatus.timings.ticks > {before_tab} || window.kartError",
                timeout=30_000,
                polling=100,
            )
            assert not page.evaluate("window.kartError"), page.evaluate(
                "window.kartError"
            )
            assert page.evaluate("window.kartStatus.state.controlled") == 2
            # Exercise the supported keyboard control. Software WebGL on CPU
            # runners can delay the two animation frames that locator.click()
            # waits for stability, despite the page accepting input.
            page.evaluate(OBSERVE_PAUSE_FRAMES)
            page.keyboard.press("p")
            page.wait_for_function(
                "document.querySelector('#pause').textContent === 'Resume'",
                polling=100,
            )
            request = page.evaluate("window.__racerPauseFrames.at(-1)")
            page.keyboard.press("r")
            acknowledgement = page.wait_for_function(
                WAIT_FOR_RESPAWN,
                arg=request["sequence"],
                timeout=30_000,
                polling=100,
            )
            result = acknowledgement.json_value()
            acknowledgement.dispose()
            assert "error" not in result, result
            assert_paused_respawn(request, result["before"], result["after"])
            assert page.locator("#countdown-number").inner_text() == "3"
            assert page.locator("#countdown-label").inner_text() == "RESPAWN PENALTY"
            assert any("policy parity PASS" in m for m in messages)
            assert model_url in requests, "No anonymous HF network request observed"
            assert not any("/demos/g1-racer/network.onnx" in url for url in requests)
            assert page.evaluate(
                "window.kartStatus.state.commands.every((cmd, i) => "
                "cmd.speed <= window.kartStatus.state.speed_ema[i] + 1.000001)"
            )
            assert not errors, errors
            assert not missing, missing
            print(
                "PASS G1 Racer: static subpath, policy parity, startup, "
                "simulation, Green control and item boxes"
            )
            print(page.evaluate("JSON.stringify(window.kartStatus.timings)"))
    except Exception:
        received = diagnostic.get("received_at_monotonic")
        print(
            "RACER FAILURE DIAGNOSTICS "
            + json.dumps(
                {
                    "snapshot_age_seconds": time.monotonic() - received
                    if received is not None
                    else None,
                    "last_snapshot": diagnostic,
                    "page_errors": [str(e)[:2000] for e in errors[-20:]],
                    "console": [m[:2000] for m in messages[-20:]],
                    "missing": missing[-20:],
                }
            ),
            flush=True,
        )
        raise
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
