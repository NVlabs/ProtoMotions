# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test the compiled game under a static-host subpath (no build tools)."""

import argparse
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
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
    handler = partial(SimpleHTTPRequestHandler, directory=str(args.site.resolve()))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    errors, missing, messages, requests = [], [], [], []
    model_url = json.loads(
        (args.site / "demos/g1-racer/policy/network.json").read_text()
    )["url"]
    flags = ["--no-sandbox"]
    if args.webgpu:
        flags += [
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan",
            "--use-angle=vulkan",
            "--disable-vulkan-surface",
        ]
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True, executable_path=args.executable, args=flags
            )
            page = browser.new_page(viewport={"width": 1280, "height": 720})
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
            if args.webgpu:
                assert "WebGPU" in page.locator("#runtime").inner_text()
            page.locator("#start").click()
            page.wait_for_function(
                f"window.kartStatus?.timings.ticks >= {args.ticks} || window.kartError",
                timeout=180_000,
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
                f"window.kartStatus.timings.ticks > {before_tab}", timeout=30_000
            )
            assert page.evaluate("window.kartStatus.state.controlled") == 2
            # Exercise the supported keyboard control. Software WebGL on CPU
            # runners can delay the two animation frames that locator.click()
            # waits for stability, despite the page accepting input.
            page.keyboard.press("p")
            page.wait_for_function(
                "document.querySelector('#pause').textContent === 'Resume'"
            )
            # Let the in-flight control tick drain before sampling the paused
            # position. Manual reset must not rewind the robot or race clock.
            page.wait_for_timeout(500)
            before_respawn = page.evaluate("window.kartStatus.state")
            before_ticks = page.evaluate("window.kartStatus.timings.ticks")
            page.keyboard.press("r")
            page.wait_for_function("window.kartStatus.state.respawn_countdown === -3")
            after_respawn = page.evaluate("window.kartStatus.state")
            after_ticks = page.evaluate("window.kartStatus.timings.ticks")
            # A paused reset must not rewind the race clock. On a slow CPU/WASM
            # runner one in-flight control tick (50 Hz) can still retire between
            # the two samples, so allow the clock to advance by at most that one
            # tick but never to rewind.
            assert after_respawn["race_time"] >= before_respawn["race_time"]
            assert 0 <= after_ticks - before_ticks <= 1
            for before, after in zip(
                before_respawn["positions"][2][:2], after_respawn["positions"][2][:2]
            ):
                assert abs(before - after) < 1e-6
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
            browser.close()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
