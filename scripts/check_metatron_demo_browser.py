# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test a compiled demo under a nested static Pages path, without app source."""

from __future__ import annotations

import argparse
import base64
import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


# The sim loop is built only after the policy promise resolves, so a load
# failure would otherwise surface as a timeout. Wait for the sim to start OR the
# loader to show an error, then let _assert_loaded fail fast with its message.
_BOOTED_OR_FAILED = (
    "window.__protomotionsDemo?.snapshot().simulationSeconds > 0.1"
    " || document.querySelector('#loader-message')?.classList.contains('error-message')"
)


def _assert_loaded(page) -> None:
    """Fail with the loader's own message if the policy did not load."""
    message = page.locator("#loader-message.error-message")
    if message.count():
        raise AssertionError(message.first.text_content() or "policy load failed")


def check_browser(site: Path, output: Path) -> None:
    """Check real simulation steps, rendering and layout in desktop/mobile sizes."""
    from playwright.sync_api import sync_playwright

    output.mkdir(parents=True, exist_ok=True)
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(site))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors, results, requests = [], [], []
    policy = json.loads(
        (site / "ProtoMotions/demos/metatron/assets/policy/manifest.json").read_text()
    )
    # Either the whole model or only its external weights is hosted; the
    # smoke test watches whichever URL the manifest pins.
    hosted = policy.get("model_download") or policy["external_data"]["download"]
    model_url = hosted["url"]
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                channel="chromium",
                args=[
                    "--no-sandbox",
                    "--use-gl=angle",
                    "--use-angle=swiftshader",
                    "--enable-unsafe-swiftshader",
                ],
            )
            page = browser.new_page(
                viewport={"width": 1440, "height": 1000}, has_touch=True
            )
            page.set_default_timeout(120_000)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("request", lambda request: requests.append(request.url))
            page.on(
                "response",
                lambda response: errors.append(response.url)
                if response.status >= 400
                else None,
            )
            url = f"http://127.0.0.1:{server.server_port}/ProtoMotions/demos/metatron/"
            page.goto(url + "?task=ik&provider=wasm&threads=1&debug=1&render_fps=1")
            page.wait_for_function(_BOOTED_OR_FAILED, timeout=300_000)
            _assert_loaded(page)
            page.evaluate("window.__protomotionsDemo.stepBudget(0)")
            for name, width, height in [("desktop", 1440, 1000), ("mobile", 390, 844)]:
                page.set_viewport_size({"width": width, "height": height})
                for task in ["ik", "scene", "steering", "location", "path", "tracking"]:
                    if (
                        name == "mobile"
                        and page.locator(".shell").get_attribute("data-mobile-panel")
                        != "task"
                    ):
                        page.locator('button[data-mobile-panel="task"]').click()
                    page.locator(f'[data-mode="{task}"]').click()
                    page.evaluate("window.__protomotionsDemo.stepBudget(4)")
                    page.wait_for_function(
                        "window.__protomotionsDemo.snapshot().remainingTestSteps === 0"
                    )
                    state = page.evaluate("window.__protomotionsDemo.snapshot()")
                    assert state["controlMode"] == task
                    assert page.evaluate(
                        "window.__protomotionsDemo.snapshot().qpos.every(Number.isFinite)"
                    )
                    assert page.evaluate(
                        "document.documentElement.scrollWidth <= innerWidth"
                    )
                    page.wait_for_timeout(1200)
                    png = page.locator("canvas").screenshot()
                    colors = page.evaluate(
                        """async encoded => {
                        const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
                        const image = await createImageBitmap(new Blob([bytes], {type: 'image/png'}));
                        const ctx = new OffscreenCanvas(image.width, image.height).getContext('2d');
                        ctx.drawImage(image, 0, 0);
                        const pixels = ctx.getImageData(0, 0, image.width, image.height).data;
                        const colors = new Set();
                        for (let i = 0; i < pixels.length; i += 16) colors.add(`${pixels[i]},${pixels[i+1]},${pixels[i+2]}`);
                        return colors.size;
                    }""",
                        base64.b64encode(png).decode(),
                    )
                    assert colors > 200, f"Blank canvas: {name}/{task}"
                    page.screenshot(path=output / f"{name}-{task}.png")
                    results.append(
                        {
                            "viewport": name,
                            "task": task,
                            "canvas_colors": colors,
                            "simulation_seconds": state["simulationSeconds"],
                        }
                    )
                    print(f"Passed {name}/{task}", flush=True)
            assert not errors, errors
            assert model_url in requests, "No anonymous HF model request observed"
            # The graph layout fetches the small bundled graph; only the hosted
            # file must never be served from Pages.
            bundled_weights = (
                "weights.bin" if "external_data" in policy else "model.onnx"
            )
            assert not any(
                f"/assets/policy/{bundled_weights}" in url for url in requests
            )

            # Second pass: the real desktop path. Without threads=1 the bootstrap
            # registers the COOP/COEP service worker, reloads once, and the reloaded
            # page is cross-origin isolated; the pinned model is its first
            # cross-origin fetch inside that path. The app's thread gate is
            # floor(hardwareConcurrency / 2), so a 2-CPU CI runner rounds to
            # single-threaded and never registers the worker; force >1 (the init
            # script re-runs on the shim's own reload, so both loads see it) to take
            # the isolation path a multi-core desktop visitor takes. Close the first
            # page first: two model sessions plus a Cache Storage copy overrun a
            # small CI runner.
            page.close()
            isolated_requests: list[str] = []
            isolated = browser.new_page(viewport={"width": 1440, "height": 1000})
            isolated.set_default_timeout(120_000)
            isolated.add_init_script(
                "Object.defineProperty(navigator, 'hardwareConcurrency', "
                "{get: () => 4})"
            )
            isolated.on(
                "request", lambda request: isolated_requests.append(request.url)
            )
            isolated.goto(
                url + "?task=ik&provider=wasm&debug=1&render_fps=1",
                wait_until="commit",
            )
            isolated.wait_for_function(_BOOTED_OR_FAILED, timeout=300_000)
            _assert_loaded(isolated)
            cores = isolated.evaluate("navigator.hardwareConcurrency")
            coi = isolated.evaluate("window.crossOriginIsolated")
            print(
                f"isolated pass: hardwareConcurrency={cores} crossOriginIsolated={coi}"
            )
            assert (
                coi is True
            ), "Page is not cross-origin isolated after the startup reload"
            assert (
                model_url in isolated_requests
            ), "No HF model request on the isolated threaded path"
            print("Passed isolated/ik", flush=True)
            isolated.close()
            browser.close()
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("output/compiled-demo-smoke")
    )
    args = parser.parse_args()
    check_browser(args.site, args.output)


if __name__ == "__main__":
    main()
