# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check compiled football hosting on plain static HTTP, without loading weights.

The test intercepts background policy startup at the Worker boundary. It verifies
the preload request but not policy initialization, gameplay, WebGPU or iPhone QA.
"""

import argparse
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright


class StaticHandler(SimpleHTTPRequestHandler):
    """Deliberately adds no isolation headers; the scoped worker must do that."""

    def do_GET(self):
        self.server.request_paths.append(urlsplit(self.path).path)
        if self.path == "/favicon.ico":
            # This host-root browser request is outside the compiled app.
            # Supply a normal response: a synthetic 204 can fail reconstruction
            # in a service-worker fetch path despite no game resource failing.
            icon = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"/>'
            self.send_response(200)
            self.send_header("Content-Type", "image/svg+xml")
            self.send_header("Content-Length", str(len(icon)))
            self.end_headers()
            self.wfile.write(icon)
            return
        super().do_GET()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path, help="Contains demos/g1-football/index.html")
    parser.add_argument("--executable", help="Optional Chromium executable")
    parser.add_argument("--timeout", type=int, default=180, help="Startup seconds")
    args = parser.parse_args()
    entry = args.site / "demos/g1-football/index.html"
    if not entry.is_file():
        parser.error(f"Missing compiled entry point: {entry}")
    handler = partial(StaticHandler, directory=str(args.site.resolve()))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.request_paths = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    app = f"{origin}/demos/g1-football/"
    errors, missing, external, requests, failed = [], [], [], [], []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                executable_path=args.executable,
                args=["--no-sandbox", "--enable-unsafe-swiftshader"],
            )
            context = browser.new_context(viewport={"width": 1280, "height": 800})
            # Production preloads automatically, before the user clicks Start.
            # Explicitly isolate this lightweight hosting test from that costly
            # stage; do not ship a separate user-facing no-preload app mode.
            context.add_init_script("""(() => {
                window.footballPreviewPreloadRequests = [];
                const NativeWorker = window.Worker;
                window.Worker = class extends NativeWorker {
                    postMessage(message, ...args) {
                        if (message?.type === 'start') {
                            window.footballPreviewPreloadRequests.push(message);
                            return;
                        }
                        return super.postMessage(message, ...args);
                    }
                };
            })();""")

            def guard_request(route):
                url = route.request.url
                if urlsplit(url).scheme in {"http", "https"} and not url.startswith(
                    origin + "/"
                ):
                    external.append(url)
                    route.abort()
                else:
                    route.continue_()

            # A preview must be self-contained. Fail rather than inadvertently
            # download policy weights or hide a CDN dependency in a CI run.
            context.route("**/*", guard_request)
            # Context events include service-worker-owned requests, unlike
            # page-only routing. Also retain server-side paths below so a
            # worker interception cannot hide local model fetches.
            context.on("request", lambda request: requests.append(request.url))
            context.on(
                "requestfailed",
                lambda request: failed.append((request.url, request.failure)),
            )
            context.on(
                "response",
                lambda response: missing.append(response.url)
                if response.status >= 400
                else None,
            )
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on(
                "console",
                lambda message: errors.append((message.text, message.location))
                if message.type == "error"
                else None,
            )
            page.goto(app, wait_until="domcontentloaded")
            timeout = args.timeout * 1000
            # The isolation bootstrap may reload once. Wait on durable app
            # state, not the first page's load event or a network-idle guess.
            page.wait_for_function(
                "window.footballReady || window.footballError", timeout=timeout
            )
            assert not page.evaluate("window.footballError"), page.evaluate(
                "window.footballError"
            )
            page.wait_for_function(
                "window.footballKeeperView?.ready || "
                "window.footballKeeperView?.failed || window.footballError",
                timeout=timeout,
            )
            assert not page.evaluate("window.footballKeeperView.failed")
            assert page.evaluate("window.crossOriginIsolated") is True
            assert page.evaluate("window.footballRuntime.crossOriginIsolated") is True
            registration = page.evaluate(
                "async () => { const r = await navigator.serviceWorker.getRegistration();"
                "return {scope:r?.scope, controlled:!!navigator.serviceWorker.controller}; }"
            )
            assert registration == {"scope": app, "controlled": True}, registration
            assert "G1 Football" in page.title()
            assert page.locator("#start-game").inner_text() == "Start game"
            assert page.locator("#start-game").is_enabled()
            assert page.locator("#view").is_visible()
            assert page.locator("#keeper-view canvas").is_visible()
            assert "5 shots" in page.locator("#scoreboard").inner_text()
            page.wait_for_function(
                "window.footballKeeper?.previewFrames > 0",
                timeout=timeout,
            )
            assert page.evaluate("window.footballSimulationTime") == 0
            assert page.evaluate("window.footballRuntime.keeperConnected") is False
            assert page.evaluate("window.footballGame == null") is True
            page.wait_for_function(
                "window.footballPreviewPreloadRequests.length === 1", timeout=timeout
            )
            # Preloading is not consent to play. Input remains gated on Start.
            before = page.evaluate("window.footballControls.yaw")
            page.keyboard.down("d")
            page.wait_for_timeout(150)
            page.keyboard.up("d")
            assert page.evaluate("window.footballControls.yaw") == before
            assert not page.evaluate("window.footballError")
            external += [
                url
                for url in requests
                if urlsplit(url).scheme in {"http", "https"}
                and not url.startswith(origin + "/")
            ]
            assert not external, f"Preview requested external resources: {external}"
            assert not any(
                urlsplit(url).path.endswith(("weights.bin", ".onnx"))
                for url in requests
            ), "Preview must not fetch policy graphs or weights"
            assert not any(
                path.endswith(("weights.bin", ".onnx")) for path in server.request_paths
            ), "Static server received a policy request during preview"
            assert any(
                "pyodide" in url for url in requests
            ), "No local perception runtime"
            assert not errors, {"console": errors, "failed_requests": failed}
            assert not failed, failed
            assert not missing, missing
            print(
                "PASS G1 Football: static subpath, scoped isolation, scene, camera, local perception and background-preload request; policy start intercepted"
            )
            print(json.dumps(page.evaluate("window.footballRuntime"), sort_keys=True))
            browser.close()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
