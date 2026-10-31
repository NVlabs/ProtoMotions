# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression tests for the browser harness's worker acknowledgements."""

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "g1_racer_browser", ROOT / "scripts/check_g1_racer_browser.py"
)
browser = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(browser)


def frame(sequence=0, ticks=20, x=1, race_time=0.4, countdown=0):
    return {
        "sequence": sequence,
        "ticks": ticks,
        "state": {
            "race_time": race_time,
            "respawn_countdown": countdown,
            "positions": [[0, 0, 0], [0, 0, 0], [x, 2, 0.8]],
        },
    }


def test_inflight_tick_moves_before_paused_respawn_not_during_it():
    request = frame()
    before = frame(1, 21, 1.1, 0.42)
    after = frame(2, 21, 1.1, 0.42, -3)
    browser.assert_paused_respawn(request, before, after)


def test_already_quiescent_worker():
    browser.assert_paused_respawn(frame(), frame(), frame(1, countdown=-3))


def test_more_than_one_retiring_tick_still_fails():
    before = frame(2, 22, 1.2, 0.44)
    after = frame(3, 22, 1.2, 0.44, -3)
    with pytest.raises(AssertionError):
        browser.assert_paused_respawn(frame(), before, after)


@pytest.mark.parametrize(
    "change",
    [
        {"sequence": 0},
        {"ticks": 19},
        {"ticks": 22},
        {"x": 1.10001},
        {"race_time": 0.39},
        {"race_time": 0.43},
        {"countdown": 0},
    ],
)
def test_bad_respawn_still_fails(change):
    values = dict(sequence=2, ticks=21, x=1.1, race_time=0.42, countdown=-3)
    values.update(change)
    with pytest.raises(AssertionError):
        browser.assert_paused_respawn(frame(), frame(1, 21, 1.1, 0.42), frame(**values))


@pytest.mark.skipif(
    shutil.which("node") is None, reason="Node tests browser JS without GPU"
)
def test_observer_waits_for_fresh_same_tick_ack_and_keeps_atomic_samples():
    initial = frame()
    # An automatic respawn in the retiring live tick must not count as the
    # acknowledgement of our manual reset, even though its countdown is -3.
    live = frame(1, 21, 1.1, 0.42, -3)
    reset = copy.deepcopy(live)
    script = f"""
        const assert = require('node:assert/strict');
        const status = frame => ({{state: frame.state, timings: {{ticks: frame.ticks}}}});
        global.window = {{kartStatus: status({json.dumps(initial)})}};
        const observe = {browser.OBSERVE_PAUSE_FRAMES};
        const poll = {browser.WAIT_FOR_RESPAWN};
        observe();
        assert.equal(poll(0), false);
        window.kartStatus = status({json.dumps(live)});
        assert.equal(poll(0), false);
        window.kartStatus = status({json.dumps(reset)});
        const ack = poll(0);
        assert.equal(ack.before.ticks, 21);
        assert.equal(ack.after.ticks, 21);
        assert.equal(ack.before.state.positions[2][0], 1.1);
        assert.equal(poll(ack.after.sequence), false);
        window.kartStatus.state.positions[2][0] = 999;
        assert.equal(ack.after.state.positions[2][0], 1.1);
        for (let i = 0; i < 10; i++) window.kartStatus = status({json.dumps(reset)});
        assert.equal(window.__racerPauseFrames.length, 3);
        window.kartError = 'inference failed';
        assert.deepEqual(poll(0), {{error: 'inference failed'}});
    """
    subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)


@pytest.mark.skipif(
    shutil.which("node") is None, reason="Node tests diagnostic observer"
)
def test_failure_diagnostics_are_bounded_atomic_and_do_not_change_worker_inputs():
    script = f"""
        const assert = require('node:assert/strict');
        const vm = require('node:vm');
        let now = 0, heartbeat, observer;
        const snapshots = [], sent = [];
        class NativeWorker extends EventTarget {{ postMessage(m) {{ sent.push(m); }} }}
        const context = {{Worker: NativeWorker, structuredClone,
            performance: {{now: () => now}},
            document: {{visibilityState: 'visible', hasFocus: () => true,
                activeElement: {{tagName: 'BUTTON', id: 'pause'}}}},
            setInterval: (fn, ms) => {{assert.equal(ms, 1000); heartbeat = fn;}},
            PerformanceObserver: class {{constructor(fn) {{observer = fn;}} observe() {{}}}},
            __racerReport: value => {{snapshots.push(structuredClone(value)); return Promise.resolve();}}
        }};
        context.window = context;
        vm.runInNewContext({json.dumps(browser.OBSERVE_DIAGNOSTICS)}, context);
        const worker = new context.Worker();
        const input = {{type: 'pause'}};
        worker.postMessage(input);
        assert.equal(sent[0], input);
        let message;
        for (let i = 0; i < 40; i++) {{
            now += 1100;
            message = {{type: 'frame', timings: {{ticks: i, wall: i * 20}},
                state: {{time: i * .02, race_time: i * .02, controlled: 2,
                    positions: [[0,0,0],[0,0,0],[i,2,1]], respawn_countdown: 0}}}};
            worker.dispatchEvent(new MessageEvent('message', {{data: message}}));
            observer({{getEntries: () => [{{startTime: now, duration: 500}}]}});
        }}
        message.state.positions[2][0] = 999;
        message.timings.ticks = 999;
        now += 1100; heartbeat();
        const last = snapshots.at(-1);
        assert.equal(last.lastFrame.timings.ticks, 39);
        assert.equal(last.lastFrame.state.positions[2][0], 39);
        assert.equal(last.deliveries.length, 16);
        assert.equal(last.longTasks.length, 16);
        assert.equal(last.focused, true);
        assert.equal(last.visibility, 'visible');
        assert.equal(last.active, 'BUTTON:pause');
        now += 1100;
        worker.dispatchEvent(new MessageEvent('message', {{data: {{type: 'error', message: 'x'.repeat(3000)}}}}));
        assert.equal(snapshots.at(-1).workerError.length, 2000);
        const count = snapshots.length;
        heartbeat(); heartbeat();
        assert.equal(snapshots.length, count, 'at most one report per second');
    """
    subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)


def test_failure_uses_cached_snapshot_and_closes_browser_and_server(
    tmp_path, monkeypatch, capsys
):
    network = tmp_path / "demos/g1-racer/policy/network.json"
    network.parent.mkdir(parents=True)
    network.write_text('{"url": "https://example.invalid/pinned-model"}')
    closed, callbacks = [], {}

    def fail_navigation(_url):
        callbacks["console"](SimpleNamespace(text="last console message"))
        callbacks["pageerror"]("worker failed")
        raise RuntimeError("renderer stalled")

    # Deliberately no evaluate method: diagnostics must not contact the stalled page.
    page = SimpleNamespace(
        expose_binding=lambda _name, callback: callback(
            None, {"lastFrame": {"timings": {"ticks": 21}}, "focused": True}
        ),
        add_init_script=lambda _script: None,
        on=lambda event, callback: callbacks.update({event: callback}),
        goto=fail_navigation,
    )
    chromium = SimpleNamespace(
        launch=lambda **_kwargs: SimpleNamespace(
            new_page=lambda **_kwargs: page, close=lambda: closed.append("browser")
        )
    )
    monkeypatch.setitem(
        sys.modules,
        "playwright.sync_api",
        SimpleNamespace(
            sync_playwright=lambda: nullcontext(SimpleNamespace(chromium=chromium))
        ),
    )
    monkeypatch.setattr(
        browser,
        "ThreadingHTTPServer",
        lambda *_args: SimpleNamespace(
            server_port=12345,
            serve_forever=lambda: None,
            shutdown=lambda: closed.append("shutdown"),
            server_close=lambda: closed.append("server"),
        ),
    )
    monkeypatch.setattr(sys, "argv", ["check_g1_racer_browser.py", str(tmp_path)])
    with pytest.raises(RuntimeError, match="renderer stalled"):
        browser.main()
    report = json.loads(capsys.readouterr().out.split("RACER FAILURE DIAGNOSTICS ")[1])
    assert report["last_snapshot"]["lastFrame"]["timings"]["ticks"] == 21
    assert report["snapshot_age_seconds"] >= 0
    assert report["page_errors"] == ["worker failed"]
    assert report["console"] == ["last console message"]
    assert closed == ["browser", "shutdown", "server"]
