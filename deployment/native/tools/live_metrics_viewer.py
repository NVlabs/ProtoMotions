# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live time-series viewer for the native launcher's metrics stream.

This is a consumer and nothing more: it reads the launcher's JSON frames
(``--udp`` live, as the launcher starts it; ``--file`` for a record), keeps
them in ring buffers on a reader thread, and redraws on the main thread. It
has no channel back to the launcher; if it stalls or dies, the launcher only
drops UDP datagrams and keeps controlling the robot.

Keys are ``group/name``; vectors arrive as ``name/0``, ``name/1``, ... Tabs and
panels are built from the keys that actually arrive: a tab appears once one of
its keys has been seen, a panel once it has data. The task tab is titled from
the stream's ``task/name`` and lays out whatever task keys arrive, so another
policy or task gets its own tab without changes here. A curated layout exists
for the tasks we ship; everything else gets the generic one.

    python tools/live_metrics_viewer.py --udp 0.0.0.0:9870
    python tools/live_metrics_viewer.py --file run/metrics.jsonl --screenshot out.png
    python tools/live_metrics_viewer.py --udp 0.0.0.0:9870 --duration 20 --screenshot out.png
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

HISTORY_SECONDS = 20.0
REDRAW_HZ = 10.0
TEMPERATURE_PATTERNS = ["motor/max_winding_c", "motor/max_driver_c", "motor/winding_c/", "motor/driver_c/"]
# The producer reports this family only where a temperature sensor exists, so
# any of these keys, even a zero count, means "supported": a reading that then
# never arrives is a fault to show, not a feature to hide.
TEMPERATURE_FAMILY = TEMPERATURE_PATTERNS + ["motor/temperature_count", "motor/temperature_valid"]
# Header and status fields: strings or enumerations, never plotted.
STATUS_KEYS = {"task/mode", "task/mode_name", "task/name", "task/active", "task/recovery_phase",
               "policy/name", "joint/names", "limits/nearest_joint", "motor/temperature_valid", "motor/temperature_count",
               "tracking/available", "tracking/selected", "tracking/motion"}
# Key families a task may publish under. Anything here that is not a status key
# belongs to the task tab.
TASK_FAMILIES = ["task/", "goal/", "ik/", "support/", "tracking/"]
# Unit labels only for suffixes that mean one thing; anything else stays
# unlabelled rather than guessed.
UNIT_SUFFIXES = [("_deg", "deg"), ("_rad", "rad"), ("_ms", "ms"), ("_m", "m")]

THERMAL = ("Temperature · winding / driver maxima", ["motor/max_winding_c", "motor/max_driver_c"], "°C")
# Core tabs: name, label, panels (title, key patterns, unit), gate patterns.
# A tab is shown once a key matching its gate has arrived; the overview always.
# Patterns: an exact key, a prefix ending in "/", or "stem/#" for the numeric
# components of a vector only (so goal/left/enabled never lands on a metres axis).
# The overview carries only runtime state every task has; a task's own summary
# panels come from its preset.
CORE_TABS = [
    ("overview", "Overview", [
        ("Torso height (world)", ["pose/torso_z"], "m"),
        ("Torso tilt", ["pose/tilt_deg"], "deg"),
        ("Mechanical power", ["power/mechanical_w"], "W"),
        ("Joint jerk · RMS over joints", ["joint/jerk_rms"], "rad/s³"),
        ("Control compute time", ["cycle/compute_ms", "cycle/policy_ms"], "ms"),
        THERMAL], []),
    ("joints", "Joints", [
        ("Joint positions", ["joint/q/"], "rad"),
        ("Joint velocities", ["joint/dq/"], "rad/s"),
        ("Joint torques", ["joint/torque/"], "N·m"),
        ("Joint jerk", ["joint/jerk/"], "rad/s³"),
        ("Minimum joint-limit margin", ["limits/min_margin_rad"], "rad"),
        ("Joints near limits", ["limits/joints_near_limit"], "count")], ["joint/", "limits/"]),
    ("thermal", "Thermal", [THERMAL, ("All motors · current winding / driver temperatures", [], "°C")],
     TEMPERATURE_FAMILY),
    ("timing", "Timing & comms", [
        ("Control cycle breakdown", ["cycle/compute_ms", "cycle/read_ms", "cycle/observe_ms",
                                     "cycle/policy_ms", "cycle/send_ms"], "ms"),
        ("State age", ["cycle/state_age_ms", "comm/lowstate_age_ms", "comm/odometry_age_ms"], "ms"),
        ("Communication errors", ["comm/missed_states", "comm/crc_errors"], "count"),
        ("Held control cycles", ["cycle/held"], "count"),
        ("Received hardware packets", ["comm/lowstate_packets"], "count")], ["cycle/", "comm/"]),
    # No charts: the operator's settings (steering caps, the motion to play); shown only with a settings file.
    ("settings", "Settings", [], []),
]
# Curated presets by the stream's exact task/name: presentation and known units
# only, never control semantics. "overview" panels join the overview tab,
# "task" panels lead the task tab; any task key a preset does not consume still
# gets a fallback panel, so an incomplete preset hides nothing.
TASK_PRESETS = {
    "Metatron v1": {
        "overview": [
            ("Task: heading error (deg)", ["task/heading_error_deg"], "deg"),
            ("Task: speed (m/s), requested vs measured, and velocity error",
             ["task/speed_requested", "task/speed_measured", "task/velocity_error_x", "task/velocity_error_y"], "m/s"),
            ("Hands: goal error (m)", ["task/hand_error_left_m", "task/hand_error_right_m"], "m"),
            ("Torso height · target / measured, support-relative", ["task/torso_height_goal_m", "task/torso_height_m"], "m"),
        ],
        "task": [
            ("Hands: goal error (m)", ["task/hand_error_left_m", "task/hand_error_right_m"], "m"),
            ("Arm IK correction", ["ik/offset_left_rad", "ik/offset_right_rad"], "rad"),
            ("Arm IK solve time", ["ik/solve_ms"], "ms"),
            ("Prompt pre-emphasis bias", ["task/prompt_bias_left_m", "task/prompt_bias_right_m"], "m"),
            ("Foot displacement from support", ["task/support_slip_m"], "m"),
            ("Re-plants", ["task/replants"], "count"),
            ("Torso height · target / measured, support-relative", ["task/torso_height_goal_m", "task/torso_height_m"], "m"),
            ("Hand goals · world position", ["goal/left/#", "goal/right/#"], "m"),
            ("Hand goals · enabled", ["goal/left/enabled", "goal/right/enabled"], "flag"),
            ("Support frame · centre", ["support/center/#"], "m"),
            ("Support frame · yaw", ["support/yaw"], "rad"),
            ("Support frame · planted", ["support/valid"], "flag"),
            ("Heading error", ["task/heading_error_deg"], "deg"),
            ("Speed and velocity error", ["task/speed_requested", "task/speed_measured",
                                          "task/velocity_error_x", "task/velocity_error_y"], "m/s"),
            ("Tracking · body and anchor-height error", ["tracking/mean_body_error_m", "tracking/anchor_height_error_m"], "m"),
            ("Tracking · rotation error", ["tracking/mean_rotation_error_deg"], "deg"),
            ("Tracking · progress", ["tracking/progress"], "fraction"),
            ("Tracking · elapsed and duration", ["tracking/elapsed_s", "tracking/duration_s"], "s"),
            ("Tracking · reference positions", ["tracking/ref/0/#", "tracking/ref/1/#", "tracking/ref/2/#",
                                                 "tracking/ref/3/#", "tracking/ref/4/#"], "m"),
        ],
    },
}
TAB_ORDER = ["overview", "task", "joints", "thermal", "timing", "settings"]
BACKGROUND, PANEL, GRID, TEXT, MUTED = "#0B1119", "#111D29", "#253341", "#D9E0E7", "#8A9BAD"
RED = "#FB7185"  # past what was qualified; the same red the alert line uses
HEADER, TILE = "#0E1824", "#121F2D"   # the top bar and the stat tiles sit a shade above the page
# The brand green marks the active tab, the playing motion and the wordmark;
# the data lines use muted, evenly bright hues so an hour at the screen is easy.
ACCENT = "#76B900"
PALETTE = ["#7FA7CC", "#C9A46A", "#A596CF", "#7DB39E", "#C98A8A", "#8FA3B5"]


class Stream:
    """Thread-safe ring buffers keyed by metric name, filled by a reader thread."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.lock = threading.Lock()
        self.series: dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=20000))
        self.frames = 0
        self.last_frame: dict = {}
        self.last_received = 0.0

    def push(self, frame: dict) -> None:
        t = frame.get("t_ns", 0) * 1e-9
        with self.lock:
            self.frames += 1
            self.last_frame = frame
            self.last_received = time.monotonic()
            for key, buffer in self.series.items():
                if matches(key, TEMPERATURE_PATTERNS) and key not in frame and buffer and math.isfinite(buffer[-1][1]):
                    buffer.append((t, math.nan))  # do not connect a later reading across sensor dropout
            for key, value in frame.items():
                if key in ("t_ns", "cycle"):
                    continue
                if value is not None and not isinstance(value, (int, float)):
                    continue  # string metadata belongs in last_frame, not numeric history
                self.series[key].append((t, math.nan if value is None else float(value)))

    def snapshot(self) -> tuple[dict[str, np.ndarray], dict, float]:
        with self.lock:
            out = {}
            cutoff = self.last_frame.get("t_ns", 0) * 1e-9 - self.seconds
            for key, buffer in self.series.items():
                while buffer and buffer[0][0] < cutoff:
                    buffer.popleft()
                # Empty arrays clear old plotted lines when a channel expires.
                out[key] = np.asarray(buffer, dtype=float).reshape(-1, 2)
            return out, dict(self.last_frame), self.last_received


def udp_reader(stream: Stream, host: str, port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
    sock.bind((host, port))
    sock.settimeout(0.5)
    while True:
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        try:
            stream.push(json.loads(data.decode()))
        except (ValueError, UnicodeDecodeError):
            continue


def file_reader(stream: Stream, path: Path, realtime: bool) -> None:
    previous = None
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            if realtime and previous is not None:
                time.sleep(max(0.0, (frame["t_ns"] - previous) * 1e-9))
            previous = frame["t_ns"]
            stream.push(frame)


def matches(key: str, patterns: list[str]) -> bool:
    for p in patterns:
        if p.endswith("/#"):
            stem, _, tail = key.rpartition("/")
            if stem == p[:-2] and tail.isdigit():
                return True
        elif key == p or (p.endswith("/") and key.startswith(p)):
            return True
    return False


def is_task_key(key: str) -> bool:
    return matches(key, TASK_FAMILIES) and key not in STATUS_KEYS


def vector_base(key: str) -> str:
    """``goal/left/2`` -> ``goal/left``; a scalar key is its own base."""
    head, _, tail = key.rpartition("/")
    return head if tail.isdigit() and head else key


def unit_of(key: str) -> str:
    for suffix, unit in UNIT_SUFFIXES:
        if key.endswith(suffix):
            return unit
    return ""


def generic_task_panels(keys) -> list[tuple[str, list[str], str]]:
    """Panels for task keys nobody curated: the numeric components of a vector
    share a panel, every other key gets its own. Equal units do not imply equal
    meaning or frame, so nothing is merged on a unit guess."""
    panels, vectors = [], set()
    for key in sorted(k for k in keys if is_task_key(k)):
        base = vector_base(key)
        if base != key:
            if base not in vectors:
                vectors.add(base)
                panels.append((base, [base + "/#"], unit_of(base)))
            continue
        panels.append((key, [key], unit_of(key)))
    return panels


def tab_visible(name: str, seen) -> bool:
    if name in ("overview", "thermal", "timing"):
        return True  # always: an operator looks for temperature and timing whether or not this stream has them yet
    if name == "task":
        return any(is_task_key(k) for k in seen)
    gate = next(g for n, _, _, g in CORE_TABS if n == name)
    return any(matches(k, gate) for k in seen)


def motor_count(seen, last: dict, declared: int = 0) -> int:
    """How many motors the per-motor bars should show: the larger of what the
    stream has named and what it has declared (``declared`` is the largest
    temperature_count seen so far, kept by the caller so a partial frame hatches
    the missing motors instead of dropping them). Never a robot-specific constant."""
    indices = [int(k.rsplit("/", 1)[1]) for k in seen
               if k.startswith(("motor/winding_c/", "motor/driver_c/")) and k.rsplit("/", 1)[1].isdigit()]
    named = max(indices) + 1 if indices else 0
    count = last.get("motor/temperature_count")
    current = int(count) if isinstance(count, (int, float)) and math.isfinite(count) and count > 0 else 0
    return max(named, declared, current)


def temperature_age(last: dict, receive_age: float) -> float:
    """A fresh UDP frame may still contain a stale hardware state."""
    age = receive_age
    for key in ("cycle/state_age_ms", "comm/lowstate_age_ms"):
        if key in last:
            value = last[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                return math.inf
            age = max(age, receive_age + value / 1000.0)
    return age


def temperature_status(last: dict, age: float, caution: float, high: float) -> tuple[str, str]:
    """Use this frame only: historical readings never imply current cooling.
    The hottest sensor is named after its joint when the stream carries names."""
    if temperature_age(last, age) >= 0.5:
        return "TEMPERATURE STALE — check telemetry", "#735416"
    values = {k: float(v) for k, v in last.items()
              if matches(k, TEMPERATURE_PATTERNS) and isinstance(v, (int, float)) and math.isfinite(v)}
    if not values:
        return "TEMPERATURE UNAVAILABLE — no sensor data", "#555555"
    key = max(values, key=values.get)
    maximum = values[key]
    # Prefer the per-motor sensor that carries the maximum over the aggregate:
    # it says which motor, the aggregate only says how hot.
    for candidate, value in values.items():
        if value == maximum and candidate.rsplit("/", 1)[1].isdigit():
            key = candidate
            break
    partial = last.get("motor/temperature_valid") == 0 or any(
        matches(k, TEMPERATURE_PATTERNS) and (v is None or not math.isfinite(float(v)))
        for k, v in last.items())
    suffix = " · PARTIAL telemetry" if partial else ""
    if not any("driver_c" in k for k in values):
        suffix += " · driver unavailable"
    source = key.removeprefix("motor/")
    names = joint_names(last)
    kind, _, index = source.partition("_c/")
    if index.isdigit() and int(index) < len(names):
        source = f"{pretty_joint(names[int(index)])} {kind}"
    if maximum >= high:
        return f"HIGH TEMPERATURE: {maximum:.1f} °C ({source}){suffix}", "#B91C1C"
    if maximum >= caution:
        return f"TEMPERATURE CAUTION: {maximum:.1f} °C ({source}){suffix}", "#9A5800"
    return f"Max temperature: {maximum:.1f} °C ({source}){suffix}", "#555555"


def joint_names(last: dict) -> list[str]:
    """Names from the stream's ``joint/names`` (comma-joined, or a list in older
    fixtures), in joint order, which is also the motor order on hardware."""
    raw = last.get("joint/names")
    if isinstance(raw, str) and raw:
        return raw.split(",")
    if isinstance(raw, list):
        return [str(n) for n in raw]
    return []


def pretty_joint(name: str) -> str:
    return name.removesuffix("_joint").replace("_", " ")


# Legend names for the cycle stages; the key tails alone would not say that the
# command send is where MuJoCo steps its physics.
STAGE_LABELS = {
    "cycle/compute_ms": "whole cycle",
    "cycle/read_ms": "sensor read",
    "cycle/observe_ms": "observation build (goals + FK)",
    "cycle/policy_ms": "policy engine",
    "cycle/send_ms": "command send / sim step",
}


def channel_label(key: str, names: list[str]) -> str:
    """``joint/q/3`` -> the joint's name when the stream has one, else the key's tail."""
    if key in STAGE_LABELS:
        return STAGE_LABELS[key]
    head, _, tail = key.rpartition("/")
    if tail.isdigit() and head.startswith(("joint/", "motor/")):
        index = int(tail)
        if index < len(names):
            return pretty_joint(names[index])
    return key.split("/", 1)[-1]


def jitter_leaders(data: dict, names: list[str], window_s: float = 1.0, top: int = 3,
                   family: str = "joint/dq/") -> list[tuple[str, float]]:
    """Joints whose channel in ``family`` varies most over the last second:
    velocity for jitter, jerk for roughness."""
    leaders = []
    for key, series in data.items():
        if not key.startswith(family) or not key.rsplit("/", 1)[1].isdigit() or len(series) < 5:
            continue
        recent = series[series[:, 0] >= series[-1, 0] - window_s, 1]
        recent = recent[np.isfinite(recent)]
        if len(recent) >= 5:
            leaders.append((channel_label(key, names), float(np.sqrt(np.mean(recent ** 2)) if family.endswith("jerk/") else np.std(recent))))
    return sorted(leaders, key=lambda item: -item[1])[:top]


class SettingsWriter:
    """The one channel from the dashboard back to the launcher: a small JSON file
    of operator settings, replaced atomically (write-then-rename) on every change
    so the launcher never reads half a document. Settings are state, not events:
    a value stays until the next change and survives a dashboard restart, which
    is why the file is read back at start (the launcher removes it when it
    starts, so a new run begins from the defaults). Which numeric keys exist, their
    defaults and ranges come from the launcher's stream (``settings/schema``),
    declared by the task that owns them; the writer itself knows no keys, and the
    launcher caps and reports what it applied (``settings/<key>``)."""

    def __init__(self, path):
        self.path = Path(path)
        self.version = 0
        self.values = {"selected_motion": ""}
        self.last_error = ""
        try:
            saved = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        if not isinstance(saved, dict):
            return
        if isinstance(saved.get("version"), int):
            self.version = saved["version"]
        for key, value in saved.items():
            if key == "selected_motion" and isinstance(value, str):
                self.values[key] = value
            elif key != "version" and isinstance(value, (int, float)) and not isinstance(value, bool) \
                    and math.isfinite(value) and value >= 0:
                self.values[key] = float(value)

    def set(self, **changes) -> bool:
        changed = {k: v for k, v in changes.items() if self.values.get(k) != v}
        if not changed:
            return False
        self.values.update(changed)
        self.version += 1
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            temporary.write_text(json.dumps({"version": self.version, **self.values}, indent=1) + "\n")
            os.replace(temporary, self.path)
            self.last_error = ""
        except OSError as error:
            self.last_error = str(error)  # the dashboard keeps running; the tab shows the failure
        return True


def mode_label(last: dict) -> str:
    name = last.get("task/mode_name")
    if isinstance(name, str) and name:
        return name
    mode = last.get("task/mode")
    if isinstance(mode, (int, float)) and math.isfinite(mode):
        return f"Mode {int(mode)}"
    return "Unspecified task"


class Dashboard:
    """One chart surface. Tabs are a strip of labels with the active one
    underlined; a tab shows up to six charts a page, paged with the arrows at
    the bottom; a chart earns its place when its data arrives and leaves the
    page while it is flat at zero. Hidden tabs retain stream data but own no
    artists."""

    PER_PAGE = 6
    STRIP_Y, STRIP_H = .866, .036      # tab strip (figure fraction)
    TILE_Y, TILE_H = .782, .044        # stat tiles under the status line
    CHART_TOP, CHART_BOTTOM = .748, .085

    def __init__(self, plt, title, temperature_caution=85.0, temperature_high=90.0, *, recorded=False, settings=None):

        if not (math.isfinite(temperature_caution) and math.isfinite(temperature_high)
                and temperature_caution < temperature_high):
            raise ValueError("temperature thresholds must be finite and caution < high")
        self.plt, self.title, self.recorded = plt, title, recorded
        self.temperature_caution, self.temperature_high = temperature_caution, temperature_high
        self.fig = plt.figure(figsize=(16, 11), facecolor=BACKGROUND)
        from matplotlib.patches import Rectangle, FancyBboxPatch as _Box
        # Top bar: a band a shade above the page with a hairline of accent under
        # it, holding the mark, the policy, the link state and the wordmark.
        self.fig.patches.append(Rectangle((0, .915), 1, .085, transform=self.fig.transFigure, facecolor=HEADER, edgecolor="none", zorder=-2))
        self.fig.patches.append(Rectangle((0, .913), 1, .002, transform=self.fig.transFigure, facecolor=ACCENT, edgecolor="none", zorder=-2))
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title("ProtoMotions Deployment Framework")
        # NVIDIA mark top-left (the white-on-dark horizontal lockup, transparent
        # PNG rasterised from the brand SVG); the policy name sits to its right.
        logo = Path(__file__).with_name("assets") / "nvidia_logo.png"
        text_x = .045
        if logo.exists():
            image = self.plt.imread(str(logo))
            logo_ax = self.fig.add_axes([.045, .936, .115, .045], anchor="W")
            logo_ax.imshow(image)
            logo_ax.axis("off")
            logo_ax.set_zorder(3)
            text_x = .172
        self.policy_text = self.fig.text(text_x, .962, title, color=TEXT, fontsize=17, weight="bold")
        self.link_pill = _Box((.455, .951), .09, .026, boxstyle="round,pad=0.002,rounding_size=0.012",
                              transform=self.fig.transFigure, facecolor=GRID, edgecolor="none", zorder=1)
        self.fig.patches.append(self.link_pill)
        self.link_text = self.fig.text(.5, .964, "WAITING", color=TEXT, ha="center", va="center", fontsize=9.5, weight="bold", zorder=2)
        self.cycle_text = self.fig.text(.556, .964, "", color=MUTED, ha="left", va="center", fontsize=8.5)
        # Two-tone wordmark: only the product name carries the brand green.
        from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
        self.powered_by_texts = [TextArea("Powered by ", textprops=dict(color=TEXT, fontsize=13, weight="bold")),
                                 TextArea("ProtoMotions", textprops=dict(color=ACCENT, fontsize=13, weight="bold"))]
        header_ax = self.fig.add_axes([0, .915, 1, .085], zorder=0)
        header_ax.axis("off")
        header_ax.add_artist(AnchoredOffsetbox(loc="center right", frameon=False, pad=0, borderpad=0,
                                               bbox_to_anchor=(.965, .962), bbox_transform=self.fig.transFigure,
                                               child=HPacker(children=self.powered_by_texts, pad=0, sep=0, align="baseline")))
        self.status_text = self.fig.text(text_x, .934, "", color=MUTED, fontsize=10.5)
        self.fig.text(.965, .934, "READ-ONLY TELEMETRY", color=MUTED, ha="right", fontsize=8.5)
        self.alert_text = self.fig.text(.045, .843, "", color=MUTED, fontsize=10)
        self.joint_text = self.fig.text(.965, .843, "", color=MUTED, fontsize=9, ha="right")
        # Stat tiles: the handful of numbers an operator keeps an eye on
        # whatever tab is up, large and tabular, amber or red when they matter.
        self.tiles = {}
        tile_specs = [("mode", "MODE"), ("compute", "COMPUTE"), ("hands", "HAND ERROR"), ("tracking", "TRACKING ERROR"),
                      ("temp", "MAX TEMPERATURE"), ("jerk", "JERK RMS")]
        width = (.92 - .01 * (len(tile_specs) - 1)) / len(tile_specs)
        for index, (key, label) in enumerate(tile_specs):
            x = .045 + index * (width + .01)
            self.fig.patches.append(_Box((x, self.TILE_Y), width, self.TILE_H, boxstyle="round,pad=0.001,rounding_size=0.006",
                                         transform=self.fig.transFigure, facecolor=TILE, edgecolor=GRID, lw=.8, zorder=-1))
            label_text = self.fig.text(x + .012, self.TILE_Y + self.TILE_H - .012, label, color=MUTED, fontsize=7.5, va="top")
            value_text = self.fig.text(x + .012, self.TILE_Y + .008, "—", color=TEXT, fontsize=13, weight="bold", va="bottom")
            self.tiles[key] = (label_text, value_text)
        self.empty_text = self.fig.text(.5, .45, "", color=MUTED, ha="center", fontsize=13)
        self.pager_text = self.fig.text(.5, .03, "", color=MUTED, ha="center", fontsize=9.5)
        # Tab strip: one axes across the width, labels drawn as text, a hairline
        # under all of them and a bar under the active one.
        self.strip = self.fig.add_axes([.045, self.STRIP_Y, .92, self.STRIP_H])
        self.strip.axis("off")
        self.strip.set_xlim(0, 1)
        self.strip.set_ylim(0, 1)
        self.strip.plot([0, 1], [.02, .02], color=GRID, lw=1)
        self.tab_labels = {}
        self.tab_ranges = {}
        self.tab_bar = None
        # Settings tab: sliders for the steering caps and the list of motions the
        # controller's button would play. Writes go through the settings file.
        self.settings = settings
        self.motion_names = []
        self.motion_items = []
        self.motion_list_ax = None
        self.settings_artists = []   # figure-level artists and axes of the settings tab
        self.slider_artists = []     # the schema-driven part, rebuilt when the launcher's schema changes
        self.schema, self.schema_text = [], None
        self.sliders, self.applied_rows = {}, {}
        self.applied_text = None
        self.visible_tabs = ["overview", "thermal", "timing"] + (["settings"] if settings is not None else [])
        self.task_name = "Task"
        self.seen = set()
        self.declared_motors = 0  # largest temperature_count seen; sticky across dropouts
        self.axes = []
        self.page = 0
        self._shown = []
        self._eligible = set()   # panels of the active tab with something to show
        self._ever_live = set()
        self._snapshot = None
        self._task_keys_laid_out = frozenset()
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._draw_tabs()
        self.set_tab("overview")

    # ----- chrome -------------------------------------------------------
    def _draw_tabs(self):
        for text in self.tab_labels.values():
            text.remove()
        if self.tab_bar is not None:
            self.tab_bar.remove()
        self.tab_labels, self.tab_ranges, self.tab_bar = {}, {}, None
        x = 0.0
        for name in self.visible_tabs:
            label = "Task" if name == "task" else next(title for tab, title, _, _ in CORE_TABS if tab == name)
            width = .022 + .0115 * len(label)
            self.tab_labels[name] = self.strip.text(x + width / 2, .52, label, color=TEXT, ha="center", va="center",
                                                    fontsize=11, weight="medium")
            self.tab_ranges[name] = (x, x + width)
            x += width + .012
        self._mark_active_tab()

    def _mark_active_tab(self):
        if self.tab_bar is not None:
            self.tab_bar.remove()
            self.tab_bar = None
        for name, text in self.tab_labels.items():
            active = name == self.active_tab if hasattr(self, "active_tab") else name == "overview"
            text.set_color(ACCENT if active else MUTED)
            text.set_weight("bold" if active else "medium")
            if active:
                a, b = self.tab_ranges[name]
                (self.tab_bar,) = self.strip.plot([a, b], [.04, .04], color=ACCENT, lw=3, solid_capstyle="butt")

    def _teardown_settings(self):
        for artist in self.slider_artists + self.settings_artists:
            artist.remove()
        self.settings_artists, self.slider_artists, self.sliders, self.applied_rows, self.motion_items = [], [], {}, {}, []
        self.motion_list_ax, self.applied_text = None, None

    def _build_settings_tab(self):
        writer = self.settings
        heading = dict(color=TEXT, fontsize=12, weight="bold", va="top")
        self.settings_artists.append(self.fig.text(.06, self.CHART_TOP, "Steering limits", **heading))
        self.settings_artists.append(self.fig.text(.06, self.CHART_TOP - .03,
            "Caps on what the sticks may command. The tick is the default; the red band is past what was qualified.\n"
            "The value the launcher actually runs with is read back from the stream beside each slider.",
            color=MUTED, fontsize=9, va="top"))
        self.applied_text = self.fig.text(.06, self.CHART_TOP - .30, "", color=TEXT, fontsize=10, va="top",
                                          family="DejaVu Sans Mono")
        self.settings_artists.append(self.applied_text)
        self.settings_artists.append(self.fig.text(.55, self.CHART_TOP, "Tracking motion", **heading))
        self.settings_artists.append(self.fig.text(.55, self.CHART_TOP - .03,
            "Selects what LB plays.\nSelecting never starts, stops or switches a motion.",
            color=MUTED, fontsize=9, va="top"))
        self.motion_list_ax = self.fig.add_axes([.55, .10, .40, .01], zorder=20)
        self.motion_list_ax.axis("off")
        self.settings_artists.append(self.motion_list_ax)
        self._build_motion_rows(self.motion_names)
        self._build_sliders()
        where = (f"Settings file  {writer.path}" if writer is not None
                 else "No settings file: this dashboard is read-only (the launcher passes one when it starts the dashboard)")
        self.settings_artists.append(self.fig.text(.06, self.CHART_BOTTOM, where, color=MUTED, fontsize=8.5, va="bottom"))

    def _build_sliders(self):
        """One slider per range the launcher's schema declares: the owner of a
        tunable says its default, where qualified ends and where the hard bound is."""
        from matplotlib.widgets import Slider

        for artist in self.slider_artists:
            artist.remove()
        self.slider_artists, self.sliders, self.applied_rows = [], {}, {}
        if not self.schema:
            self.slider_artists.append(self.fig.text(.06, self.CHART_TOP - .10,
                "Waiting for the launcher to declare its settings…", color=MUTED, fontsize=10, va="top"))
            return
        writer = self.settings
        for row, entry in enumerate(self.schema):
            key, hard, qualified, default = entry["key"], entry["hard_max"], entry["qualified_max"], entry["default"]
            y = self.CHART_TOP - .12 - .075 * row
            ax = self.fig.add_axes([.15, y, .22, .022], facecolor=PANEL)
            if hard > qualified:
                ax.axvspan(qualified, hard, color=RED, alpha=.28, lw=0)
            ax.axvline(default, color=TEXT, lw=1.2)
            start = writer.values.get(key, default) if writer is not None else default
            slider = Slider(ax, f"{entry['label']} ({entry['unit']})  ", 0.0, hard, valinit=min(hard, max(0.0, start)),
                            valfmt="%.2f", color=ACCENT, track_color=PANEL, initcolor="none")
            slider.label.set_color(TEXT)
            slider.label.set_fontsize(9.5)
            slider.valtext.set_fontsize(9.5)
            slider.valtext.set_color(RED if slider.val > qualified else TEXT)

            def changed(value, key=key, qualified=qualified, slider=slider):
                slider.valtext.set_color(RED if value > qualified else TEXT)
                if writer is not None:
                    writer.set(**{key: round(float(value), 3)})
            slider.on_changed(changed)
            if writer is None:
                slider.set_active(False)
            self.sliders[key] = slider
            self.slider_artists.append(ax)
            applied = self.fig.text(.425, y + .011, "", color=MUTED, fontsize=9.5, va="center", family="DejaVu Sans Mono")
            self.applied_rows[key] = applied
            self.slider_artists.append(applied)

    def _choose_motion(self, name):
        if self.settings is not None and name in self.motion_names:
            self.settings.set(selected_motion=name)

    def _update_settings(self, last):
        if self.active_tab != "settings":
            return
        schema_text = last.get("settings/schema")
        if isinstance(schema_text, str) and schema_text != self.schema_text:
            try:
                schema = json.loads(schema_text)
            except ValueError:
                schema = None
            if isinstance(schema, list) and all(isinstance(e, dict) and "key" in e for e in schema):
                self.schema, self.schema_text = schema, schema_text
                self._build_sliders()
        available = last.get("tracking/available")
        motion_names = available.split(",") if isinstance(available, str) and available else []
        if motion_names != self.motion_names:
            self.motion_names = motion_names
            self._build_motion_rows(motion_names)
        playing, selected = last.get("tracking/motion"), last.get("tracking/selected")
        for (highlight, text), name in zip(self.motion_items, self.motion_names):
            highlight.set_visible(name == selected)
            text.set_color(ACCENT if name == playing else TEXT)
            text.set_weight("bold" if name == playing else "normal")
        version = last.get("settings/version")
        for entry in self.schema:
            applied = last.get("settings/" + entry["key"])
            text = self.applied_rows.get(entry["key"])
            if text is None:
                continue
            if applied is None or not isinstance(applied, (int, float)):
                text.set_text("launcher: —")
                text.set_color(MUTED)
            else:
                text.set_text(f"launcher: {applied:.2f} {entry['unit']}")
                text.set_color(RED if applied > entry["qualified_max"] else TEXT)
        if self.applied_text is not None:
            if version is None:
                lines = ["Launcher has not reported its settings yet"]
            else:
                lines = [f"Launcher settings v{int(version)}",
                         f"  selected motion  {selected or 'none'}" + (f"   playing {playing}" if playing else "")]
                error = last.get("settings/error")
                if isinstance(error, str) and error:
                    lines.append(f"  rejected: {error}")
            if self.settings is not None and self.settings.last_error:
                lines.append(f"  write failed: {self.settings.last_error}")
            self.applied_text.set_text("\n".join(lines))

    def _build_motion_rows(self, names):
        if self.motion_list_ax is None:
            return
        ax = self.motion_list_ax
        ax.cla()
        ax.axis("off")
        self.motion_items = []
        count = max(1, len(names))
        height = min(.036 * count, self.CHART_TOP - .07 - .12)
        ax.set_position([.55, self.CHART_TOP - .07 - height, .40, height])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, count)
        if not names:
            ax.text(.02, .5, "No motions in this stream", color=MUTED, fontsize=9.5, va="center")
            return
        ax.add_patch(self.plt.Rectangle((0, 0), 1, count, facecolor=PANEL, edgecolor=GRID, lw=1))
        for index, name in enumerate(names):
            highlight = self.plt.Rectangle((0, count - index - 1), 1, 1, facecolor=GRID, visible=False)
            ax.add_patch(highlight)
            text = ax.text(.03, count - index - .5, name, color=TEXT, fontsize=10, va="center")
            self.motion_items.append((highlight, text))

    def _on_click(self, event):
        if event.inaxes is self.strip and event.xdata is not None:
            for name, (a, b) in self.tab_ranges.items():
                if a <= event.xdata <= b:
                    self.set_tab(name)
                    return
        if self.motion_list_ax is not None and event.inaxes is self.motion_list_ax and event.ydata is not None:
            index = len(self.motion_names) - 1 - int(event.ydata)
            if 0 <= index < len(self.motion_names):
                self._choose_motion(self.motion_names[index])
            return
        if event.inaxes is None and event.y is not None and event.y < self.fig.bbox.height * .06 and event.x is not None:
            # the pager at the bottom: left half previous page, right half next
            self.turn_page(1 if event.x > self.fig.bbox.width / 2 else -1)

    # ----- panels -------------------------------------------------------
    def _style_axis(self, ax, title, unit):
        ax.set_facecolor(PANEL)
        # Latest values sit in the bottom-right corner, away from the legend in the top-left.
        ax.readout = ax.text(.99, .05, "", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=TEXT,
                             family="DejaVu Sans Mono", bbox=dict(facecolor=PANEL, edgecolor="none", pad=1.5, alpha=.85), zorder=5)
        ax.set_prop_cycle(color=PALETTE)
        ax.set_title(title, loc="left", color=TEXT, fontsize=10.5, pad=8)
        ax.set_ylabel(unit, color=MUTED, fontsize=8.5)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
        ax.grid(True, color=GRID, linewidth=.6, alpha=.5)
        ax.margins(x=0)
        for side, spine in ax.spines.items():
            spine.set_color(GRID)
            spine.set_visible(side in ("left", "bottom"))

    def panels_for_tab(self, name):
        preset = TASK_PRESETS.get(self.task_name, {})
        if name == "overview":
            return preset.get("overview", []) + next(p for n, _, p, _ in CORE_TABS if n == name)
        if name != "task":
            return next(p for n, _, p, _ in CORE_TABS if n == name)
        self._task_keys_laid_out = frozenset(k for k in self.seen if is_task_key(k))
        curated = preset.get("task", [])
        consumed = [pat for _, pats, _ in curated + preset.get("overview", []) for pat in pats]
        leftover = [k for k in self._task_keys_laid_out if not matches(k, consumed)]
        return curated + generic_task_panels(leftover)

    def set_tab(self, name):
        if name not in TAB_ORDER:
            raise ValueError(f"Unknown tab: {name}")
        if name not in self.visible_tabs:
            self.visible_tabs = [t for t in TAB_ORDER if t == name or t in self.visible_tabs]
            self.active_tab = name
            self._draw_tabs()
        for ax in self.axes:
            ax.remove()
        self._teardown_settings()
        self.active_tab, self.panels = name, self.panels_for_tab(name)
        self.page = 0
        self._shown = []
        self._eligible = set()
        self.axes = [self.fig.add_subplot(1, 1, 1, label=f"{name}-{i}") for i in range(len(self.panels))]
        self.lines, self.counts = {}, {}
        self.temperature_panel = None
        self.temperature_bands = []
        self.motor_bars = []
        for index, ax in enumerate(self.axes):
            title, _, unit = self.panels[index]
            self._style_axis(ax, title, unit)
            ax.set_visible(False)  # a panel earns its place when its data arrives
        if THERMAL in self.panels:
            self.temperature_panel = self.panels.index(THERMAL)
            ax = self.axes[self.temperature_panel]
            ax.axhline(self.temperature_caution, color="#D18A00", ls="--", lw=1)
            ax.axhline(self.temperature_high, color="#B91C1C", ls="--", lw=1)
            self.temperature_text = ax.text(.02, .97, "", transform=ax.transAxes, va="top", fontsize=9.5, weight="bold")
            ax.text(.99, .97, "Display thresholds only · not a safety interlock", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.5, color=MUTED)
        if name == "thermal":
            ax = self.axes[1]
            ax.axhline(self.temperature_caution, color="#D18A00", ls="--")
            ax.axhline(self.temperature_high, color="#B91C1C", ls="--")
            self.motor_status = ax.text(.01, .97, "", transform=ax.transAxes, color=MUTED, va="top", fontsize=9)
            self._motor_count = -1
        if name == "settings":
            self._build_settings_tab()
        self._mark_active_tab()
        if self._snapshot is not None:
            self.update(*self._snapshot)
        else:
            self._layout()
        self.fig.canvas.draw_idle()

    def _layout(self):
        """Place the shown panels on the current page: thermal is two full-width
        rows, every other tab up to six charts in two columns."""
        shown = sorted(self._eligible)
        thermal = self.active_tab == "thermal"
        per_page = 2 if thermal else self.PER_PAGE
        pages = max(1, math.ceil(len(shown) / per_page))
        self.page = min(self.page, pages - 1)
        on_page = shown[self.page * per_page:(self.page + 1) * per_page]
        cols = 1 if thermal else (1 if len(on_page) == 1 else 2)
        rows = max(1, math.ceil(len(on_page) / cols))
        grid = self.fig.add_gridspec(rows, cols, left=.06, right=.965, bottom=.16 if thermal else self.CHART_BOTTOM,
                                    top=self.CHART_TOP, hspace=.42 if thermal else .5, wspace=.16)
        for index, ax in enumerate(self.axes):
            if index in on_page:
                slot = on_page.index(index)
                ax.set_subplotspec(grid[slot // cols, slot % cols])
                ax.set_in_layout(True)
                last_row = slot // cols == rows - 1
                ax.set_xlabel("Stream time (s)" if last_row and not (thermal and index == 1) else "", color=MUTED, fontsize=8.5)
                ax.set_visible(True)
            else:
                ax.set_visible(False)  # on another page or quiet; it keeps its lines
        self._shown = shown
        self.pager_text.set_text(f"‹  page {self.page + 1} of {pages}  ›" if pages > 1 else "")

    def turn_page(self, step):
        self.page = max(0, self.page + step)
        self._layout()
        self.fig.canvas.draw_idle()

    def _legend(self, ax):
        # Thermal panels carry a status line in the top-left corner; their legends step aside.
        thermal_panel = self.temperature_panel is not None and ax is self.axes[self.temperature_panel]
        bars_panel = self.active_tab == "thermal" and len(self.axes) > 1 and ax is self.axes[1]
        loc = "lower right" if thermal_panel else "upper right" if bars_panel else "upper left"
        return ax.legend(fontsize=8, loc=loc, ncol=3, frameon=False, labelcolor=TEXT,
                         handlelength=1.2, columnspacing=1.0, borderaxespad=.2)

    def panel_for(self, key):
        for index, (_, patterns, _) in enumerate(self.panels):
            if matches(key, patterns):
                return index
        return None

    # ----- thermal ------------------------------------------------------
    def _build_motor_bars(self, count):
        ax = self.axes[1]
        for bars in self.motor_bars:
            bars.remove()
        self.motor_bars = []
        self._motor_count = count
        if count == 0:
            ax.set_xticks([])
            ax.set_xlabel("No per-motor telemetry in this stream", color=MUTED)
            return
        positions = np.arange(count)
        self.motor_bars = [ax.bar(positions - .19, np.zeros(count), .38, color=PALETTE[0], label="Winding"),
                           ax.bar(positions + .19, np.zeros(count), .38, color=PALETTE[1], label="Driver")]
        names = joint_names(self._snapshot[1]) if self._snapshot else []
        labels = [pretty_joint(names[i]) if i < len(names) else str(i) for i in positions]
        ax.set_xticks(positions, labels, rotation=90 if names else 0, fontsize=7.5)
        ax.set_xlabel("Motor" if names else "Motor index (stream order)", color=MUTED)
        ax.set_xlim(-.75, count - .25)
        self._legend(ax)

    def _update_thermal(self, last, age):
        if self.temperature_panel is None:
            return
        ax = self.axes[self.temperature_panel]
        family_seen = any(matches(k, TEMPERATURE_FAMILY) for k in self.seen)
        for band in self.temperature_bands:
            band.remove()
        ax.relim()
        ax.set_autoscaley_on(True)
        ax.autoscale_view()
        bottom, top = ax.get_ylim()
        top = max(top, self.temperature_high + 15)
        ax.set_ylim(min(bottom, self.temperature_caution - 20), top)
        self.temperature_bands = [ax.axhspan(self.temperature_caution, self.temperature_high, color="#FBBF24", alpha=.12),
                                  ax.axhspan(self.temperature_high, top, color="#EF4444", alpha=.13)]
        status, color = temperature_status(last, age, self.temperature_caution, self.temperature_high)
        if not family_seen:
            status, color = "No temperature telemetry in this stream (a simulation has no sensor)", MUTED
        self.temperature_text.set_text(status)
        self.temperature_text.set_color("#FB7185" if color == "#B91C1C" else "#F6B65D" if "CAUTION" in status else MUTED)
        for (panel, key), line in self.lines.items():
            if panel != self.temperature_panel:
                continue
            value = last.get(key)
            current = temperature_age(last, age) < .5 and isinstance(value, (int, float)) and math.isfinite(value)
            hot, warm = current and value >= self.temperature_high, current and value >= self.temperature_caution
            line.set_color("#B91C1C" if hot else "#D18A00" if warm else PALETTE[1 if "driver" in key else 0])
            line.set_linewidth(2.5 if hot else 1.5)
            line.set_linestyle("--" if "driver" in key else "-")
        if ax.get_legend() is not None:
            self._legend(ax)
        if self.active_tab != "thermal":
            return
        count = motor_count(self.seen, last, self.declared_motors)
        self.declared_motors = max(self.declared_motors, count)
        if count != self._motor_count:
            self._build_motor_bars(count)
        if count == 0:
            self.motor_status.set_text("no per-motor telemetry" + (" · maxima only" if family_seen else ""))
            return
        counts, maximum = [], self.temperature_high + 15
        stale = temperature_age(last, age) >= .5
        for kind, bars, base in zip(["winding", "driver"], self.motor_bars, PALETTE):
            valid = 0
            for index, bar in enumerate(bars):
                value = last.get(f"motor/{kind}_c/{index}")
                available = not stale and isinstance(value, (int, float)) and math.isfinite(value)
                valid += int(available)
                # Missing sensors have a hatched placeholder, never a fabricated temperature.
                bar.set_height(value if available else 3)
                bar.set_hatch(None if available else "////")
                bar.set_facecolor("#B91C1C" if available and value >= self.temperature_high else
                                  "#D18A00" if available and value >= self.temperature_caution else base if available else GRID)
                if available:
                    maximum = max(maximum, value + 12)
            counts.append(f"{kind} {valid}/{count}")
        self.axes[1].set_ylim(0, maximum)
        self.motor_status.set_text(" · ".join(counts) + " current · hatched = unavailable / stale")

    # ----- stream -------------------------------------------------------
    def _refresh_tabs(self, last):
        """Tabs follow the stream: a new key family opens its tab, a task name
        may switch the task tab to a curated layout."""
        name = last.get("task/name")
        task_name = name if isinstance(name, str) and name else "Task"
        # An explicitly requested tab keeps its place while it is active, even
        # with no data yet: it shows "no data" rather than vanishing under the user.
        visible = [t for t in TAB_ORDER if tab_visible(t, self.seen) or t == self.active_tab
                   or (t == "settings" and self.settings is not None)]
        rename = task_name != self.task_name
        self.task_name = task_name
        if visible != self.visible_tabs:
            self.visible_tabs = visible
            self._draw_tabs()
        if rename and self.active_tab == "overview":
            self.set_tab("overview")  # the preset's summary panels join the overview
            return True
        if self.active_tab == "task":
            task_keys = frozenset(k for k in self.seen if is_task_key(k))
            if rename or task_keys != self._task_keys_laid_out:
                self.set_tab("task")
                return True
        return False

    def update(self, data, last, last_received):
        self._snapshot = (data, last, last_received)
        self.seen.update(data)
        if self._refresh_tabs(last):
            return  # set_tab re-entered update with the new layout
        names = joint_names(last)
        touched, live = set(), set()
        for key in sorted(data):
            panel = self.panel_for(key)
            if panel is None:
                continue
            ax, series = self.axes[panel], data[key]
            line = self.lines.get((panel, key))
            if line is None and not len(series):
                continue
            if line is None:
                (line,) = ax.plot([], [], lw=1.3, label=channel_label(key, names))
                self.lines[(panel, key)] = line
                count = sum(p == panel for p, _ in self.lines)
                if count <= 6:
                    self._legend(ax)
                else:
                    if ax.get_legend() is not None:
                        ax.get_legend().remove()
                    if panel not in self.counts:
                        self.counts[panel] = ax.text(.99, .97, "", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED)
                    per_joint = key.startswith(("joint/", "motor/"))
                    self.counts[panel].set_text(f"{count} joints, named from the robot model" if names and per_joint
                                                else f"{count} channels")
            line.set_data(series[:, 0], series[:, 1])
            touched.add(panel)
            channels = [k for (p, k) in self.lines if p == panel]
            if len(channels) <= 3 and panel != self.temperature_panel:  # the thermal panel has its status line
                unit = self.panels[panel][2]
                parts = []
                for k in channels:
                    ys = data.get(k)
                    if ys is not None and len(ys) and math.isfinite(ys[-1, 1]):
                        parts.append(f"{channel_label(k, names)} {ys[-1, 1]:.3g}{(' ' + unit) if unit and unit not in ('flag', 'count', 'fraction') else ''}")
                ax.readout.set_text("  ".join(parts))
            values = series[:, 1]
            # A channel that has only ever read zero is quiet; a chart of quiet
            # channels waits off the page until one of them moves.
            if len(values) and np.any(np.isfinite(values) & (values != 0)):
                live.add(panel)
        self._ever_live.update((self.active_tab, p) for p in live)
        for panel in touched:
            show = (self.active_tab, panel) in self._ever_live or self.active_tab == "thermal"
            if show:
                self._eligible.add(panel)
                self.axes[panel].relim()
                self.axes[panel].autoscale_view()
        if self.active_tab == "thermal":
            self._eligible.update(range(len(self.axes)))
        age = (0.0 if last else math.inf) if self.recorded else (time.monotonic() - last_received if last_received else math.inf)
        self._update_thermal(last, age)
        shown = sorted(self._eligible)
        if shown != self._shown:
            self._layout()
        label = "Task" if self.active_tab == "task" else next(title for tab, title, _, _ in CORE_TABS if tab == self.active_tab)
        self.empty_text.set_text("" if shown or self.active_tab in ("thermal", "settings") else f"No {label} data in this stream yet")
        alert, color = temperature_status(last, age, self.temperature_caution, self.temperature_high)
        if "UNAVAILABLE" in alert and not any(matches(k, TEMPERATURE_FAMILY) for k in self.seen):
            alert = "No temperature telemetry in this stream"
        self.alert_text.set_text(alert + ("  ·  recorded" if self.recorded else ""))
        self.alert_text.set_color("#FB7185" if color == "#B91C1C" else "#F6B65D" if "CAUTION" in alert else MUTED)
        self._update_settings(last)
        policy = str(last.get("policy/name") or "Policy not identified")
        self.policy_text.set_text(policy if len(policy) <= 32 else policy[:29] + "…")
        link = "RECORDED" if self.recorded else "LIVE" if age < .5 else "STALE" if last else "WAITING"
        self.link_text.set_text(link)
        self.cycle_text.set_text(f"cycle {last.get('cycle', 0)}" if last else "")
        self.link_pill.set_facecolor({"LIVE": "#2F5D1A", "RECORDED": "#2C3A4A", "STALE": "#6B4A12", "WAITING": GRID}[link])
        self.link_text.set_color({"LIVE": "#C7F08A", "RECORDED": TEXT, "STALE": "#FFD58A", "WAITING": MUTED}[link])
        # Stat tiles
        def tile(key, value, color=TEXT):
            self.tiles[key][1].set_text(value)
            self.tiles[key][1].set_color(color)
        tile("mode", mode_label(last) if last else "—", ACCENT if last.get("task/active") else TEXT)
        compute_now = last.get("cycle/compute_ms")
        policy_now = last.get("cycle/policy_ms")
        compute_label = "—"
        if isinstance(compute_now, (int, float)):
            compute_label = f"{compute_now:.1f} ms"
            if isinstance(policy_now, (int, float)):
                compute_label += f"  ·  policy {policy_now:.2f}"
        tile("compute", compute_label,
             "#FB7185" if isinstance(compute_now, (int, float)) and compute_now > 15 else TEXT)
        hands = [last.get(k) for k in ("task/hand_error_left_m", "task/hand_error_right_m")]
        hands = [h for h in hands if isinstance(h, (int, float)) and math.isfinite(h)]
        tile("hands", f"{100 * sum(hands) / len(hands):.1f} cm" if hands else "—")
        track_err = last.get("tracking/mean_body_error_m")
        tile("tracking", f"{100 * track_err:.1f} cm" if isinstance(track_err, (int, float)) and math.isfinite(track_err) else "—")
        temps = [float(v) for k, v in last.items() if matches(k, TEMPERATURE_PATTERNS) and isinstance(v, (int, float)) and math.isfinite(v)]
        if temps and temperature_age(last, age) < .5:
            hottest = max(temps)
            tile("temp", f"{hottest:.0f} °C", "#FB7185" if hottest >= self.temperature_high else "#F6B65D" if hottest >= self.temperature_caution else TEXT)
        else:
            tile("temp", "—")
        jerk = last.get("joint/jerk_rms")
        tile("jerk", f"{jerk:.0f} rad/s³" if isinstance(jerk, (int, float)) and math.isfinite(jerk) else "—")
        # Name the joints worth looking at: the one nearest its limit and the
        # ones whose velocity is noisiest over the last second.
        readout = []
        nearest = last.get("limits/nearest_joint")
        margin = last.get("limits/min_margin_rad")
        if isinstance(nearest, str) and nearest and isinstance(margin, (int, float)) and math.isfinite(margin):
            readout.append(f"nearest limit  {pretty_joint(nearest)} {math.degrees(margin):.0f}°")
        leaders = jitter_leaders(data, names, top=2)
        if leaders:
            readout.append("jitter  " + ", ".join(f"{name} {std:.2f} rad/s" for name, std in leaders))
        rough = jitter_leaders(data, names, top=2, family="joint/jerk/")
        if rough:
            readout.append("jerk  " + ", ".join(f"{name} {rms:.0f} rad/s³" for name, rms in rough))
        self.joint_text.set_text("  ·  ".join(readout))
        compute = last.get("cycle/compute_ms")
        compute_label = f"{compute:.2f} ms" if isinstance(compute, (int, float)) else "unavailable"
        mode = mode_label(last)
        playing = last.get("tracking/motion")
        if isinstance(playing, str) and playing:
            elapsed, duration = last.get("tracking/elapsed_s"), last.get("tracking/duration_s")
            if isinstance(elapsed, (int, float)) and isinstance(duration, (int, float)):
                mode += f" {playing} {elapsed:.1f}/{duration:.1f} s"
        self.status_text.set_text(f"{self.task_name} · {mode} · {'Active' if last.get('task/active') else 'Held'}"
                                  f"  |  Compute {compute_label}  |  {self.title}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--udp", default=None, help="host:port to listen on, e.g. 0.0.0.0:9870")
    parser.add_argument("--file", default=None, help="metrics JSONL written with --metrics-file")
    parser.add_argument("--realtime", action="store_true", help="replay a file at its recorded pace")
    parser.add_argument("--screenshot", default=None, help="render to this PNG and exit (headless, Agg backend)")
    parser.add_argument("--tab", choices=TAB_ORDER, help="initial tab; with --screenshot, save only this tab")
    parser.add_argument("--duration", type=float, default=None, help="with --udp and --screenshot: listen this long, then save")
    parser.add_argument("--seconds", type=float, default=HISTORY_SECONDS, help="history window")
    parser.add_argument("--settings-file", default=None,
                        help="operator settings JSON the launcher reads (the launcher passes its own); enables the Settings tab")
    parser.add_argument("--title", default="Native launcher metrics")
    parser.add_argument("--temperature-caution-c", type=float, default=85.0, help="amber display threshold, not a hardware safety limit")
    parser.add_argument("--temperature-high-c", type=float, default=90.0, help="red display threshold, not a hardware safety limit")
    args = parser.parse_args()
    if not args.udp and not args.file:
        parser.error("one of --udp or --file is required")
    stream = Stream(args.seconds)
    if args.screenshot:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.file:
        if args.screenshot:
            file_reader(stream, Path(args.file), realtime=False)
        else:
            threading.Thread(target=file_reader, args=(stream, Path(args.file), args.realtime), daemon=True).start()
    if args.udp:
        host, port = args.udp.rsplit(":", 1)
        threading.Thread(target=udp_reader, args=(stream, host, int(port)), daemon=True).start()

    settings = SettingsWriter(args.settings_file) if args.settings_file else None
    board = Dashboard(plt, args.title, args.temperature_caution_c, args.temperature_high_c,
                      recorded=bool(args.file), settings=settings)
    if args.screenshot:
        if args.udp:
            time.sleep(args.duration or 10.0)
        data, last, received = stream.snapshot()
        board.update(data, last, received)
        if args.tab:
            board.set_tab(args.tab)
            board.fig.savefig(args.screenshot, dpi=110)
        else:
            board.fig.savefig(args.screenshot, dpi=110)
            destination = Path(args.screenshot)
            for tab in list(board.visible_tabs):
                board.set_tab(tab)
                board.fig.savefig(destination.with_name(f"{destination.stem}_{tab}{destination.suffix}"), dpi=110)
        print(f"wrote {args.screenshot} ({stream.frames} frames, {len(data)} series, tabs: {', '.join(board.visible_tabs)})")
        return 0
    if args.tab:
        board.set_tab(args.tab)
    plt.ion()
    board.fig.show()
    period = 1.0 / REDRAW_HZ
    try:
        while plt.fignum_exists(board.fig.number):
            started = time.monotonic()
            data, last, received = stream.snapshot()
            board.update(data, last, received)
            board.fig.canvas.draw_idle()
            board.fig.canvas.flush_events()
            time.sleep(max(0.01, period - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
