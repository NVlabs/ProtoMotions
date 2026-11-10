# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Temperature display is telemetry only and must not imply missing data is safe."""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

_path = Path(__file__).resolve().parents[1] / "tools/live_metrics_viewer.py"
_spec = importlib.util.spec_from_file_location("live_metrics_viewer", _path)
viewer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(viewer)


@pytest.mark.parametrize("reading,expected", [(84.9, "Max temperature"), (85, "CAUTION"), (90, "HIGH TEMPERATURE"), (110, "HIGH TEMPERATURE")])
def test_temperature_boundaries(reading, expected):
    text, _ = viewer.temperature_status({"motor/max_winding_c": reading}, 0, 85, 90)
    assert expected in text


def test_temperature_unavailable_stale_and_partial():
    assert "UNAVAILABLE" in viewer.temperature_status({}, 0, 85, 90)[0]
    assert "UNAVAILABLE" in viewer.temperature_status({"motor/max_winding_c": None}, 0, 85, 90)[0]
    assert "UNAVAILABLE" in viewer.temperature_status({"motor/max_winding_c": float("nan")}, 0, 85, 90)[0]
    assert "STALE" in viewer.temperature_status({"motor/max_winding_c": 95}, 1, 85, 90)[0]
    text, _ = viewer.temperature_status({"motor/winding_c/1": 94, "motor/driver_c/1": None,
                                        "motor/temperature_valid": 0}, 0, 85, 90)
    assert "HIGH TEMPERATURE" in text and "PARTIAL" in text and "driver unavailable" in text


def test_driver_hot_when_winding_cool():
    text, color = viewer.temperature_status({"motor/max_winding_c": 50, "motor/max_driver_c": 95}, 0, 85, 90)
    assert "HIGH TEMPERATURE" in text and "driver" in text and color == "#B91C1C"


@pytest.mark.parametrize("key", ["cycle/state_age_ms", "comm/lowstate_age_ms"])
@pytest.mark.parametrize("age_ms", [500, 5000, None, float("nan"), -1])
def test_fresh_packet_with_old_or_invalid_sensor_age_is_stale(key, age_ms):
    text, _ = viewer.temperature_status({"motor/max_winding_c": 45, key: age_ms}, 0, 85, 90)
    assert "STALE" in text


def test_sensor_age_keeps_advancing_after_packet_received():
    text, _ = viewer.temperature_status({"motor/max_winding_c": 45, "cycle/state_age_ms": 450}, 0.1, 85, 90)
    assert "STALE" in text


def test_dashboard_keeps_other_tasks_and_dynamic_temperature_range(tmp_path):
    stream = viewer.Stream(20)
    board = viewer.Dashboard(plt, "Synthetic test telemetry")
    try:
        stream.push({"t_ns": 0, "cycle": 0, "motor/max_winding_c": 84, "task/new_metric": 1})
        board.update(*stream.snapshot())
        stream.push({"t_ns": 1_000_000_000, "cycle": 1, "motor/max_winding_c": 120,
                     "motor/max_driver_c": 95, "task/new_metric": 2})
        board.update(*stream.snapshot())
        assert board.axes[board.temperature_panel].get_ylim()[1] > 120
        assert board.visible_tabs == ["overview", "task", "thermal", "timing"]
        board.set_tab("task")
        assert board.axes[board.panel_for("task/new_metric")].get_visible()
        board.set_tab("overview")
        assert "HIGH TEMPERATURE" in board.temperature_text.get_text()
        assert board.lines[(board.temperature_panel, "motor/max_winding_c")].get_color() == "#B91C1C"
        legend = board.axes[board.temperature_panel].get_legend()
        assert all(handle.get_color() == "#B91C1C" for handle in legend.legend_handles)
        stream.push({"t_ns": 2_000_000_000, "cycle": 2})
        board.update(*stream.snapshot())
        assert "UNAVAILABLE" in board.temperature_text.get_text()
        board.fig.savefig(tmp_path / "missing_temperature.png")
    finally:
        plt.close(board.fig)


def test_temperature_dropout_gap_and_expiry():
    stream = viewer.Stream(20)
    stream.push({"t_ns": 0, "motor/max_winding_c": 95})
    stream.push({"t_ns": 1_000_000_000, "task/active": 1})
    stream.push({"t_ns": 2_000_000_000, "motor/max_winding_c": 50})
    series = stream.snapshot()[0]["motor/max_winding_c"]
    assert np.isnan(series[1, 1])
    stream.push({"t_ns": 23_000_000_000, "task/active": 1})
    # Dropout marker remains briefly, but no old sensor value survives.
    assert np.isnan(stream.snapshot()[0]["motor/max_winding_c"][:, 1]).all()
    stream.push({"t_ns": 44_000_000_000, "task/active": 1})
    assert stream.snapshot()[0]["motor/max_winding_c"].shape == (0, 2)


def test_history_prunes_before_snapshot_conversion():
    stream = viewer.Stream(20)
    for tick in range(1500):
        stream.push({"t_ns": tick * 20_000_000, "motor/max_winding_c": 60})
    data, _, _ = stream.snapshot()
    assert len(data["motor/max_winding_c"]) <= 1001
    assert len(stream.series["motor/max_winding_c"]) <= 1001


@pytest.mark.parametrize("caution,high", [(90, 85), (90, 90), (float("nan"), 90), (85, float("inf"))])
def test_bad_thresholds_rejected(caution, high):
    with pytest.raises(ValueError):
        viewer.Dashboard(plt, "test", caution, high)


@pytest.fixture
def tabbed_board():
    board = viewer.Dashboard(plt, "Synthetic tab regression")
    yield board
    plt.close(board.fig)


def complete_motor_frame(t_ns=0):
    frame = {"t_ns": t_ns, "cycle": 1, "motor/temperature_valid": 1,
             "policy/name": "Synthetic policy", "task/mode": 2, "task/mode_name": "Teleop", "task/active": 1,
             "motor/max_winding_c": 96, "motor/max_driver_c": 86}
    for index in range(29):
        frame[f"motor/winding_c/{index}"] = 40 + index
        frame[f"motor/driver_c/{index}"] = 35 + index
    frame["motor/winding_c/28"] = 96
    frame["motor/driver_c/28"] = 86
    return frame


def test_string_metadata_is_preserved_without_numeric_history():
    stream = viewer.Stream(20)
    frame = {"t_ns": 0, "cycle": 1, "policy/name": "Synthetic named policy",
             "policy/version": "v1", "source": {"backend": "test"},
             "joint/names": ["synthetic_joint"], "task/speed_measured": 0.3}
    stream.push(frame)
    data, last, _ = stream.snapshot()
    assert last == frame
    assert set(data) == {"task/speed_measured"}
    assert data["task/speed_measured"][0, 1] == 0.3


def test_switch_tabs_keeps_history_and_destroys_hidden_artists(tabbed_board):
    stream = viewer.Stream(20)
    for tick in range(3):
        stream.push({"t_ns": tick * 20_000_000, "joint/q/28": tick * .1,
                     "task/speed_measured": tick * .2, "task/new_metric": tick})
    tabbed_board.update(*stream.snapshot())
    assert tabbed_board.active_tab == "overview"
    assert not any(key == "joint/q/28" for _, key in tabbed_board.lines)
    old_axes = tuple(tabbed_board.axes)
    tabbed_board.set_tab("joints")
    assert all(ax not in tabbed_board.fig.axes for ax in old_axes)
    panel = tabbed_board.panel_for("joint/q/28")
    np.testing.assert_allclose(tabbed_board.lines[(panel, "joint/q/28")].get_ydata(), [0, .1, .2])
    assert tabbed_board.panel_for("task/speed_measured") is None
    assert not any(key == "task/speed_measured" for _, key in tabbed_board.lines)
    tabbed_board.set_tab("task")
    assert tabbed_board.axes[tabbed_board.panel_for("task/new_metric")].get_visible()
    for tab in ("thermal", "timing", "task", "overview", "joints"):
        tabbed_board.set_tab(tab)
        assert len(tabbed_board.axes) == len(tabbed_board.panels)
    np.testing.assert_allclose(tabbed_board.lines[(panel, "joint/q/28")].get_ydata(), [0, .1, .2])


def test_all_29_winding_and_driver_bars_and_last_motor_warning(tabbed_board):
    stream = viewer.Stream(20)
    stream.push(complete_motor_frame())
    tabbed_board.update(*stream.snapshot())
    tabbed_board.set_tab("thermal")
    assert [len(bars) for bars in tabbed_board.motor_bars] == [29, 29]
    assert "winding 29/29" in tabbed_board.motor_status.get_text()
    assert "driver 29/29" in tabbed_board.motor_status.get_text()
    assert all(bar.get_hatch() is None for bars in tabbed_board.motor_bars for bar in bars)
    assert tabbed_board.motor_bars[0][28].get_height() == 96
    assert tabbed_board.motor_bars[1][28].get_height() == 86
    assert tabbed_board.motor_bars[0][28].get_facecolor() == matplotlib.colors.to_rgba("#B91C1C")
    assert tabbed_board.motor_bars[1][28].get_facecolor() == matplotlib.colors.to_rgba("#D18A00")
    assert tabbed_board.axes[1].get_ylim()[1] > 96
    assert not any(key.startswith(("motor/winding_c/", "motor/driver_c/"))
                   for _, key in tabbed_board.lines)


def test_missing_current_motor_values_do_not_reuse_previous_bars(tabbed_board):
    stream = viewer.Stream(20)
    stream.push(complete_motor_frame())
    tabbed_board.update(*stream.snapshot())
    tabbed_board.set_tab("thermal")
    frame = complete_motor_frame(20_000_000)
    frame.pop("motor/winding_c/28")
    frame["motor/driver_c/28"] = None
    frame["motor/temperature_valid"] = 0
    stream.push(frame)
    tabbed_board.update(*stream.snapshot())
    assert "winding 28/29" in tabbed_board.motor_status.get_text()
    assert "driver 28/29" in tabbed_board.motor_status.get_text()
    assert "PARTIAL" in tabbed_board.temperature_text.get_text()
    for bars in tabbed_board.motor_bars:
        assert bars[28].get_hatch()
        assert bars[28].get_height() not in (86, 96)


@pytest.mark.parametrize("reason", ["receive", "sensor"])
def test_stale_motor_bars_are_unavailable(tabbed_board, reason):
    stream = viewer.Stream(20)
    frame = complete_motor_frame()
    if reason == "sensor":
        frame["comm/lowstate_age_ms"] = 600
    stream.push(frame)
    data, last, received = stream.snapshot()
    if reason == "receive":
        received -= 2
    tabbed_board.update(data, last, received)
    tabbed_board.set_tab("thermal")
    assert "STALE" in tabbed_board.temperature_text.get_text()
    assert "winding 0/29" in tabbed_board.motor_status.get_text()
    assert "driver 0/29" in tabbed_board.motor_status.get_text()
    assert all(bar.get_hatch() for bars in tabbed_board.motor_bars for bar in bars)


def test_aggregate_only_telemetry_does_not_invent_per_motor_readings(tabbed_board):
    stream = viewer.Stream(20)
    stream.push({"t_ns": 0, "motor/max_winding_c": 91})
    tabbed_board.update(*stream.snapshot())
    tabbed_board.set_tab("thermal")
    assert "HIGH TEMPERATURE" in tabbed_board.temperature_text.get_text()
    assert "driver unavailable" in tabbed_board.temperature_text.get_text()
    assert tabbed_board.motor_bars == []  # no motor was named or counted, so no bar is drawn
    assert "no per-motor telemetry" in tabbed_board.motor_status.get_text()
    stream.push({"t_ns": 20_000_000, "motor/max_winding_c": 91, "motor/temperature_count": 12})
    tabbed_board.update(*stream.snapshot())
    assert [len(bars) for bars in tabbed_board.motor_bars] == [12, 12]
    assert "winding 0/12" in tabbed_board.motor_status.get_text()
    assert all(bar.get_hatch() for bars in tabbed_board.motor_bars for bar in bars)


@pytest.mark.parametrize("tab", ["joints", "timing", "task"])
def test_high_temperature_warning_remains_in_header_on_nonthermal_tabs(tabbed_board, tab):
    stream = viewer.Stream(20)
    stream.push(complete_motor_frame())
    tabbed_board.update(*stream.snapshot())
    tabbed_board.set_tab(tab)
    assert tabbed_board.temperature_panel is None
    assert "HIGH TEMPERATURE" in tabbed_board.alert_text.get_text()
    assert tabbed_board.alert_text in tabbed_board.fig.texts


def test_recorded_screenshots_do_not_age_during_rendering():
    stream = viewer.Stream(20)
    stream.push(complete_motor_frame())
    data, last, received = stream.snapshot()
    board = viewer.Dashboard(plt, "Synthetic record", recorded=True)
    try:
        board.update(data, last, received - 20)
        board.set_tab("thermal")
        assert "RECORDED" in board.link_text.get_text()
        assert "recorded" in board.alert_text.get_text()
        assert "STALE" not in board.temperature_text.get_text()
        assert "winding 29/29" in board.motor_status.get_text()
        stale_last = dict(last, **{"cycle/state_age_ms": 600})
        board.update(data, stale_last, received - 20)
        assert "STALE" in board.temperature_text.get_text()
    finally:
        plt.close(board.fig)


def test_global_header_clears_hot_sensor_claim_when_current_frame_has_no_temperature(tabbed_board):
    stream = viewer.Stream(20)
    stream.push(complete_motor_frame())
    tabbed_board.update(*stream.snapshot())
    tabbed_board.set_tab("joints")
    assert "HIGH TEMPERATURE" in tabbed_board.alert_text.get_text()
    stream.push({"t_ns": 20_000_000, "joint/q/28": .5})
    tabbed_board.update(*stream.snapshot())
    assert "UNAVAILABLE" in tabbed_board.alert_text.get_text()
    assert "HIGH TEMPERATURE" not in tabbed_board.alert_text.get_text()


CHART_TABS = tuple(t for t in viewer.TAB_ORDER if t != "settings")


@pytest.mark.parametrize("tab", CHART_TABS)
def test_every_tab_screenshot_has_policy_identity_and_active_tab(tabbed_board, tmp_path, tab):
    stream = viewer.Stream(20)
    stream.push(dict(complete_motor_frame(), **{"task/speed_measured": .1, "joint/q/0": .2, "cycle/compute_ms": 3}))
    tabbed_board.update(*stream.snapshot())
    assert tabbed_board.visible_tabs == list(CHART_TABS)  # no settings file: no settings tab
    tabbed_board.set_tab(tab)
    assert tabbed_board.policy_text.get_text() == "Synthetic policy"
    assert tabbed_board.active_tab == tab
    assert tabbed_board.tab_labels[tab].get_color() == viewer.ACCENT
    powered = [area._text for area in tabbed_board.powered_by_texts]
    assert [text.get_text() for text in powered] == ["Powered by ", "ProtoMotions"]
    assert powered[0].get_color() == viewer.TEXT and powered[1].get_color() == viewer.ACCENT
    output = tmp_path / f"{tab}.png"
    tabbed_board.fig.savefig(output, dpi=55)
    image = plt.imread(output)
    assert image.shape[0] > 100 and image.shape[1] > 100
    assert np.std(image[..., :3]) > .01


def test_tabs_follow_the_stream_and_thermal_is_absent_without_motor_keys(tabbed_board):
    stream = viewer.Stream(20)
    stream.push({"t_ns": 0, "cycle/compute_ms": 2.0, "pose/tilt_deg": 1.0})
    tabbed_board.update(*stream.snapshot())
    assert tabbed_board.visible_tabs == ["overview", "thermal", "timing"]  # thermal and timing always have a tab
    assert "No temperature telemetry" in tabbed_board.alert_text.get_text()
    assert "UNAVAILABLE" not in tabbed_board.alert_text.get_text()
    stream.push({"t_ns": 20_000_000, "joint/q/3": .1, "motor/winding_c/3": None, "motor/temperature_valid": 0})
    tabbed_board.update(*stream.snapshot())
    assert tabbed_board.visible_tabs == ["overview", "joints", "thermal", "timing"]
    assert "UNAVAILABLE" in tabbed_board.alert_text.get_text()  # a named sensor that reads nothing is a fault, not absence
    assert set(tabbed_board.tab_labels) == set(tabbed_board.visible_tabs)


def test_task_tab_is_titled_from_the_stream_and_uses_the_curated_layout(tabbed_board):
    stream = viewer.Stream(20)
    stream.push({"t_ns": 0, "task/name": "Metatron v1", "task/mode_name": "Teleop", "task/active": 1,
                 "task/hand_error_left_m": .05, "ik/offset_left_rad": .01, "task/replants": 0})
    tabbed_board.update(*stream.snapshot())
    assert tabbed_board.task_name == "Metatron v1"
    assert tabbed_board.tab_labels["task"].get_text() == "Task"  # the task name lives in the status line
    assert tabbed_board.status_text.get_text().startswith("Metatron v1 · Teleop · Active")
    tabbed_board.set_tab("task")
    assert tabbed_board.panels == viewer.TASK_PRESETS["Metatron v1"]["task"]  # every key consumed: no fallback panel
    assert tabbed_board.axes[tabbed_board.panel_for("ik/offset_left_rad")].get_visible()
    assert not tabbed_board.axes[tabbed_board.panel_for("task/replants")].get_visible()  # only ever zero: quiet, off the page


def test_unknown_task_gets_a_generic_layout_grouped_by_unit_and_vector():
    keys = ["task/reach_error_m", "task/gripper_force_n", "task/yaw_error_deg", "task/phase",
            "goal/object/0", "goal/object/1", "goal/object/2", "support/center/0", "support/center/1",
            "task/mode", "task/name", "task/active"]
    panels = viewer.generic_task_panels(keys)
    titles = [title for title, _, _ in panels]
    # Vectors share a panel; every scalar keeps its own: equal units do not mean equal meaning.
    assert titles == ["goal/object", "support/center", "task/gripper_force_n", "task/phase",
                      "task/reach_error_m", "task/yaw_error_deg"]
    by_title = {title: (patterns, unit) for title, patterns, unit in panels}
    assert by_title["task/reach_error_m"] == (["task/reach_error_m"], "m")
    assert by_title["task/gripper_force_n"] == (["task/gripper_force_n"], "")  # unknown unit stays unlabelled
    assert by_title["goal/object"] == (["goal/object/#"], "")
    assert not any("task/mode" in patterns or "task/name" in patterns for _, patterns, _ in panels)


def test_generic_task_tab_relays_out_when_new_keys_arrive(tabbed_board):
    stream = viewer.Stream(20)
    stream.push({"t_ns": 0, "task/name": "Box pickup", "task/reach_error_m": .3})
    tabbed_board.update(*stream.snapshot())
    tabbed_board.set_tab("task")
    assert tabbed_board.tab_labels["task"].get_text() == "Task"
    assert tabbed_board.status_text.get_text().startswith("Box pickup")
    assert [t for t, _, _ in tabbed_board.panels] == ["task/reach_error_m"]
    stream.push({"t_ns": 20_000_000, "task/reach_error_m": .2, "goal/object/0": 1.0, "goal/object/1": 0.0, "goal/object/2": .5})
    tabbed_board.update(*stream.snapshot())
    assert [t for t, _, _ in tabbed_board.panels] == ["goal/object", "task/reach_error_m"]
    panel = tabbed_board.panel_for("goal/object/2")
    assert tabbed_board.axes[panel].get_visible()
    np.testing.assert_allclose(tabbed_board.lines[(tabbed_board.panel_for("task/reach_error_m"), "task/reach_error_m")].get_ydata(), [.3, .2])


def test_legacy_record_without_task_identity_still_renders(tabbed_board):
    stream = viewer.Stream(20)
    stream.push({"t_ns": 0, "task/mode": 1, "task/active": 1, "task/speed_measured": .4})
    tabbed_board.update(*stream.snapshot())
    assert tabbed_board.task_name == "Task"
    assert tabbed_board.status_text.get_text().startswith("Task · Mode 1 · Active")
    tabbed_board.set_tab("task")
    assert tabbed_board.axes[tabbed_board.panel_for("task/speed_measured")].get_visible()


def test_vector_flags_are_not_plotted_as_metres(tabbed_board):
    stream = viewer.Stream(20)
    stream.push({"t_ns": 0, "task/name": "Metatron v1", "goal/left/0": 1.0, "goal/left/1": 0.0,
                 "goal/left/2": .9, "goal/left/enabled": 1, "goal/left/novel": 3})
    tabbed_board.update(*stream.snapshot())
    tabbed_board.set_tab("task")
    metres = tabbed_board.panel_for("goal/left/2")
    assert tabbed_board.panels[metres][0] == "Hand goals · world position"
    assert tabbed_board.panel_for("goal/left/enabled") != metres
    assert tabbed_board.panel_for("goal/left/enabled") is not None
    # a key the preset never heard of still gets a panel, on its own
    novel = tabbed_board.panel_for("goal/left/novel")
    assert novel is not None and novel not in (metres, tabbed_board.panel_for("goal/left/enabled"))
    assert tabbed_board.panels[novel] == ("goal/left/novel", ["goal/left/novel"], "")


def test_hardware_without_readings_keeps_thermal_visible_and_unavailable(tabbed_board):
    stream = viewer.Stream(20)
    stream.push({"t_ns": 0, "motor/temperature_count": 0, "motor/temperature_valid": 0, "cycle/compute_ms": 2})
    tabbed_board.update(*stream.snapshot())
    assert "thermal" in tabbed_board.visible_tabs
    assert "UNAVAILABLE" in tabbed_board.alert_text.get_text()
    # a dropout later never removes the tab; the window expiring does not either
    stream.push({"t_ns": 25_000_000_000, "cycle/compute_ms": 2})
    tabbed_board.update(*stream.snapshot())
    assert "thermal" in tabbed_board.visible_tabs
    assert "UNAVAILABLE" in tabbed_board.alert_text.get_text()


def test_null_metric_is_discovered_as_unavailable_not_unsupported(tabbed_board):
    stream = viewer.Stream(20)
    stream.push({"t_ns": 0, "task/grasp_force": None})
    tabbed_board.update(*stream.snapshot())
    assert "task" in tabbed_board.visible_tabs
    tabbed_board.set_tab("task")
    assert tabbed_board.panel_for("task/grasp_force") is not None


def test_overview_task_summaries_come_only_from_the_preset(tabbed_board):
    stream = viewer.Stream(20)
    stream.push({"t_ns": 0, "task/heading_error_deg": 3.0, "pose/tilt_deg": 1.0})
    tabbed_board.update(*stream.snapshot())
    assert tabbed_board.panel_for("task/heading_error_deg") is None  # unknown task: overview stays generic
    assert tabbed_board.axes[tabbed_board.panel_for("pose/tilt_deg")].get_visible()
    stream.push({"t_ns": 20_000_000, "task/name": "Metatron v1", "task/heading_error_deg": 2.0, "pose/tilt_deg": 1.0})
    tabbed_board.update(*stream.snapshot())
    assert tabbed_board.active_tab == "overview"
    assert tabbed_board.axes[tabbed_board.panel_for("task/heading_error_deg")].get_visible()


def test_requested_tab_without_data_is_labelled_not_borrowed(tabbed_board):
    stream = viewer.Stream(20)
    stream.push({"t_ns": 0, "cycle/compute_ms": 2})
    tabbed_board.update(*stream.snapshot())
    assert "joints" not in tabbed_board.visible_tabs
    tabbed_board.set_tab("joints")  # an explicit --tab request
    assert tabbed_board.active_tab == "joints"
    assert "No Joints data" in tabbed_board.empty_text.get_text()
    assert not any(ax.get_visible() for ax in tabbed_board.axes)
    tabbed_board.set_tab("timing")
    assert tabbed_board.empty_text.get_text() == ""


def test_declared_motor_count_outlives_a_partial_frame(tabbed_board):
    stream = viewer.Stream(20)
    frame = complete_motor_frame()
    frame["motor/temperature_count"] = 29
    stream.push(frame)
    tabbed_board.update(*stream.snapshot())
    tabbed_board.set_tab("thermal")
    assert [len(bars) for bars in tabbed_board.motor_bars] == [29, 29]
    # a frame naming one motor and declaring none still shows 29 bars, 28 of them hatched
    stream.push({"t_ns": 20_000_000, "motor/winding_c/0": 50, "motor/temperature_valid": 0})
    tabbed_board.update(*stream.snapshot())
    assert [len(bars) for bars in tabbed_board.motor_bars] == [29, 29]
    assert "winding 1/29" in tabbed_board.motor_status.get_text()
    assert viewer.motor_count({"motor/winding_c/0"}, {"motor/temperature_count": 29}) == 29


def test_requested_empty_tab_keeps_its_button_across_updates(tabbed_board):
    stream = viewer.Stream(20)
    stream.push({"t_ns": 0, "cycle/compute_ms": 2})
    tabbed_board.update(*stream.snapshot())
    tabbed_board.set_tab("joints")
    stream.push({"t_ns": 20_000_000, "cycle/compute_ms": 2})
    tabbed_board.update(*stream.snapshot())
    assert tabbed_board.active_tab == "joints"
    assert "joints" in tabbed_board.tab_labels
    assert tabbed_board.tab_labels["joints"].get_color() == viewer.ACCENT
    assert "No Joints data" in tabbed_board.empty_text.get_text()


def test_preset_keeps_support_yaw_and_validity_apart(tabbed_board):
    stream = viewer.Stream(20)
    stream.push({"t_ns": 0, "task/name": "Metatron v1", "support/yaw": .3, "support/valid": 1})
    tabbed_board.update(*stream.snapshot())
    tabbed_board.set_tab("task")
    assert tabbed_board.panel_for("support/yaw") != tabbed_board.panel_for("support/valid")
    assert tabbed_board.panels[tabbed_board.panel_for("support/yaw")][2] == "rad"


NAMES = ",".join(f"j{i}_joint" if i != 15 else "left_shoulder_pitch_joint" for i in range(29))


def test_joint_names_label_lines_bars_and_the_hot_motor(tabbed_board):
    stream = viewer.Stream(20)
    frame = complete_motor_frame()
    frame["joint/names"] = NAMES
    frame["motor/winding_c/15"] = 97
    frame["motor/max_winding_c"] = 97
    frame["joint/q/15"] = .2
    stream.push(frame)
    tabbed_board.update(*stream.snapshot())
    assert "left shoulder pitch winding" in tabbed_board.alert_text.get_text()
    tabbed_board.set_tab("joints")
    panel = tabbed_board.panel_for("joint/q/15")
    assert tabbed_board.lines[(panel, "joint/q/15")].get_label() == "left shoulder pitch"
    tabbed_board.set_tab("thermal")
    labels = [t.get_text() for t in tabbed_board.axes[1].get_xticklabels()]
    assert labels[15] == "left shoulder pitch" and len(labels) == 29
    assert tabbed_board.motor_bars[0][15].get_facecolor() == matplotlib.colors.to_rgba("#B91C1C")


def test_joint_readout_names_nearest_limit_and_jitter(tabbed_board):
    stream = viewer.Stream(20)
    rng = np.random.default_rng(0)
    for tick in range(100):
        frame = {"t_ns": tick * 20_000_000, "joint/names": NAMES, "limits/nearest_joint": "j3_joint",
                 "limits/min_margin_rad": .1, "joint/dq/0": .0, "joint/dq/15": float(rng.normal(0, .5)), "joint/dq/2": .3}
        stream.push(frame)
    tabbed_board.update(*stream.snapshot())
    text = tabbed_board.joint_text.get_text()
    assert "nearest limit  j3 6°" in text
    assert "jitter  left shoulder pitch" in text  # the noisy joint leads
    assert viewer.channel_label("joint/dq/15", NAMES.split(",")) == "left shoulder pitch"
    assert viewer.channel_label("task/hand_error_left_m", NAMES.split(",")) == "hand_error_left_m"


SCHEMA = [{"key": "max_speed_mps", "label": "Max speed", "unit": "m/s", "default": 1.0, "qualified_max": 1.0, "hard_max": 5.0},
          {"key": "max_yaw_rate_rps", "label": "Max yaw rate", "unit": "rad/s", "default": 1.57, "qualified_max": 1.57, "hard_max": 3.14}]


def test_settings_tab_writes_the_file_atomically_and_shows_what_the_launcher_applied(tmp_path):
    import json as _json
    path = tmp_path / "settings.json"
    board = viewer.Dashboard(plt, "Synthetic tracking", settings=viewer.SettingsWriter(path))
    try:
        assert "settings" in board.visible_tabs and not path.exists()  # nothing written until a change
        stream = viewer.Stream(20)
        stream.push({"t_ns": 0, "task/name": "Metatron v1", "tracking/available": "dancing,kick,spinning",
                     "tracking/selected": "kick", "tracking/motion": "", "tracking/progress": 0.0})
        board.update(*stream.snapshot())
        board.set_tab("settings")
        assert board.motion_names == ["dancing", "kick", "spinning"]
        assert [h.get_visible() for h, _ in board.motion_items] == [False, True, False]  # the stream's selection
        assert not board.sliders and "Waiting for the launcher" in board.slider_artists[0].get_text()
        # The launcher's schema builds the sliders: the owner of a tunable sets default, qualified and hard bounds.
        stream.push({"t_ns": 10_000_000, "tracking/available": "dancing,kick,spinning", "tracking/selected": "kick",
                     "settings/schema": _json.dumps(SCHEMA), "settings/version": 0,
                     "settings/max_speed_mps": 1.0, "settings/max_yaw_rate_rps": 1.57, "settings/error": ""})
        board.update(*stream.snapshot())
        assert list(board.sliders) == ["max_speed_mps", "max_yaw_rate_rps"]
        speed = board.sliders["max_speed_mps"]
        assert speed.valmax == 5.0 and speed.val == 1.0 and speed.valtext.get_color() == viewer.TEXT
        assert board.applied_rows["max_speed_mps"].get_text() == "launcher: 1.00 m/s"
        board._choose_motion("dancing")  # what a click on the row does
        saved = _json.loads(path.read_text())
        assert saved == {"version": 1, "selected_motion": "dancing"}
        assert not path.with_name("settings.json.tmp").exists()  # renamed into place
        speed.set_val(2.5)  # past the qualified 1 m/s: allowed, shown in red
        board.sliders["max_yaw_rate_rps"].set_val(1.2)
        saved = _json.loads(path.read_text())
        assert saved["version"] == 3 and saved["max_speed_mps"] == 2.5 and saved["max_yaw_rate_rps"] == 1.2
        assert speed.valtext.get_color() == viewer.RED
        assert board.sliders["max_yaw_rate_rps"].valtext.get_color() == viewer.TEXT
        board._choose_motion("dancing")  # mirroring an unchanged value writes nothing
        assert _json.loads(path.read_text())["version"] == 3
        # The tab shows the launcher's applied values from the stream, not the file, red past qualified.
        stream.push({"t_ns": 20_000_000, "task/name": "Metatron v1", "task/mode_name": "Tracking",
                     "tracking/available": "dancing,kick,spinning", "tracking/selected": "dancing",
                     "tracking/motion": "dancing", "tracking/elapsed_s": 1.5, "tracking/duration_s": 6.3,
                     "tracking/progress": .24, "settings/schema": _json.dumps(SCHEMA), "settings/version": 3,
                     "settings/max_speed_mps": 2.5, "settings/max_yaw_rate_rps": 1.2, "settings/error": ""})
        board.update(*stream.snapshot())
        assert board.applied_rows["max_speed_mps"].get_text() == "launcher: 2.50 m/s"
        assert board.applied_rows["max_speed_mps"].get_color() == viewer.RED
        assert board.applied_rows["max_yaw_rate_rps"].get_color() == viewer.TEXT
        text = board.applied_text.get_text()
        assert "v3" in text and "playing dancing" in text
        assert board.motion_items[0][1].get_color() == viewer.ACCENT  # playing row
        assert "Tracking dancing 1.5/6.3 s" in board.status_text.get_text()
        board.set_tab("overview")
        assert board.motion_list_ax is None and not board.sliders  # the tab's widgets leave with it
        output = tmp_path / "settings.png"
        board.set_tab("settings")
        assert list(board.sliders) == ["max_speed_mps", "max_yaw_rate_rps"]  # rebuilt from the kept schema
        assert board.sliders["max_speed_mps"].val == 2.5  # from the file, so a reopened tab shows what was set
        board.fig.savefig(output, dpi=55)
        assert plt.imread(output).shape[0] > 100
    finally:
        plt.close(board.fig)


def test_settings_writer_reads_back_the_saved_file_and_ignores_junk(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"version": 7, "max_speed_mps": 0.3, "max_yaw_rate_rps": "fast", "selected_motion": "kick", "flag": true}')
    writer = viewer.SettingsWriter(path)
    assert writer.version == 7 and writer.values == {"max_speed_mps": .3, "selected_motion": "kick"}
    assert writer.set(max_speed_mps=.4) and writer.version == 8
    path.write_text("not json")
    assert viewer.SettingsWriter(path).values == {"selected_motion": ""}


def test_without_a_settings_file_the_dashboard_stays_read_only(tabbed_board):
    stream = viewer.Stream(20)
    stream.push({"t_ns": 0, "tracking/available": "kick", "tracking/progress": 0.0})
    tabbed_board.update(*stream.snapshot())
    assert tabbed_board.settings is None and "settings" not in tabbed_board.visible_tabs


def test_charts_page_six_at_a_time_and_quiet_charts_wait(tabbed_board):
    stream = viewer.Stream(20)
    frame = {"t_ns": 0, "task/name": "Box pickup"}
    for i in range(9):
        frame[f"task/metric_{i}"] = float(i + 1)
    frame["task/quiet"] = 0.0
    stream.push(frame)
    tabbed_board.update(*stream.snapshot())
    tabbed_board.set_tab("task")
    assert len(tabbed_board.panels) == 10
    shown = [i for i, ax in enumerate(tabbed_board.axes) if ax.get_visible()]
    assert len(shown) == 6 and "page 1 of 2" in tabbed_board.pager_text.get_text()
    assert tabbed_board.panel_for("task/quiet") not in tabbed_board._shown
    tabbed_board.turn_page(1)
    shown = [i for i, ax in enumerate(tabbed_board.axes) if ax.get_visible()]
    assert len(shown) == 3 and "page 2 of 2" in tabbed_board.pager_text.get_text()
    stream.push({"t_ns": 20_000_000, "task/name": "Box pickup", "task/quiet": .5})
    tabbed_board.update(*stream.snapshot())
    assert tabbed_board.panel_for("task/quiet") in tabbed_board._shown  # it moved, so it joins
