# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the BaseEnv parity harness (``protomotions/tests/parity``).

The fast tests only build the 3-env fake simulator scenarios (``fake3``,
``fake3_mimic`` and ``fake3_target``). The MuJoCo scenarios are behind
``@pytest.mark.parity_slow`` and run only with ``PROTOMOTIONS_PARITY_SLOW=1``.
"""

import copy
import os
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")  # the fake simulator is MuJoCo-backed kinematics

import protomotions.envs.base_env.reset.perturbation as perturbation  # noqa: E402
import protomotions.envs.component_manager as component_manager  # noqa: E402
from protomotions.agents.evaluators.base_evaluator import BaseEvaluator  # noqa: E402
from protomotions.envs.base_env.env import BaseEnv  # noqa: E402
from protomotions.envs.base_env.publishers import StatePublisher  # noqa: E402
from protomotions.envs.base_env.owners import (  # noqa: E402
    ActionStage,
    ContactMemory,
    EpisodeBuffers,
    HistoryState,
    ObservationCache,
    SensorsState,
)
from protomotions.envs.base_env.reset import RecoveryBank, SpawnPlacer  # noqa: E402
from protomotions.envs.component_manager import ComponentManager  # noqa: E402
from protomotions.tests.parity import (  # noqa: E402
    MUTATIONS,
    SCENARIOS,
    build_env,
    compare,
    format_result,
    record,
    save_trace,
)
from protomotions.tests.parity.__main__ import main as cli_main  # noqa: E402
from protomotions.tests.parity.scenarios import TARGET_CHECKS  # noqa: E402

FAKE3 = SCENARIOS["fake3"]
FAKE3_MIMIC = SCENARIOS["fake3_mimic"]
FAKE3_TARGET = SCENARIOS["fake3_target"]
IDENTITY_KEYS = (
    "identity/rew_is_rew_buf",
    "identity/done_is_reset_buf",
    "identity/term_is_terminate_buf",
    "identity/extras_terminate_is_terminate_buf",
)
# Mutations whose effect reaches a recorded VALUE on the 3-env fake scenario.
FAKE3_VALUE_MUTATIONS = [
    "progress_reset_all_rows",
    "stale_context_after_reset",
    "restore_skips_obs_cache",
    "restore_skips_history",
    "restore_skips_sensor_state",
    "history_seeded_before_sim_write",
    "user_reset_ignored",
    "respawn_offset_not_applied",
    "raw_extras_dropped",
    "reset_noise_before_recovery",
    "progress_tick_after_checks",
    "contact_forces_not_cleared_on_reset",
    "evaluator_actions_zeroed",
]

slow = pytest.mark.parity_slow
skip_unless_slow = pytest.mark.skipif(
    os.environ.get("PROTOMOTIONS_PARITY_SLOW") != "1",
    reason="MuJoCo scenario builds; set PROTOMOTIONS_PARITY_SLOW=1",
)


@pytest.fixture(scope="module")
def fake3_trace():
    return record(FAKE3)


@pytest.fixture(scope="module")
def fake3_mimic_trace():
    return record(FAKE3_MIMIC)


@pytest.fixture(scope="module")
def fake3_target_trace():
    return record(FAKE3_TARGET)


def _records(trace, suffix):
    return [r for r in trace["records"] if r["tag"].endswith(suffix)]


def _index_of(trace, tag):
    return next(i for i, r in enumerate(trace["records"]) if r["tag"] == tag)


def _copy_trace(trace):
    return {"meta": trace["meta"], "records": list(trace["records"])}


def _assert_keyboard_reset_fired(trace):
    """The R key pressed inside the first train1 step asks every env to reset.

    The press record (the step it was delivered in) is an ordinary step: the
    keyboard phase sets the episode buffers' request mask and leaves the clock
    alone, so its ``progress_buf`` is the previous step's plus one and its
    dones are untouched. On the following step every env that was not reset
    in between is done. For the fake scenarios no env is reset in between, so
    that is every env.
    """
    records = trace["records"]
    press = _index_of(trace, "train1/step")
    # The clock is untouched: the rows the reset before the press left alone
    # are one step past the record before it (an eval/restored or a step).
    prior, press_reset = records[press - 2], records[press - 1]
    assert press_reset["tag"] == "train1/reset"
    kept = torch.ones_like(records[press]["done"])
    kept[press_reset["reset_ids"]] = False
    assert torch.equal(
        records[press]["progress_buf"][kept], prior["progress_buf"][kept] + 1
    ), (records[press]["progress_buf"], prior["progress_buf"])
    assert not bool(records[press]["done"].any()), records[press]["done"]
    following_reset, following_step = records[press + 1], records[press + 2]
    assert following_reset["tag"] == "train1/reset"
    assert following_step["tag"] == "train1/step"
    not_reset = torch.ones_like(following_step["done"])
    not_reset[following_reset["reset_ids"]] = False
    assert bool(following_step["done"][not_reset].all()), following_step["done"]
    return following_reset["reset_ids"].numel() == 0


def test_fake3_recorded_twice_is_identical(fake3_trace):
    result = compare(fake3_trace, record(FAKE3))
    assert result.identical, format_result(result)
    eval_steps = int(_records(fake3_trace, "eval/restored")[0]["eval_env_steps"])
    # init/reset + 2 per training step + one per evaluator step + eval/restored
    assert result.num_records == 2 * FAKE3.steps + 2 + eval_steps
    assert eval_steps == FAKE3.eval_max_steps == len(_records(fake3_trace, "eval/step"))


def test_fake3_mimic_recorded_twice_and_tracks_a_reference_motion(fake3_mimic_trace):
    result = compare(fake3_mimic_trace, record(FAKE3_MIMIC))
    assert result.identical, format_result(result)
    steps = _records(fake3_mimic_trace, "/step")
    for key in (
        "motion_ids",
        "motion_times",
        "obs/mimic_target_poses",
        "ctx/mimic.ref_state.rigid_body_pos",
        "ctx/mimic.ref_state.dof_pos",
        "ctx/mimic.future_pos",
        "extras/raw_r/gt_rew",
        "extras/termination/tracking_error",
    ):
        assert key in steps[0], key
    # Both clips get played, partial resets resample rows, clips end and the
    # tracking-error termination fires somewhere in the rollout.
    assert {int(m) for r in steps for m in r["motion_ids"]} == {0, 1}
    sampled = {c for r in fake3_mimic_trace["records"] for c in r["calls"]}
    assert {"motion.sample_motions[n=1]", "motion.sample_motions[n=3]"} <= sampled
    assert "motion.post_physics_step" in steps[0]["calls"]
    assert any(bool(r["term"].any()) for r in steps)
    # The real MimicEvaluator ran (two clips scored on three envs) and its
    # write-back updated the sampling weights, which the record holds AFTER
    # update_training_state.
    restored = _records(fake3_mimic_trace, "eval/restored")[0]
    assert int(restored["eval/num_items"]) == FAKE3_MIMIC.fake_motions
    assert "eval/logs/eval/success_rate" in restored
    assert restored["eval/failed"].shape == (FAKE3_MIMIC.fake_motions,)
    assert not torch.equal(restored["motion_weights"], torch.ones(2))
    # The excursion restored the clip assignment (motion ids and times)
    # training had before it.
    before_eval = fake3_mimic_trace["records"][
        _index_of(fake3_mimic_trace, "eval/step") - 1
    ]
    assert torch.equal(restored["motion_ids"], before_eval["motion_ids"])
    assert torch.equal(restored["motion_times"], before_eval["motion_times"])


def test_fake3_target_recorded_twice_and_runs_the_target_task(fake3_target_trace):
    """fake3_target reproduces itself and exercises the target control's three
    surfaces: its published goal (observed, rewarded and recorded as context),
    its in-episode resampling, and its own episode-ending checks."""
    result = compare(fake3_target_trace, record(FAKE3_TARGET))
    assert result.identical, format_result(result)
    steps = _records(fake3_target_trace, "/step")
    for key in ("obs/target", "ctx/target.tar_pos", "extras/raw_r/target"):
        assert key in steps[0], key
    assert "motion_ids" not in steps[0]  # no reference motion on this task
    # The command source resamples the goal on its own schedule: the target
    # of a running episode (progress advanced, no reset) changes between
    # consecutive steps somewhere in the rollout.
    resampled = 0
    for before, after in zip(steps, steps[1:]):
        running = (after["progress_buf"] > before["progress_buf"]) & (
            before["progress_buf"] > 0
        )
        moved = (before["ctx/target.tar_pos"] != after["ctx/target.tar_pos"]).any(-1)
        resampled += int((running & moved).sum())
    assert resampled > 0
    # The env binds no termination component, so every terminated flag comes
    # from the control's fall, gap and stuck checks, and at least once they
    # fire on a row subset (a check that terminates every env at once would
    # not show a per-row mistake).
    terminated = [r["term"] for r in steps if bool(r["term"].any())]
    assert terminated
    assert any(not bool(t.all()) for t in terminated)


@pytest.mark.parametrize("check", TARGET_CHECKS)
def test_fake3_target_each_check_fires_on_its_own(check):
    """Each target check, enabled alone, ends an episode somewhere in the
    fake3_target rollout, so the recording is sensitive to all three and not
    only to the fall check."""
    trace = record(replace(FAKE3_TARGET, fake_target_checks=(check,)))
    assert any(bool(r["term"].any()) for r in _records(trace, "/step")), check


def test_identity_booleans_hold(fake3_trace):
    steps = _records(fake3_trace, "/step")
    assert steps
    for rec in steps:
        for key in IDENTITY_KEYS:
            assert rec[key] is True, (rec["tag"], key)


def test_step_records_have_calls_and_the_real_evaluator_ran(fake3_trace):
    for rec in _records(fake3_trace, "/step"):
        assert rec["calls"], rec["tag"]
        assert "sim.step" in rec["calls"]
        assert "all" in rec["compile_path"], rec["tag"]
        if not rec["tag"].startswith("eval/"):
            assert rec["calls"][0] == "env.step"
    for rec in _records(fake3_trace, "/reset"):
        assert any(c.startswith("env.reset[n=") for c in rec["calls"]), rec["tag"]
    eval_steps = _records(fake3_trace, "eval/step")
    assert len(eval_steps) == FAKE3.eval_max_steps and not _records(
        fake3_trace, "eval/skipped"
    )
    # The first evaluator step is preceded by the snapshot and the eval reset.
    assert eval_steps[0]["calls"][0] == "env.save_state"
    assert "env.reset[n=3]" in eval_steps[0]["calls"]
    for rec in eval_steps:
        for key in (
            "obs/actor",
            "rew",
            "done",
            "term",
            "extras/terminate",
            "rng/torch",
        ):
            assert key in rec, key
    restored = _records(fake3_trace, "eval/restored")
    assert len(restored) == 1
    assert restored[0]["calls"][0] == "env.restore_state"
    # The evaluator restores the whole batch; partial resets use subset contexts.
    assert "sim.reset_envs[n=3]" in restored[0]["calls"]
    assert fake3_trace["records"][0]["compile_path"] == ["subset:3"]
    for key in ("eval/num_items", "eval/score", "eval/failed", "eval/evaluated"):
        assert key in restored[0], key
    assert any(key.startswith("eval/logs/") for key in restored[0])


def test_records_hold_the_state_the_restore_bugs_missed(fake3_trace):
    init, step = fake3_trace["records"][0], fake3_trace["records"][2]
    for key in (
        "sim/actions",
        "sim/ticks",
        "max_episode_length",
        "history/raw_actions",
        "history/processed_actions",
        "ctx/prev_contact_force_magnitudes",
        "ctx/contact_forces",
        "ctx/noisy.dof_pos",
    ):
        assert key in step, key
    assert {k for k in init if k.startswith("fall_cache/")} >= {
        "fall_cache/root_pos",
        "fall_cache/root_rot",
        "fall_cache/dof_pos",
    }
    assert not any(k.startswith("fall_cache/") for k in step)
    # The informative fake: distinct spawn offsets per row and real contact
    # memory, so respawn and contact-clearing mistakes have something to show.
    assert torch.equal(
        init["respawn_root_offset"][:, :2],
        torch.tensor([[0.0, 0.0], [1.0, -1.0], [2.0, -2.0]]),
    )
    magnitudes = torch.stack(
        [r["ctx/prev_contact_force_magnitudes"] for r in _records(fake3_trace, "/step")]
    )
    assert bool((magnitudes > 0).any())
    # The epoch boundary bumps the episode length AFTER the evaluator excursion.
    restored = _index_of(fake3_trace, "eval/restored")
    lengths = [int(r["max_episode_length"]) for r in fake3_trace["records"]]
    assert lengths[restored] == 17 and lengths[restored + 1] == 22


def test_partial_resets_are_scheduled_on_fake3(fake3_trace):
    # The initial env.reset() has no reset_ids; every training reset does.
    sizes = {
        int(rec["reset_ids"].numel())
        for rec in _records(fake3_trace, "/reset")
        if "reset_ids" in rec
    }
    assert {0, 1} <= sizes, sizes  # some steps reset nobody, some a single row


def test_keyboard_reset_fires_inside_the_step(
    fake3_trace, fake3_mimic_trace, fake3_target_trace
):
    assert _assert_keyboard_reset_fired(fake3_trace)  # nobody reset in between
    assert _assert_keyboard_reset_fired(fake3_mimic_trace)
    assert _assert_keyboard_reset_fired(fake3_target_trace)


@pytest.mark.parametrize("name", FAKE3_VALUE_MUTATIONS)
def test_fake3_catches_value_mutations(fake3_trace, name):
    result = compare(fake3_trace, record(FAKE3, mutation=name))
    assert not result.values_identical, name
    assert result.index is not None and result.key is not None
    assert result.key not in ("calls", "compile_path"), (name, result.first_divergence)


def test_evaluator_actions_zeroed_is_seen_only_through_eval_steps(fake3_trace):
    # The write-back restores everything training reads afterwards, so the
    # only records that can see what the evaluator did are its own steps.
    result = compare(fake3_trace, record(FAKE3, mutation="evaluator_actions_zeroed"))
    assert not result.values_identical
    assert result.tag == "eval/step", format_result(result)
    assert result.index > _index_of(fake3_trace, "eval/step") - 1
    assert "obs/actor" in result.keys and "sim_actions" in result.keys


def test_user_reset_ignored_is_seen_at_the_step_after_the_press(fake3_trace):
    # The request mask is taken at the NEXT step's check, so the press step's
    # record is unchanged and the ignored request shows in the following dones.
    result = compare(fake3_trace, record(FAKE3, mutation="user_reset_ignored"))
    assert (result.index, result.key) == (
        _index_of(fake3_trace, "train1/step") + 2,
        "done",
    )


def test_actions_not_zeroed_on_reset_is_invisible_on_todays_main(fake3_trace):
    # On today's main the two zeroing lines this mutation removes are dead
    # code: StateHistoryBuffer.reset zeroes its own action slots whatever the
    # sample carries, and env.step overwrites both action buffers before
    # anything reads them. No scenario can see the difference, so the harness
    # must report the mutated trace as identical. If this test starts failing,
    # something began reading the buffers between reset and step.
    result = compare(fake3_trace, record(FAKE3, mutation="actions_not_zeroed_on_reset"))
    assert result.identical, format_result(result)


def test_fake3_catches_full_reset_global_ctx_through_compile_path(fake3_trace):
    mutated = record(FAKE3, mutation="full_reset_uses_global_ctx")
    values_only = compare(fake3_trace, mutated, ignore_prefixes=("compile_path",))
    assert values_only.identical, format_result(values_only)
    result = compare(fake3_trace, mutated)
    assert not result.values_identical and result.calls_identical
    assert result.key == "compile_path" and result.keys == ("compile_path",)
    assert result.index == 0  # init/reset resets every env
    assert mutated["records"][0]["compile_path"] == ["all"]
    assert "compile_path" in format_result(result)


def test_fake3_catches_reset_recomputes_all_obs_through_compile_path(fake3_trace):
    # A reset that recomputes EVERY env's observations from the all-env
    # context gives the same values (the other rows' state has not changed
    # since the step computed them) but moves the post-reset observations onto
    # the compiled path; the first reset (record 0) already shows it.
    mutated = record(FAKE3, mutation="reset_recomputes_all_obs")
    values_only = compare(fake3_trace, mutated, ignore_prefixes=("compile_path",))
    assert values_only.identical, format_result(values_only)
    result = compare(fake3_trace, mutated)
    assert not result.values_identical and result.calls_identical
    assert result.key == "compile_path" and result.keys == ("compile_path",)
    assert result.index == 0
    assert mutated["records"][0]["compile_path"] == ["all"]


def test_fake3_mimic_sees_control_before_motion_clock_as_call_order_only(
    fake3_mimic_trace,
):
    # MimicControl.step() is a no-op on today's main, so swapping it with the
    # motion clock cannot change a value: only the order of the two calls.
    # This is the verdict the harness must give for such a refactor. The
    # mutation is a reordered copy of the declared advance list.
    from protomotions.tests.parity.mutations import applied

    with applied("control_before_motion_clock"):
        built = build_env(FAKE3_MIMIC)
        try:
            names = [owner.name for owner in built.env._advance_owners]
        finally:
            built.env.close()
    assert names == ["controls", "motion", "spawn"]
    mutated = record(FAKE3_MIMIC, mutation="control_before_motion_clock")
    result = compare(fake3_mimic_trace, mutated)
    assert result.values_identical and not result.calls_identical
    assert not result.identical
    assert result.call_index == 2 and result.call_tag == "train0/step"
    assert format_result(result).startswith(
        "VALUES IDENTICAL, CALL ORDER CHANGED record 2"
    )
    assert "motion.post_physics_step" in result.first_call_divergence
    # fake3 has no motion manager, so there the mutation changes nothing.


def test_fake3_cannot_catch_perturbation_order(fake3_trace):
    # The fake simulator has no pushes, and reset noise is only drawn inside
    # env.reset (the write-back calls simulator.reset_envs directly), so
    # re-enabling the perturbations before the write-back draws nothing.
    # g1_stress is the scenario that catches this mutation (slow test below).
    result = compare(
        fake3_trace, record(FAKE3, mutation="perturbations_reenabled_before_restore")
    )
    assert result.identical, format_result(result)


def test_fall_cache_leak_is_visible_on_the_fake_through_the_stage_generators(
    fake3_trace,
):
    # Skipping the restore after generating the fall cache leaves the fake in
    # the fallen poses, which the initial env.reset() rewrites, and the fake
    # has no solver state to carry over; but the restore also puts back the
    # per-stage generators the generation advanced (reset noise, recovery), so
    # skipping it shifts those streams and the first reset draws differently.
    # The leak that used to be invisible on the fake is caught at record 0.
    result = compare(
        fake3_trace, record(FAKE3, mutation="fall_cache_leaks_into_training")
    )
    assert not result.identical and result.index == 0, format_result(result)
    assert any(key.startswith("rng/randomness/") for key in result.keys), result.keys


@slow
def test_stage_generators_are_isolated_from_each_other(fake3_trace):
    """C6: each stage draws from its own generator. fake3 resets on timeouts
    only, so building it without the recovery bank keeps the sequence of rows
    that draw the same, and the control, reset-noise and motion-sampling
    streams are bit-identical on every record (the same draws in the same
    order); the recovery stream, which the bank alone draws from, parts from
    its seed only in the recording that has the bank."""
    without_bank = record(replace(FAKE3, recovery=False))
    assert len(fake3_trace["records"]) == len(without_bank["records"])
    recovery_differs = False
    for a, b in zip(fake3_trace["records"], without_bank["records"]):
        for stage in ("control", "reset_noise", "motion_sampling"):
            key = f"rng/randomness/{stage}"
            assert torch.equal(a[key], b[key]), (a["tag"], stage)
        recovery_differs |= not torch.equal(
            a["rng/randomness/recovery"], b["rng/randomness/recovery"]
        )
    assert recovery_differs


@slow
def test_control_stage_draws_only_where_a_control_samples(
    fake3_trace, fake3_target_trace
):
    """fake3 has no control component, so its control stream never leaves its
    seed; fake3_target's target control samples goals from it, so the stream
    moves. The reset-noise stream moves on both (every reset draws noise)."""
    first, last = fake3_trace["records"][0], fake3_trace["records"][-1]
    assert torch.equal(first["rng/randomness/control"], last["rng/randomness/control"])
    assert not torch.equal(
        first["rng/randomness/reset_noise"], last["rng/randomness/reset_noise"]
    )
    first, last = fake3_target_trace["records"][0], fake3_target_trace["records"][-1]
    assert not torch.equal(
        first["rng/randomness/control"], last["rng/randomness/control"]
    )


def test_compare_reports_values_first_then_call_order(fake3_trace):
    other = _copy_trace(fake3_trace)
    other["records"][5] = dict(other["records"][5])
    other["records"][5]["progress_buf"] = other["records"][5]["progress_buf"] + 1
    result = compare(fake3_trace, other)
    assert not result.identical and not result.values_identical
    assert (result.index, result.key) == (5, "progress_buf")
    assert result.keys == ("progress_buf",)
    assert result.tag == fake3_trace["records"][5]["tag"]
    assert "record 5" in result.first_divergence
    assert "progress_buf" in result.first_divergence
    assert result.tag in result.first_divergence
    assert format_result(result).startswith("FIRST DIVERGENCE record 5")

    # Every differing key of the first diverging record is listed; a key
    # missing on one side counts as differing; a later record's divergence
    # does not show up.
    other["records"][5]["respawn_root_offset"] = (
        other["records"][5]["respawn_root_offset"] + 1
    )
    del other["records"][5]["ctx/current.root_pos"]
    other["records"][7] = dict(other["records"][7])
    other["records"][7]["progress_buf"] = other["records"][7]["progress_buf"] + 1
    result = compare(fake3_trace, other)
    assert result.index == 5 and result.key == "progress_buf"
    assert set(result.keys) == {
        "progress_buf",
        "respawn_root_offset",
        "ctx/current.root_pos",
    }

    # A changed call order with identical values is reported as such, after
    # the (clean) value pass, and `--ignore calls` hides it.
    other = _copy_trace(fake3_trace)
    other["records"][5] = dict(fake3_trace["records"][5])
    other["records"][5]["calls"] = other["records"][5]["calls"][1:]
    result = compare(fake3_trace, other)
    assert result.values_identical and not result.calls_identical
    assert (result.call_index, result.call_tag) == (5, fake3_trace["records"][5]["tag"])
    assert result.index is None and result.key is None
    assert format_result(result).startswith(
        "VALUES IDENTICAL, CALL ORDER CHANGED record 5"
    )
    assert compare(fake3_trace, other, ignore_prefixes=("calls",)).identical

    # A value divergence takes precedence over a call-order change.
    other["records"][9] = dict(fake3_trace["records"][9])
    other["records"][9]["progress_buf"] = other["records"][9]["progress_buf"] + 1
    result = compare(fake3_trace, other)
    assert not result.values_identical and not result.calls_identical
    assert result.index == 9 and result.call_index == 5
    assert format_result(result).startswith("FIRST DIVERGENCE record 9")

    # compile_path is a value key.
    other = _copy_trace(fake3_trace)
    other["records"][5] = dict(fake3_trace["records"][5])
    other["records"][5]["compile_path"] = ["subset:1"] + other["records"][5][
        "compile_path"
    ]
    result = compare(fake3_trace, other)
    assert not result.values_identical and result.key == "compile_path"

    # Float tolerance, and a shorter trace.
    other = _copy_trace(fake3_trace)
    other["records"][5] = dict(fake3_trace["records"][5])
    other["records"][5]["ctx/current.root_pos"] = (
        other["records"][5]["ctx/current.root_pos"] + 1e-6
    )
    assert compare(fake3_trace, other).key == "ctx/current.root_pos"
    assert compare(fake3_trace, other, atol=1e-5).identical
    shorter = {"meta": fake3_trace["meta"], "records": fake3_trace["records"][:-1]}
    result = compare(fake3_trace, shorter)
    assert not result.values_identical and result.index == len(shorter["records"])
    assert "record count differs" in result.first_divergence


def test_cli_compare_exit_codes(fake3_trace, tmp_path, capsys):
    clean = tmp_path / "clean.pt"
    save_trace(fake3_trace, clean)

    value = _copy_trace(fake3_trace)
    value["records"][5] = dict(value["records"][5])
    value["records"][5]["progress_buf"] = value["records"][5]["progress_buf"] + 1
    value_path = tmp_path / "value.pt"
    save_trace(value, value_path)

    calls = _copy_trace(fake3_trace)
    calls["records"][5] = dict(calls["records"][5])
    calls["records"][5]["calls"] = calls["records"][5]["calls"][1:]
    calls_path = tmp_path / "calls.pt"
    save_trace(calls, calls_path)

    assert cli_main(["compare", str(clean), str(clean)]) == 0
    assert capsys.readouterr().out.startswith("IDENTICAL")
    assert cli_main(["compare", str(clean), str(value_path)]) == 1
    assert capsys.readouterr().out.startswith("FIRST DIVERGENCE record 5")
    assert cli_main(["compare", str(clean), str(calls_path)]) == 2
    assert capsys.readouterr().out.startswith("VALUES IDENTICAL, CALL ORDER CHANGED")
    assert cli_main(["compare", str(clean), str(calls_path), "--ignore", "calls"]) == 0
    capsys.readouterr()
    assert cli_main(["compare", str(clean), str(tmp_path / "missing.pt")]) == 3
    assert "COMPARE FAILED" in capsys.readouterr().out


def _patched_surface():
    """Everything the spies and the mutation catalogue may rebind."""
    return (
        ComponentManager.execute_all,
        BaseEnv.reset,
        BaseEnv.restore_state,
        BaseEnv.step,
        BaseEnv.observe,
        BaseEnv._raw_state_extras,
        BaseEnv.__init__,
        BaseEvaluator.evaluate,
        BaseEvaluator._policy_action,
        perturbation.apply_reset_noise,
        component_manager.TORCH_COMPILE_AVAILABLE,
        # The owner methods the mutations replace, rewrite or wrap.
        EpisodeBuffers.reset_rows,
        ContactMemory.reset_rows,
        ActionStage.reset_rows,
        ObservationCache.load_state,
        HistoryState.load_state,
        SensorsState.load_state,
        StatePublisher.publish,
        SpawnPlacer.move_to_spawn,
        RecoveryBank.generate,
    )


def test_spies_and_mutations_leave_the_process_clean():
    before = _patched_surface()
    torch.manual_seed(99)
    np.random.seed(99)
    random.seed(99)
    rng_before = (
        torch.get_rng_state().clone(),
        np.random.get_state(),
        random.getstate(),
    )
    for name in (
        "progress_reset_all_rows",
        "perturbations_reenabled_before_restore",
        "evaluator_actions_zeroed",
        "control_before_motion_clock",
        "progress_tick_after_checks",
    ):
        assert name in MUTATIONS
        record(FAKE3, mutation=name)
    after = _patched_surface()
    assert before == after
    # record() seeds the global generators for the scenario and puts the
    # caller's streams back afterwards.
    assert torch.equal(torch.get_rng_state(), rng_before[0])
    assert np.random.get_state()[2] == rng_before[1][2]
    assert np.array_equal(np.random.get_state()[1], rng_before[1][1])
    assert random.getstate() == rng_before[2]


def test_spies_are_pass_through(fake3_trace, monkeypatch):
    # Recording without any instrumentation must give the same values; only
    # the ``calls`` and ``compile_path`` lists (produced by the spies) differ.
    from protomotions.tests.parity import recorder as recorder_module
    from protomotions.tests.parity.spies import CompilePathSpy

    monkeypatch.setattr(recorder_module, "install_call_spies", lambda env, log: log)
    monkeypatch.setattr(CompilePathSpy, "install", lambda self: self)
    unspied = record(FAKE3)
    assert all(rec["calls"] == [] for rec in unspied["records"])
    assert all(rec["compile_path"] == [] for rec in unspied["records"])
    result = compare(fake3_trace, unspied, ignore_prefixes=("calls", "compile_path"))
    assert result.identical, format_result(result)


def test_every_mutation_pattern_matches_todays_source():
    from protomotions.tests.parity.mutations import applied

    for name in MUTATIONS:
        with applied(name):
            pass


def test_publish_drops_derived_names_is_caught_by_the_build_check():
    """A publisher that forgets a required name never reaches a rollout: the
    env's vocabulary check rejects it while the env is being built."""
    from protomotions.tests.parity.mutations import applied

    with applied("publish_drops_derived_names"):
        with pytest.raises(ValueError, match="current.root_local_ang_vel"):
            build_env(FAKE3)


def test_rng_and_tensors_are_owned_copies(fake3_trace):
    init, reset, step = fake3_trace["records"][:3]
    assert init["rng/torch"].dtype == torch.uint8
    # The step-0 reset of every env draws recovery-reset masks and reset noise
    # from their stage generators, not the global torch stream; the sensor
    # generator advances on every step.
    assert not torch.equal(
        init["rng/randomness/reset_noise"], reset["rng/randomness/reset_noise"]
    )
    assert not torch.equal(
        init["rng/randomness/recovery"], reset["rng/randomness/recovery"]
    )
    assert torch.equal(init["rng/torch"], reset["rng/torch"])
    assert not torch.equal(reset["rng/sensors"], step["rng/sensors"])
    assert reset["rng/python"].shape == step["rng/python"].shape
    # A record holds clones, not views of live buffers.
    assert reset["progress_buf"].data_ptr() != step["progress_buf"].data_ptr()
    assert copy.deepcopy(reset["calls"]) == reset["calls"]


def test_fake_env_defaults_are_unchanged():
    # The informative variants are opt-in; existing sensor tests keep the
    # impulse-free, origin-spawning double with no reset noise or motions.
    from protomotions.tests.support.sensor_simulator import (
        FlatSensorTestTerrain,
        SensorTestSimulator,
        make_sensor_test_env,
    )

    sim = SensorTestSimulator(3)
    assert not sim.contact_model and torch.equal(
        sim.get_contact_forces(), torch.zeros(3, 4, 5, 3)
    )
    assert torch.equal(
        FlatSensorTestTerrain().sample_valid_locations(3), torch.zeros(3, 2)
    )
    env = make_sensor_test_env()
    try:
        assert env.robot_config.reset_noise is None
        assert env.reference_motion.motion_manager is None
        assert env.config.reward_components in (None, {})
    finally:
        env.close()


MUJOCO_SCENARIOS = [name for name, s in SCENARIOS.items() if s.experiment is not None]


def test_build_refuses_a_recipe_that_sets_save_dir(monkeypatch):
    """The pin before ``build_env_from_configs``: a recipe whose ``env.save_dir``
    is not the empty default would make the build read ``env_<motion>.ckpt`` as
    scene weights, so the harness refuses it, naming the scenario and the value,
    before any simulator is built (the config build is sim-free, so this runs
    in every session)."""
    from protomotions.tests.parity import scenarios as scenarios_module
    from protomotions.utils import config_builder

    build_experiment = config_builder.build_experiment

    def recipe_that_sets_save_dir(argv, **kwargs):
        experiment = build_experiment(argv, **kwargs)
        experiment.env.save_dir = "results/pinned"
        return experiment

    monkeypatch.setattr(config_builder, "build_experiment", recipe_that_sets_save_dir)
    with pytest.raises(
        ValueError, match="g1_tracker: the recipe sets env.save_dir='results/pinned'"
    ):
        scenarios_module._build_experiment_env(SCENARIOS["g1_tracker"])


@slow
@skip_unless_slow
@pytest.mark.parametrize("name", MUJOCO_SCENARIOS)
def test_mujoco_scenario_is_deterministic(name, monkeypatch):
    scenario = SCENARIOS[name]
    clean = record(scenario)
    result = compare(clean, record(scenario))
    assert result.identical, format_result(result)
    _assert_keyboard_reset_fired(clean)
    # Every spy target is pass-through on the real env too: without any
    # instrumentation the values are the same.
    from protomotions.tests.parity import recorder as recorder_module
    from protomotions.tests.parity.spies import CompilePathSpy

    monkeypatch.setattr(recorder_module, "install_call_spies", lambda env, log: log)
    monkeypatch.setattr(CompilePathSpy, "install", lambda self: self)
    unspied = record(scenario)
    result = compare(clean, unspied, ignore_prefixes=("calls", "compile_path"))
    assert result.identical, format_result(result)


@slow
@skip_unless_slow
def test_g1_stress_catches_perturbations_reenabled_before_restore():
    scenario = SCENARIOS["g1_stress"]
    clean = record(scenario)
    mutated = record(scenario, mutation="perturbations_reenabled_before_restore")
    result = compare(clean, mutated)
    assert not result.values_identical
    assert result.index >= _index_of(clean, "eval/restored"), format_result(result)
    assert result.key == "rng/torch", format_result(result)


@slow
@skip_unless_slow
@pytest.mark.parametrize(
    "name", ["fall_cache_leaks_into_training", "contact_forces_not_cleared_on_reset"]
)
def test_g1_stress_catches_simulator_state_mutations(name):
    # The fall cache is generated by stepping the real simulator: skipping
    # the restore leaves MuJoCo's solver state and action buffers advanced,
    # which the fake cannot show (its initial reset rewrites everything).
    # Contact memory IS visible on fake3 too (see FAKE3_VALUE_MUTATIONS); on
    # G1 it is the feet standing on the ground.
    scenario = SCENARIOS["g1_stress"]
    clean = record(scenario)
    result = compare(clean, record(scenario, mutation=name))
    assert not result.values_identical, (name, format_result(result))


@slow
@skip_unless_slow
def test_g1_tracker_catches_full_reset_global_ctx_through_compile_path():
    scenario = SCENARIOS["g1_tracker"]
    clean = record(scenario)
    mutated = record(scenario, mutation="full_reset_uses_global_ctx")
    values_only = compare(clean, mutated, ignore_prefixes=("compile_path",))
    assert values_only.identical, format_result(values_only)
    result = compare(clean, mutated)
    assert not result.values_identical and result.calls_identical
    assert result.key == "compile_path", format_result(result)
