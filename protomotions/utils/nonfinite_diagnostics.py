"""Bounded local diagnostics for non-finite simulator states; no collectives."""

from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import time
import traceback

import torch

log = logging.getLogger(__name__)
STATE_FIELDS = (
    "rigid_body_pos", "rigid_body_rot", "rigid_body_vel", "rigid_body_ang_vel",
    "dof_pos", "dof_vel", "rigid_body_contact_forces",
)
MAX_SAVED_ENVS = 32


def capture_step_context(env):
    """Keep pre-action motion identity/time and episode progress on device."""
    env._failure_local_step = getattr(env, "_failure_local_step", 0) + 1
    env._failure_pre_state = {}
    env._failure_pre_motion = {}
    env._failure_capture_errors = {}
    for key, owner, name in (
        ("motion_ids", getattr(env, "motion_manager", None), "motion_ids"),
        ("motion_times", getattr(env, "motion_manager", None), "motion_times"),
        ("episode_progress", env, "progress_buf"),
    ):
        try:
            value = getattr(owner, name, None)
            if isinstance(value, torch.Tensor):
                env._failure_pre_motion[key] = value.detach().clone()
        except Exception as exc:
            env._failure_capture_errors[key] = f"{type(exc).__name__}: {exc}"


def capture_pre_physics_state(env, context):
    """Retain clean state before physics without CPU synchronization or disk I/O."""
    state = {}
    for name in STATE_FIELDS:
        try:
            value = getattr(context.current, name, None)
            if isinstance(value, torch.Tensor):
                state[name] = value.detach().clone()
        except Exception as exc:
            env._failure_capture_errors["pre_physics_state." + name] = f"{type(exc).__name__}: {exc}"
    env._failure_pre_state = state


def _tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    import warp as wp
    return wp.to_torch(value)


def _selected(value, ids):
    value = _tensor(value)
    index = torch.tensor(ids, device=value.device, dtype=torch.long)
    return value.index_select(0, index).detach().cpu().clone()


def _summary(value):
    finite = torch.isfinite(value)
    ok = value[finite].float()
    return {
        "shape": list(value.shape),
        "nan_count": int(torch.isnan(value).sum()) if value.is_floating_point() else 0,
        "inf_count": int(torch.isinf(value).sum()) if value.is_floating_point() else 0,
        "finite_max_abs": float(ok.abs().max()) if ok.numel() else None,
        "nonfinite_coordinates_first_20": (~finite).nonzero().tolist()[:20],
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_nonfinite_diagnostic(env, error, *, phase, queried_env_ids=None):
    """Save JSON and exact CPU tensors once, never replacing the original error."""
    if error.diagnostic_attempted:
        return None
    error.diagnostic_attempted = True
    try:
        return _save(env, error, phase, queried_env_ids)
    except Exception as logging_error:
        log.exception("Could not save non-finite-state diagnostic; original error preserved")
        error.add_note(f"Diagnostic save failed: {type(logging_error).__name__}: {logging_error}")
        return None


def _save(env, error, phase, queried_env_ids):
    capture_errors = dict(getattr(env, "_failure_capture_errors", {})) if getattr(env, "_failure_in_step", False) else {}

    def read(label, fn):
        try:
            return fn()
        except Exception as exc:
            capture_errors[label] = f"{type(exc).__name__}: {exc}"
            return None

    provider = getattr(env, "failure_diagnostics_context", None)
    context = read("training_context", provider) if provider else {}
    context = context or {}
    rank = int(context.get("rank", os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))))
    experiment = getattr(getattr(env.simulator, "config", None), "experiment_name", "unbound")
    run_dir = Path(context.get("run_dir", Path("results") / experiment))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    folder = run_dir / "failures" / f"nonfinite_{stamp}_rank{rank}_pid{os.getpid()}_{time.time_ns()}"
    folder.mkdir(parents=True, exist_ok=False)

    all_rows = error.bad_env_ids
    all_envs = all_rows if queried_env_ids is None else queried_env_ids.detach().cpu()[all_rows].tolist()
    rows, ids = all_rows[:MAX_SAVED_ENVS], all_envs[:MAX_SAVED_ENVS]
    payload = {"env_ids": ids, "state": {}, "pre_physics_state": {}, "raw_simulator": {}, "motion": {}, "actions": {}, "pd_parameters": {}}
    for name in STATE_FIELDS:
        value = getattr(error.state, name, None)
        if value is not None:
            result = read(f"state.{name}", lambda v=value: _selected(v, rows))
            if result is not None:
                payload["state"][name] = result

    if getattr(env, "_failure_in_step", False):
        for name, value in getattr(env, "_failure_pre_state", {}).items():
            payload["pre_physics_state"][name] = _selected(value, ids)
        for name, value in getattr(env, "_failure_pre_motion", {}).items():
            payload["motion"]["pre_step_" + name] = _selected(value, ids)

    for name, attr in (("raw_action", "_current_raw_action"), ("pd_target", "_current_processed_action"), ("episode_progress", "progress_buf"), ("reset_buf", "reset_buf"), ("terminate_buf", "terminate_buf")):
        value = getattr(env, attr, None)
        if isinstance(value, torch.Tensor):
            payload["actions"][name] = _selected(value, ids)
    action_config = getattr(env.config, "action_config", None) or {}
    for name in ("pd_action_offset", "pd_action_scale", "stiffness", "damping"):
        value = action_config.get(name)
        if isinstance(value, torch.Tensor):
            payload["pd_parameters"][name] = value.detach().cpu().clone()

    simulator = env.simulator
    robot = getattr(simulator, "_robot", None)
    data = getattr(robot, "data", None)
    if data is not None:
        for name in ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "joint_pos", "joint_vel", "applied_torque", "joint_pos_target", "joint_vel_target", "joint_effort_target"):
            value = read(f"raw_simulator.{name}", lambda n=name: _selected(getattr(data, n), ids))
            if value is not None:
                payload["raw_simulator"][name] = value
    contact_reader = getattr(simulator, "_get_simulator_bodies_contact_buf", None)
    if contact_reader is not None:
        value = read("contact_forces", lambda: contact_reader(torch.tensor(ids, device=env.device)).rigid_body_contact_forces.detach().cpu().clone())
        if value is not None:
            payload["raw_simulator"]["contact_forces"] = value

    manager = getattr(env, "motion_manager", None)
    library = getattr(env, "motion_lib", None)
    for name in ("motion_ids", "motion_times"):
        value = getattr(manager, name, None)
        if isinstance(value, torch.Tensor):
            payload["motion"][name] = _selected(value, ids)
    motion_metadata = {}
    for prefix in ("pre_step_", ""):
        mids = payload["motion"].get(prefix + "motion_ids")
        times = payload["motion"].get(prefix + "motion_times")
        if library is None or mids is None or times is None:
            continue
        mids, times = mids.to(env.device), times.to(env.device)
        frame_data = read(prefix + "frame_indices", lambda: library._calc_frame_blend_from_id_and_time(mids, times))
        if frame_data is not None:
            frame0, frame1, blend = [x.detach().cpu() for x in frame_data]
            motion_metadata[prefix + "frames"] = {"frame0": frame0.tolist(), "frame1": frame1.tolist(), "blend": blend.tolist()}
            starts = getattr(library, "length_starts", None)
            if starts is not None:
                motion_metadata[prefix + "frames"]["packed_frame0"] = (starts[mids].detach().cpu() + frame0).tolist()
        files = getattr(library, "motion_files", None)
        if files:
            source_files = read(prefix + "source_files", lambda: [str(files[i]) for i in mids.detach().cpu().tolist()])
            if source_files is not None:
                motion_metadata[prefix + "source_files"] = source_files
    mids = payload["motion"].get("motion_ids")
    times = payload["motion"].get("motion_times")
    if library is not None and mids is not None and times is not None:
        reference = read("unaligned_reference_state", lambda: library.get_motion_state(mids.to(env.device), times.to(env.device)))
        if reference is not None:
            payload["reference_unaligned"] = {name: getattr(reference, name).detach().cpu().clone() for name in STATE_FIELDS if isinstance(getattr(reference, name, None), torch.Tensor)}

    params = getattr(getattr(simulator, "config", None), "sim", None)
    physx = getattr(params, "physx", None)
    physx_config = {f.name: getattr(physx, f.name) for f in fields(physx)} if is_dataclass(physx) else str(physx)
    metadata = {
        "schema_version": 1, "timestamp_utc": stamp, "rank": rank, "pid": os.getpid(),
        "phase": phase, "message": str(error), "failed_field": error.field_name,
        "affected_env_ids": all_envs, "saved_env_ids": ids, "saved_env_limit": MAX_SAVED_ENVS,
        "epoch": context.get("epoch"), "step_count": context.get("step_count"),
        "rollout_step": getattr(env, "_failure_rollout_step", None),
        "local_env_step": getattr(env, "_failure_local_step", None),
        "motion_file": getattr(library, "motion_file", None), "motion_id_scope": "zero-based within motion_file",
        "motion": {name: value.tolist() for name, value in payload["motion"].items()},
        "motion_frames": motion_metadata,
        "sim_body_names": getattr(robot, "body_names", None),
        "sim_joint_names": getattr(robot, "joint_names", None),
        "state_conversion": str(getattr(error.state, "state_conversion", None)),
        "dt": getattr(env, "dt", None), "physx": physx_config,
        "action_transform": action_config.get("action_transform"),
        "tensor_summaries": {group: {name: _summary(v) for name, v in values.items()} for group, values in payload.items() if isinstance(values, dict)},
        "capture_errors": capture_errors,
        "note": "Only saved_env_ids are included in tensors.pt; reference_unaligned is raw MotionLib state before environment alignment.",
    }
    temporary = folder / "tensors.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(folder / "tensors.pt")
    (folder / "metadata.json").write_text(json.dumps(_json_safe(metadata), indent=2, allow_nan=False))
    error.add_note(f"Non-finite diagnostic saved: {folder.resolve()}")
    (folder / "traceback.txt").write_text("".join(traceback.format_exception(error)))
    log.error("Non-finite state diagnostic saved: %s", folder.resolve())
    return folder
