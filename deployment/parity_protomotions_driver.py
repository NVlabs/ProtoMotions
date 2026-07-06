"""Ground-truth parity dump for ProtoMotions stack (mirrors inference_agent.main).

Runs H1_2 BM tracker on humaneva motion index 0 in MuJoCo, forces deterministic
motion (id 0, t=0), and captures per-step raw obs-function inputs + actor action
for the first N control steps.  Scratch/debug tool -- not part of the library.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/glvov-wbc/xdgruntime")
os.makedirs(os.environ["XDG_RUNTIME_DIR"], exist_ok=True, mode=0o700)

CKPT = "/oscar/data/stellex/glvov/imprint/third_party/ProtoMotions/results/h1_2_bm_dr_amass/epoch_3400.ckpt"
MOTION = "/users/glvov/data/glvov/wbc_data/retarget_test/humaneva_h1_2.pt"
OUT = "/tmp/glvov-wbc/parity_protomotions.npz"
NUM_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 55

# Simulator import must precede torch.
from protomotions.utils.simulator_imports import import_simulator_before_torch
import_simulator_before_torch("mujoco")

import logging
from pathlib import Path
import numpy as np
import torch
from protomotions.utils.hydra_replacement import get_class
from protomotions.utils.fabric_config import FabricConfig
from lightning.fabric import Fabric

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------- config load
checkpoint = Path(CKPT)
resolved = torch.load(checkpoint.parent / "resolved_configs_inference.pt",
                      map_location="cpu", weights_only=False)
robot_config = resolved["robot"]
simulator_config = resolved["simulator"]
terrain_config = resolved.get("terrain")
scene_lib_config = resolved["scene_lib"]
motion_lib_config = resolved["motion_lib"]
env_config = resolved["env"]
agent_config = resolved["agent"]

current_simulator = simulator_config._target_.split(".")[-3]
if current_simulator != "mujoco":
    from protomotions.simulator.factory import update_simulator_config_for_test
    simulator_config = update_simulator_config_for_test(
        current_simulator_config=simulator_config,
        new_simulator="mujoco",
        robot_config=robot_config,
    )

simulator_config.num_envs = 1
simulator_config.headless = True
motion_lib_config.motion_file = MOTION

# ---------------------------------------------------------------- monkeypatches
# (1) Force deterministic motion: id 0, start time 0.
from protomotions.envs.motion_manager.mimic_motion_manager import MimicMotionManager

def _forced_sample_motions(self, env_ids, new_motion_ids=None):
    self.motion_ids[env_ids] = 0
    self.motion_times[env_ids] = 0.0
MimicMotionManager.sample_motions = _forced_sample_motions

# (2) Capture raw obs-function inputs at the exact call site (MdpComponent.compute).
from protomotions.envs.mdp_component import MdpComponent
_orig_compute = MdpComponent.compute
CAP = {"reduced": [], "target": [], "hist_actions": []}
_TARGET_FUNCS = {
    "compute_humanoid_reduced_coords_observations": "reduced",
    "build_reduced_coords_target_poses": "target",
    "compute_historical_actions_from_state": "hist_actions",
}

def _capturing_compute(self, ctx):
    fname = getattr(self.compute_func, "__name__", "")
    if fname in _TARGET_FUNCS:
        resolved_vars, func_params = self.resolve_args(ctx)
        rec = {}
        for k, v in resolved_vars.items():
            rec[k] = v.detach().cpu().numpy().copy() if torch.is_tensor(v) else v
        rec["_static"] = {k: v for k, v in func_params.items()}
        CAP[_TARGET_FUNCS[fname]].append(rec)
    return _orig_compute(self, ctx)
MdpComponent.compute = _capturing_compute

# ---------------------------------------------------------------- build components
fabric = Fabric(**FabricConfig(accelerator="cpu", devices=1, num_nodes=1,
                               loggers=[], callbacks=[]).as_kwargs())
fabric.launch()

from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
terrain_config, simulator_config = convert_friction_for_simulator(terrain_config, simulator_config)

from protomotions.utils.component_builder import build_all_components
components = build_all_components(
    terrain_config=terrain_config, scene_lib_config=scene_lib_config,
    motion_lib_config=motion_lib_config, simulator_config=simulator_config,
    robot_config=robot_config, device=fabric.device, save_dir=None,
)

from protomotions.envs.base_env.env import BaseEnv
EnvClass = get_class(env_config._target_)
env = EnvClass(config=env_config, robot_config=robot_config, device=fabric.device,
               terrain=components["terrain"], scene_lib=components["scene_lib"],
               motion_lib=components["motion_lib"], simulator=components["simulator"])

from protomotions.agents.base_agent.agent import BaseAgent
AgentClass = get_class(agent_config._target_)
agent = AgentClass(config=agent_config, env=env, fabric=fabric, root_dir=checkpoint.parent)
agent.setup()
agent.load(CKPT, load_env=False, load_training_state=False)
agent.eval()

# ---------------------------------------------------------------- rollout loop
records = []
done_indices = None
step = 0
n_red_before = 0
n_tgt_before = 0

for i in range(NUM_STEPS):
    obs, _ = env.reset(done_indices)
    agent.pre_collect_step(step)
    obs = agent.add_agent_info_to_obs(obs)
    obs_td = agent.obs_dict_to_tensordict(obs)

    with torch.no_grad():
        model_outs = agent.model(obs_td)
    action = model_outs["mean_action"] if "mean_action" in model_outs else model_outs["action"]

    # Snapshot clean current state from context for this step.
    ctx = env.context
    cur = ctx.current
    rec = {
        "step": np.int64(i),
        "motion_time": float(env.motion_manager.motion_times[0].item()),
        "obs.noisy_reduced_coords_obs": obs_td["noisy_reduced_coords_obs"][0].cpu().numpy().copy(),
        "obs.noisy_mimic_reduced_coords_target_poses": obs_td["noisy_mimic_reduced_coords_target_poses"][0].cpu().numpy().copy(),
        "obs.historical_previous_processed_actions": obs_td["historical_previous_processed_actions"][0].cpu().numpy().copy(),
        "action.mean_action": action[0].cpu().numpy().copy(),
        "ctx.current.dof_pos": cur.dof_pos[0].cpu().numpy().copy(),
        "ctx.current.dof_vel": cur.dof_vel[0].cpu().numpy().copy(),
        "ctx.current.anchor_rot": cur.anchor_rot[0].cpu().numpy().copy(),
        "ctx.current.root_local_ang_vel": cur.root_local_ang_vel[0].cpu().numpy().copy(),
    }
    records.append(rec)

    _, _, dones, _, extras = env.step(action)
    done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
    step += 1

# ---------------------------------------------------------------- persist
def _stack_records(recs):
    keys = recs[0].keys()
    return {k: np.stack([r[k] for r in recs]) for k in keys}

save = _stack_records(records)

# The obs-function input captures: align to steps. There may be extra calls
# during reset/materialization; keep them all, tagged by capture order.
def _pack_captures(cap_list, prefix):
    out = {}
    if not cap_list:
        return out
    # Union of keys across records (skip _static, store separately).
    all_keys = set()
    for r in cap_list:
        all_keys.update(k for k in r.keys() if k != "_static")
    for k in all_keys:
        vals = []
        ok = True
        for r in cap_list:
            if k not in r or not isinstance(r[k], np.ndarray):
                ok = False
                break
            vals.append(r[k])
        if ok:
            try:
                out[f"{prefix}.{k}"] = np.stack(vals)
            except Exception:
                pass
    out[f"{prefix}.__ncalls__"] = np.int64(len(cap_list))
    return out

for pfx, key in (("cap_reduced", "reduced"), ("cap_target", "target"),
                 ("cap_histact", "hist_actions")):
    save.update(_pack_captures(CAP[key], pfx))

# Persist target static_params (future_steps) from the last target capture.
if CAP["target"]:
    fs = CAP["target"][-1]["_static"].get("future_steps")
    if fs is not None:
        save["cap_target.future_steps"] = np.asarray(fs)

np.savez(OUT, **save)
print("PROTOMOTIONS_DONE steps=%d reduced_calls=%d target_calls=%d -> %s" % (
    len(records), len(CAP["reduced"]), len(CAP["target"]), OUT))
print("step0 motion_time=%.5f dof_pos[:4]=%s" % (
    records[0]["motion_time"], records[0]["ctx.current.dof_pos"][:4]))

if hasattr(env.simulator, "shutdown"):
    env.simulator.shutdown()
