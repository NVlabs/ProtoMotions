#!/usr/bin/env python3
# -----------------------------------------------------------
# test.py – run Masked‑Mimic with dummy obs, correct masks
# -----------------------------------------------------------

import os, time, math, builtins, torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from hydra.utils import instantiate

# ── 0 · register YAML resolvers ─────────────────────────────
OmegaConf.register_new_resolver(
    "eval", lambda e: eval(e, {"math": math, **vars(builtins)}, {}), replace=True
)
OmegaConf.register_new_resolver("len",  lambda x: len(x),              replace=True)

# ── 1 · paths ───────────────────────────────────────────────
ROOT  = "/home/rover2/OrcaRL/ProtoMotions"
CKPT  = f"{ROOT}/data/pretrained_models/masked_mimic/smpl/last.ckpt"
YAML  = f"{ROOT}/data/pretrained_models/masked_mimic/smpl/config.yaml"

# ── 2 · load + resolve cfg ─────────────────────────────────
cfg_dir, cfg_name = os.path.split(YAML)
cfg_name = os.path.splitext(cfg_name)[0]

os.chdir(ROOT)
with initialize_config_dir(config_dir=os.path.relpath(cfg_dir, ROOT), version_base="1.1"):
    cfg = compose(config_name=cfg_name)
OmegaConf.resolve(cfg)
print("✓  YAML loaded & resolved.")

# ── 3 · build model ────────────────────────────────────────
model_cfg = cfg.agent.config.model
model     = instantiate(model_cfg)
state     = torch.load(CKPT, map_location="cpu")["model"]
model.load_state_dict(state, strict=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval().to(device).half()
print("✓  Model ready on", device)

# ── 4 · dimensions ─────────────────────────────────────────
E, MM = cfg.env.config, cfg.env.config.masked_mimic
OBS_DIM   = E.humanoid_obs.obs_size                  # 358
LATENT    = cfg.agent.config.vae.latent_dim          # 64
FUT_N     = MM.masked_mimic_target_pose.num_future_steps + 1   # 11
SPARSE    = MM.masked_mimic_target_pose.num_obs_per_sparse_target_pose  # 184
HIST_STEPS= MM.historical_obs.num_historical_conditioned_steps           # 15
HIST_DIM  = (OBS_DIM + 1) * HIST_STEPS               # 5385
MIMIC_DIM = E.mimic_target_pose.num_future_steps * E.mimic_target_pose.num_obs_per_target_pose  # 6495
TERRAIN   = cfg.terrain.config.terrain_obs_num_samples           # 256
MOTION_DIM= MM.motion_text_embeddings.embedding_dim   # 512

print(f"DIMS  obs={OBS_DIM} latent={LATENT} mimic={MIMIC_DIM} "
      f"sparse={SPARSE}*{FUT_N} hist={HIST_DIM} terrain={TERRAIN} motion={MOTION_DIM}")

# ── 5 · dummy batch with **correct mask lengths** ───────────
z   = lambda n: torch.zeros(1, n, device=device).half()
dummy = {
    "self_obs":                       z(OBS_DIM),
    "vae_noise":                      z(LATENT),
    "mimic_target_poses":             z(MIMIC_DIM),
    "masked_mimic_target_poses":      z(SPARSE * FUT_N),
    # masks –‑ one bool per token
    "masked_mimic_target_poses_masks": z(FUT_N),     # 11
    "historical_pose_obs":            z(HIST_DIM),
    "motion_text_embeddings":         z(MOTION_DIM),
    "motion_text_embeddings_mask":    z(1),          # single token
    "terrain":                        z(TERRAIN),
}

# ── 6 · inference loop ─────────────────────────────────────
print("\nRunning 10 dummy steps …\n")
for i in range(10):
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=device=="cuda"):
        act = model.act(dummy, with_encoder=False)
    print(f"step {i:02d}  action[:6] = {act[0,:6].cpu().numpy().round(3)}")
    time.sleep(0.02)

print("\n✓  Success – network runs with correct dummy data.")
