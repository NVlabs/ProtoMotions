#!/usr/bin/env python3
# -----------------------------------------------------------
# masked_mimic_node.py  – idle / sit / path / reach / steer
# -----------------------------------------------------------

import os, sys, time, math, builtins, random, collections, threading
import torch, numpy as np
from contextlib import contextmanager
from hydra import initialize, initialize_config_dir,  compose
from omegaconf import OmegaConf
from hydra.utils import instantiate


# ── version-agnostic autocast ───────────────────────────────
@contextmanager
def autocast_cuda(enabled: bool):
    if not enabled:
        yield
    elif torch.__version__.startswith("1."):
        with torch.cuda.amp.autocast(enabled=True):
            yield
    else:
        with torch.amp.autocast(device_type="cuda", enabled=True):
            yield
# ────────────────────────────────────────────────────────────


class MaskedMimicNode:
    MODES = ["idle", "sit_on_chair", "follow_path", "reach_with_hand", "steer"]

    def __init__(self,
                 root="/home/rover2/OrcaRL/ProtoMotions",
                 ckpt_rel="data/pretrained_models/masked_mimic/smpl/last.ckpt",
                 yaml_rel="data/pretrained_models/masked_mimic/smpl/config.yaml",
                 loop_hz=50.0):

        self.root   = root
        self.ckpt   = os.path.join(root, ckpt_rel)
        self.yaml   = os.path.join(root, yaml_rel)
        print(self.yaml)
        print(self.yaml)
        print(self.yaml)
        print(self.yaml)
        self.period = 1.0 / loop_hz

        self._register_resolvers()
        self.cfg = self._load_cfg()
        self._extract_dims()
        self._build_model()
        self._reset_buffers()

        # runtime state
        self.mode = "idle"
        self.steer_cmd = [0.0, 0.0]   # left / right wheel pwm in steer mode

        threading.Thread(target=self._stdin_listener, daemon=True).start()
        print("🌟  MaskedMimicNode ready – keys: [c] chair, [p] path, "
              "[h] hand, [t] steer, [i/j/k/l] drive, any other → idle.\n")

    # ---------- main loop ------------------------------------------
    def run(self):
        step = 0
        while True:
            tic = time.time()

            sensor   = self._read_sensors_random()
            self_obs = self._make_self_obs(sensor)
            hist     = self._update_history(self_obs)

            if self.mode == "idle":
                left, right = 0.0, 0.0

            elif self.mode == "steer":
                left, right = self.steer_cmd

            else:
                inp = self._build_input_dict(self_obs, hist)
                with torch.no_grad(), autocast_cuda(self.cuda):
                    act = self.model.act(inp, with_encoder=False)
                left, right = act[0,0].item(), act[0,1].item()

            print(f"{self.mode:<15} step {step:04d}  L={left:+.3f}  R={right:+.3f}")
            step += 1
            time.sleep(max(0, self.period - (time.time() - tic)))

    # ---------- stdin listener (mode + steer commands) -------------
    def _stdin_listener(self):
        while True:
            key = sys.stdin.readline().strip().lower()
            if key == "c":
                self.mode = "sit_on_chair"
            elif key == "p":
                self.mode = "follow_path"
            elif key == "h":
                self.mode = "reach_with_hand"
            elif key == "t":
                self.mode = "steer"
                self.steer_cmd = [0.0, 0.0]
            elif key in {"i","j","k","l"} and self.mode == "steer":
                self._update_steer_cmd(key)
            else:
                self.mode = "idle"
            print(f"\n>>> mode = {self.mode}\n")

    def _update_steer_cmd(self, key):
        if key == "i":
            self.steer_cmd = [ 0.8,  0.8]
        elif key == "k":
            self.steer_cmd = [ 0.0,  0.0]
        elif key == "j":
            self.steer_cmd = [-0.3,  0.3]
        elif key == "l":
            self.steer_cmd = [ 0.3, -0.3]

    # ---------- Hydra / model --------------------------------------
    @staticmethod
    def _register_resolvers():
        OmegaConf.register_new_resolver(
            "eval", lambda e: eval(e, {"math": math, **vars(builtins)}, {}),
            replace=True)
        OmegaConf.register_new_resolver("len", lambda x: len(x), replace=True)

    def _load_cfg(self):
        cfg_dir, cfg_name = os.path.split(self.yaml)
        cfg_name = os.path.splitext(cfg_name)[0]
        os.chdir(self.root)
        with initialize_config_dir(config_dir=cfg_dir, version_base="1.1"):
            cfg = compose(config_name=cfg_name)
        OmegaConf.resolve(cfg)
        print("✓  YAML loaded & resolved.")
        return cfg

    def _extract_dims(self):
        E, MM = self.cfg.env.config, self.cfg.env.config.masked_mimic
        self.OBS_DIM   = E.humanoid_obs.obs_size
        self.LATENT    = self.cfg.agent.config.vae.latent_dim
        self.FUT_N     = MM.masked_mimic_target_pose.num_future_steps + 1
        self.SPARSE    = MM.masked_mimic_target_pose.num_obs_per_sparse_target_pose
        self.HIST_STEPS= MM.historical_obs.num_historical_conditioned_steps
        self.HIST_DIM  = (self.OBS_DIM + 1) * self.HIST_STEPS
        self.MIMIC_DIM = (E.mimic_target_pose.num_future_steps *
                          E.mimic_target_pose.num_obs_per_target_pose)
        self.TERRAIN   = self.cfg.terrain.config.terrain_obs_num_samples
        self.MOTION_DIM= MM.motion_text_embeddings.embedding_dim

    def _build_model(self):
        self.model = instantiate(self.cfg.agent.config.model)
        self.model.load_state_dict(torch.load(self.ckpt,
                                              map_location="cpu")["model"],
                                   strict=False)
        self.cuda   = torch.cuda.is_available()
        self.device = "cuda" if self.cuda else "cpu"
        self.model.eval().to(self.device).half()
        print("✓  Model ready on", self.device)

    def _reset_buffers(self):
        self.hist_buf = collections.deque(maxlen=self.HIST_STEPS)
        self.clip_tokens = {
            "sit_on_chair"   : self._fake_clip_embed("sit on chair"),
            "follow_path"    : self._fake_clip_embed("follow the path"),
            "reach_with_hand": self._fake_clip_embed("reach target with hand"),
        }

    # ---------- random data stubs ----------------------------------
    def _rand_vec(self, n, scale=1.):
        return [random.uniform(-scale, scale) for _ in range(n)]

    def _read_sensors_random(self):
        return {
            "height":    random.uniform(0.9, 1.1),
            "imu_lin":   self._rand_vec(3, 0.5),
            "imu_ang":   self._rand_vec(3, 1.0),
            "wheel_enc": self._rand_vec(2, 2.0),
        }

    def _get_chair_pose_random(self):
        pos  = self._rand_vec(3, 0.5)
        quat = np.random.randn(4); quat /= np.linalg.norm(quat)
        return pos + quat.tolist()

    def _get_path_waypoints_random(self):
        return [self._rand_vec(3, 2.0) for _ in range(self.FUT_N)]

    def _get_hand_goal_pose_random(self):
        pos  = self._rand_vec(3, 0.6)
        quat = np.random.randn(4); quat /= np.linalg.norm(quat)
        return pos + quat.tolist()

    # ---------- CLIP stub -----------------------------------------
    def _fake_clip_embed(self, text):
        torch.manual_seed(abs(hash(text)) % 2**31)
        vec = torch.randn(self.MOTION_DIM)
        return (vec / vec.norm()).to(self.device).half()

    # ---------- observation helpers -------------------------------
    def _make_self_obs(self, s):
        v = torch.zeros(self.OBS_DIM, device=self.device).half()
        v[0]   = s["height"]
        v[1:4] = torch.tensor(s["imu_lin"], device=self.device).half()
        v[4:7] = torch.tensor(s["imu_ang"], device=self.device).half()
        v[7:11]= torch.tensor([1,0,0,0], device=self.device).half()
        v[11:13]= torch.tensor(s["wheel_enc"], device=self.device).half()
        return v

    def _update_history(self, self_obs):
        sample = torch.cat([self_obs,
                            torch.zeros(1, device=self.device).half()])
        self.hist_buf.append(sample)
        if len(self.hist_buf) < self.HIST_STEPS:
            pad = [torch.zeros_like(sample)] * (self.HIST_STEPS - len(self.hist_buf))
            seq = pad + list(self.hist_buf)
        else:
            seq = list(self.hist_buf)
        return torch.cat(seq)

    # ---------- build input dict per mode -------------------------
    def _base_dict(self, self_obs, hist):
        z = lambda n: torch.zeros(1, n, device=self.device).half()
        return {
            "self_obs":            self_obs.unsqueeze(0),
            "vae_noise":           z(self.LATENT),
            "mimic_target_poses":  z(self.MIMIC_DIM),
            "historical_pose_obs": hist.unsqueeze(0),
            "terrain":             z(self.TERRAIN),
        }

    def _build_input_dict(self, self_obs, hist):
        if self.mode == "sit_on_chair":
            chair_T = self._get_chair_pose_random()
            sparse = torch.zeros(self.SPARSE * self.FUT_N, device=self.device).half()
            sparse[:7] = torch.tensor(chair_T, device=self.device).half()
            d = self._base_dict(self_obs, hist)
            d.update({
                "masked_mimic_target_poses":       sparse.unsqueeze(0),
                "masked_mimic_target_poses_masks": torch.ones(1, self.FUT_N,
                                                              device=self.device).half(),
                "motion_text_embeddings":          self.clip_tokens["sit_on_chair"].unsqueeze(0),
                "motion_text_embeddings_mask":     torch.ones(1,1, device=self.device).half(),
            })
            return d

        elif self.mode == "follow_path":
            path_flat = torch.tensor(np.array(self._get_path_waypoints_random()).flatten(),
                                     device=self.device).half()
            need = self.SPARSE * self.FUT_N
            if path_flat.numel() < need:
                path_flat = torch.cat([path_flat,
                                       torch.zeros(need - path_flat.numel(),
                                                   device=self.device).half()])
            d = self._base_dict(self_obs, hist)
            d.update({
                "masked_mimic_target_poses":       path_flat.unsqueeze(0),
                "masked_mimic_target_poses_masks": torch.ones(1, self.FUT_N,
                                                              device=self.device).half(),
                "motion_text_embeddings":          self.clip_tokens["follow_path"].unsqueeze(0),
                "motion_text_embeddings_mask":     torch.ones(1,1, device=self.device).half(),
            })
            return d

        elif self.mode == "reach_with_hand":
            hand_T = torch.tensor(self._get_hand_goal_pose_random(),
                                  device=self.device).half().unsqueeze(0)
            d = self._base_dict(self_obs, hist)
            d.update({
                "right_hand_goal_pose":            hand_T,
                "right_hand_goal_pose_mask":       torch.ones(1,1, device=self.device).half(),
                "motion_text_embeddings":          self.clip_tokens["reach_with_hand"].unsqueeze(0),
                "motion_text_embeddings_mask":     torch.ones(1,1, device=self.device).half(),
            })
            return d

        else:  # should not reach here
            raise RuntimeError("invalid mode")


# -------------------------------------------------------------------
if __name__ == "__main__":
    MaskedMimicNode(loop_hz=50.0).run()
