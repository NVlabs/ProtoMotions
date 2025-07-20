#!/usr/bin/env python3
# ------------------------------------------------------------------
# masked_mimic_node.py – idle / sit / path / reach / steer
# with Jetson‑friendly FP16 (+optional INT8) checkpoint
# ------------------------------------------------------------------
#
#   keys:  c chair · p path · h hand · t steer(i j k l) · others idle
#
# Parameters (ROS‑style) are provided through `params` dict:
#   root, ckpt_rel, yaml_rel, loop_hz, init_mode,
#   use_cuda   (auto / cpu / cuda),
#   enable_int8 (bool),
#   convert_to_fp16 (bool)
#   use_sim_time (bool),
#   qos_in_mode  (0 default | 1 reliable | 2 best‑effort),
#   qos_out_mode (same)
# ------------------------------------------------------------------

import os, sys, time, math, builtins, random, collections, threading
import torch, numpy as np
from contextlib import contextmanager
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from hydra.utils import instantiate

# ═════════════ utilities ══════════════
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
# ══════════════════════════════════════


def default_params() -> dict:
    """All run‑time options collected here for easy ROS2 mapping."""
    return dict(
        # file paths ------------------------------------------------
        root        = "/home/rover2/OrcaRL/ProtoMotions",
        ckpt_rel    = "data/pretrained_models/masked_mimic/smpl/last.ckpt",
        yaml_rel    = "data/pretrained_models/masked_mimic/smpl/config.yaml",

        # execution -------------------------------------------------
        loop_hz     = 10,
        init_mode   = "idle",            # idle | sit_on_chair | follow_path | …
        use_cuda    = "auto",            # auto | cuda | cpu
        enable_int8 = True,              # dynamic Linear→INT8 on CPU
        convert_to_fp16 = True,         # build / use FP16 checkpoint?

        # “ROS2” parameters kept for future launch files ------------
        use_sim_time = False,
        qos_in_mode  = 0,                # 0 default, 1 reliable, 2 best‑effort
        qos_out_mode = 0,
    )


class MaskedMimicNode:
    """Keyboard‑selectable behaviours + parameterizable execution."""

    # ─────────── init ───────────
    def __init__(self, params: dict | None = None):

        P = default_params() if params is None else {**default_params(), **params}

        # ---------------- paths / timing -----------------
        self.root       = P["root"]
        self.ckpt_full  = os.path.join(self.root, P["ckpt_rel"])
        self.ckpt_fp16  = self.ckpt_full.replace(".ckpt", "_fp16.pt")
        self.yaml       = os.path.join(self.root, P["yaml_rel"])
        self.period     = 1.0 / P["loop_hz"]

        # ROS‑style bookkeeping (not used yet) ------------
        self.use_sim_time = P["use_sim_time"]
        self.qos_in_mode  = P["qos_in_mode"]
        self.qos_out_mode = P["qos_out_mode"]

        # ---------------- config & model -----------------
        self._register_resolvers()
        self.cfg = self._load_cfg()
        self._extract_dims()
        self._build_or_load_model(force_device=P["use_cuda"],
                                  enable_int8=P["enable_int8"],
                                  convert_fp16=P["convert_to_fp16"])
        self._reset_buffers()

        # ---------------- runtime state ------------------
        self.mode = P["init_mode"]
        self.steer_cmd = [0.0, 0.0]
        threading.Thread(target=self._stdin_listener, daemon=True).start()

        print("🌟  MaskedMimicNode ready.  "
              "c chair · p path · h hand · t steer(i j k l) · others idle\n")

    # ─────────────────────────── main loop ───────────────────────────
    def run(self):
        step = 0
        while True:
            tic = time.time()

            sensor   = self._sensor_stub()
            self_obs = self._make_self_obs(sensor)
            hist     = self._update_hist(self_obs)

            if self.mode == "idle":
                left, right = 0.0, 0.0

            elif self.mode == "steer":
                left, right = self.steer_cmd

            else:
                inp = self._build_input_dict(self_obs, hist)
                with torch.no_grad(), autocast_cuda(self.cuda):
                    act = self.model.act(inp, with_encoder=False)
                left, right = act[0, 0].item(), act[0, 1].item()

            print(f"{self.mode:<15}  step {step:04d}  "
                  f"L={left:+.3f}  R={right:+.3f}")
            step += 1
            time.sleep(max(0, self.period - (time.time() - tic)))

    # ────────────────── keyboard interface ──────────────────
    def _stdin_listener(self):
        keymap = {"c":"sit_on_chair","p":"follow_path",
                  "h":"reach_with_hand","t":"steer"}
        while True:
            key = sys.stdin.readline().strip().lower()
            if key in keymap:
                self.mode = keymap[key]
                if self.mode == "steer":
                    self.steer_cmd = [0.0, 0.0]
            elif self.mode == "steer" and key in {"i","j","k","l"}:
                self.steer_cmd = {"i":[0.8,0.8],"k":[0,0],
                                  "j":[-0.3,0.3],"l":[0.3,-0.3]}[key]
            else:
                self.mode = "idle"
            print(f"\n>>> mode = {self.mode}\n")

    # ═══════════ Hydra helpers ════════════
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
        return cfg

    # ═══════════ model build / conversion ════════════
    def _build_or_load_model(self, *, force_device:str="auto",
                             enable_int8:bool=True,
                             convert_fp16:bool=False):
        """force_device: 'auto' | 'cuda' | 'cpu'"""
        # --- choose checkpoint -------------------------------------------------
        if convert_fp16:
            # use / create FP16‑compressed checkpoint
            slim = (torch.load(self.ckpt_fp16, map_location="cpu")
                    if os.path.exists(self.ckpt_fp16)
                    else self._convert_ckpt_to_fp16())
        else:
            # load original FP32 checkpoint untouched
            full = torch.load(self.ckpt_full, map_location="cpu")
            if "model" in full:
                weights = full["model"]
            elif "state_dict" in full:
                weights = {k.replace("model.", "", 1): v
                           for k,v in full["state_dict"].items()}
            else:
                raise KeyError("checkpoint missing 'model/state_dict'")
            slim = {"model": weights}

        # --- restore model -----------------------------------------------------
        self.model = instantiate(self.cfg.agent.config.model)
        self.model.load_state_dict(slim["model"], strict=False)

        # --- device selection & optional INT8 ----------------------------------
        self.cuda = (torch.cuda.is_available() if force_device=="auto"
                     else (force_device=="cuda"))
        if not self.cuda and enable_int8:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8)

        self.device = "cuda" if self.cuda else "cpu"
        self.model.eval().to(self.device)
        if convert_fp16:                       # cast to half if we asked for FP16
            self.model.half()
        print("✓  Model ready on", self.device,
              "(FP16)" if convert_fp16 else "(FP32)")

    def _convert_ckpt_to_fp16(self):
        print("⏳  Converting original checkpoint ➜ FP16 …")
        full = torch.load(self.ckpt_full, map_location="cpu")
        if "model" in full:
            weights = full["model"]
        elif "state_dict" in full:
            weights = {k.replace("model.", "", 1): v
                       for k,v in full["state_dict"].items()}
        else:
            raise KeyError("checkpoint missing 'model/state_dict'")

        for k in list(weights.keys()):
            weights[k] = weights[k].half()

        slim = {"model": weights}
        torch.save(slim, self.ckpt_fp16,
                   _use_new_zipfile_serialization=False)
        print(f"✓  FP16 weights saved: {os.path.basename(self.ckpt_fp16)}")
        return slim

    # ═══════════  dimension helpers ════════════
    def _extract_dims(self):
        E, MM = self.cfg.env.config, self.cfg.env.config.masked_mimic
        self.OBS_DIM   = E.humanoid_obs.obs_size
        self.LATENT    = self.cfg.agent.config.vae.latent_dim
        self.FUT_N     = MM.masked_mimic_target_pose.num_future_steps + 1
        self.SPARSE    = MM.masked_mimic_target_pose.num_obs_per_sparse_target_pose
        self.HIST_N    = MM.historical_obs.num_historical_conditioned_steps
        self.HIST_DIM  = (self.OBS_DIM + 1) * self.HIST_N
        self.MIMIC_DIM = (E.mimic_target_pose.num_future_steps *
                          E.mimic_target_pose.num_obs_per_target_pose)
        self.TERRAIN   = self.cfg.terrain.config.terrain_obs_num_samples
        self.MOTION_DIM= MM.motion_text_embeddings.embedding_dim

    # ═══════════  runtime buffers ════════════
    def _reset_buffers(self):
        self.hist_buf = collections.deque(maxlen=self.HIST_N)
        self.clip_tokens = {n: self._fake_clip(t)
                            for n,t in [("sit_on_chair","sit on chair"),
                                        ("follow_path","follow the path"),
                                        ("reach_with_hand","reach target with hand")]}

    # ═══════════ stubs (replace with real I/O) ════════════
    def _rand(self, n, s=1): return [random.uniform(-s,s) for _ in range(n)]
    def _sensor_stub(self): return dict(height=random.uniform(0.9,1.1),
                                        imu_lin=self._rand(3,0.5),
                                        imu_ang=self._rand(3,1.0),
                                        wheel_enc=self._rand(2,2))
    def _chair_stub(self): 
        return self._rand(3,0.5)+list(self._rand(4,1))
    
    def _path_stub(self):  
        return [self._rand(3,2) for _ in range(self.FUT_N)]
    
    def _hand_stub(self):  
        return self._rand(3,0.6)+list(self._rand(4,1))
    
    def _fake_clip(self, txt):
        torch.manual_seed(abs(hash(txt))&0xFFFFFFFF)
        v=torch.randn(self.MOTION_DIM)
        return (v/v.norm()).to(self.device).half()

    # ═══════════ observation & history ════════════
    def _make_self_obs(self,s):
        v=torch.zeros(self.OBS_DIM,device=self.device).half()
        v[0],v[1:4],v[4:7]=s["height"],torch.tensor(s["imu_lin"],device=self.device).half(),torch.tensor(s["imu_ang"],device=self.device).half()
        v[7:11]=torch.tensor([1,0,0,0],device=self.device).half()
        v[11:13]=torch.tensor(s["wheel_enc"],device=self.device).half()
        return v
    def _update_hist(self,so):
        sample=torch.cat([so,torch.zeros(1,device=self.device).half()])
        self.hist_buf.append(sample)
        if len(self.hist_buf)<self.HIST_N:
            pad=[torch.zeros_like(sample)]*(self.HIST_N-len(self.hist_buf))
            seq=pad+list(self.hist_buf)
        else: seq=list(self.hist_buf)
        return torch.cat(seq)

    # ═══════════ network input builders ════════════
    def _base(self,so,h): z=lambda n:torch.zeros(1,n,device=self.device).half();return{
        "self_obs":so.unsqueeze(0),"vae_noise":z(self.LATENT),
        "mimic_target_poses":z(self.MIMIC_DIM),
        "historical_pose_obs":h.unsqueeze(0),"terrain":z(self.TERRAIN)}

    def _build_input_dict(self,so,h):
        if self.mode=="sit_on_chair":
            sparse=torch.zeros(self.SPARSE*self.FUT_N,device=self.device).half()
            sparse[:7]=torch.tensor(self._chair_stub(),device=self.device).half()
            d=self._base(so,h);d.update({
                "masked_mimic_target_poses":sparse.unsqueeze(0),
                "masked_mimic_target_poses_masks":torch.ones(1,self.FUT_N,device=self.device).half(),
                "motion_text_embeddings":self.clip_tokens["sit_on_chair"].unsqueeze(0),
                "motion_text_embeddings_mask":torch.ones(1,1,device=self.device).half()})
            return d
        if self.mode=="follow_path":
            flat=torch.tensor(np.array(self._path_stub()).flatten(),device=self.device).half()
            need=self.SPARSE*self.FUT_N
            if flat.numel()<need:
                flat=torch.cat([flat,torch.zeros(need-flat.numel(),device=self.device).half()])
            d=self._base(so,h);d.update({
                "masked_mimic_target_poses":flat.unsqueeze(0),
                "masked_mimic_target_poses_masks":torch.ones(1,self.FUT_N,device=self.device).half(),
                "motion_text_embeddings":self.clip_tokens["follow_path"].unsqueeze(0),
                "motion_text_embeddings_mask":torch.ones(1,1,device=self.device).half()})
            return d
        hand=torch.tensor(self._hand_stub(),device=self.device).half().unsqueeze(0)
        d=self._base(so,h);d.update({
            "right_hand_goal_pose":hand,
            "right_hand_goal_pose_mask":torch.ones(1,1,device=self.device).half(),
            "motion_text_embeddings":self.clip_tokens["reach_with_hand"].unsqueeze(0),
            "motion_text_embeddings_mask":torch.ones(1,1,device=self.device).half()})
        return d


# --------------------------------------------------------------------
if __name__ == "__main__":
    # Nodes can now be launched with custom params, e.g. from ROS 2:
    # node = MaskedMimicNode(params_from_rclpy)
    MaskedMimicNode().run()
