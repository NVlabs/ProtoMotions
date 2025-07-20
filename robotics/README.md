Below is a **step‑by‑step guide** (with code snippets you can paste into
`test.py`) that shows **exactly where to insert your real sensor values
and what each field means.**  You only need to replace the content of the
`dummy` dict; the model, masks, and batch layout stay exactly the same.

---

## 1 · What each tensor represents

| key in `dummy`                    | shape *(batch = 1)* | where it comes from in sim                                                         | what you feed on the real robot                                                                                                                                       |
| --------------------------------- | ------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `self_obs`                        | `(1, 358)`          | **root & joint state** – 3‑D root pos / vel, joint angles / vel, root height, etc. | Concatenate: ① root‑frame linear & angular vel from IMU, ② wheel/joint encoder positions + vels, ③ (optionally) a root‑height estimate. Keep order fixed (see below). |
| `vae_noise`                       | `(1, 64)`           | random noise during training                                                       | Leave **all zeros** for deterministic behaviour, or `torch.randn_like()` if you want motion diversity.                                                                |
| `mimic_target_poses`              | `(1, 6495)`         | future SMPL target poses from motion library                                       | Unless you’ve built a **target‑trajectory generator**, keep zeros.                                                                                                    |
| `masked_mimic_target_poses`       | `(1, 184 × 11)`     | sparse future poses (one token per future step)                                    | Same as above – zeros are safe.                                                                                                                                       |
| `masked_mimic_target_poses_masks` | `(1, 11)`           | 1 = token visible, 0 = masked                                                      | All zeros (meaning “mask everything”) is valid.                                                                                                                       |
| `historical_pose_obs`             | `(1, 5385)`         | stack of the last 15 self‑obs + phase flag                                         | Build a **rolling FIFO buffer** of your `self_obs`.  First run fill with zeros.                                                                                       |
| `motion_text_embeddings`          | `(1, 512)`          | CLIP text embedding of the current motion label                                    | If you don’t use text, keep zeros.                                                                                                                                    |
| `motion_text_embeddings_mask`     | `(1, 1)`            | should this token be visible?                                                      | Zero (masked).                                                                                                                                                        |
| `terrain`                         | `(1, 256)`          | 16 × 16 height samples under each foot                                             | If you have no depth map / height grid, keep zeros.                                                                                                                   |

Only **`self_obs`** and **`historical_pose_obs`** must contain meaningful
real data for the controller to react; the rest can stay zeros if you’re
not using the advanced conditioning.

---

## 2 · Packing `self_obs` (358 floats)

The default SMPL layout is:

| section                | dim       | build from robot …                                |
| ---------------------- | --------- | ------------------------------------------------- |
| root height (`z`)      |  1        | barometer / ground‑clearance estimator *(else 0)* |
| root lin vel (x y z)   |  3        | IMU or differentiated odometry                    |
| root ang vel (xyz)     |  3        | IMU gyro                                          |
| root rot (quat)        |  4        | fused roll‑pitch‑yaw → quaternion                 |
| joint angles           |  138      | each joint pos in **radians** (69 DOFs)           |
| joint velocities       |  138      | each joint vel in **rad/s**                       |
| foot contact flags     |  ?        | leave 0 if N/A                                    |
| any extra SMPL markers | remainder | leave 0                                           |

**Minimal viable mapping for a 2‑wheel robot**

If you only have wheels + IMU:

```python
# wheel_enc is length 2, imu_lin_vel is 3‑D, imu_ang_vel is 3‑D
self_obs_vec = torch.zeros(OBS_DIM, device=device)
self_obs_vec[0]  = root_height_est                # or 0
self_obs_vec[1:4] = torch.tensor(imu_lin_vel)
self_obs_vec[4:7] = torch.tensor(imu_ang_vel)
self_obs_vec[7:11]= torch.tensor([1,0,0,0])       # identity quat
self_obs_vec[11:13] = torch.tensor(wheel_enc)     # first 2 joint pos
# leave the rest zeros
```

The controller was trained on a humanoid, so its semantics won’t match
your two‑wheeler perfectly, but it **will still output a 69‑D action
vector**; you’ll pick two entries (e.g. the first two) and map them to
left / right motor PWM.

---

## 3 · Rolling `historical_pose_obs`

Add this at the top of your script:

```python
from collections import deque
hist_buffer = deque(maxlen=HIST_STEPS)   # stores latest self_obs
```

Inside the loop (just before you build `dummy`):

```python
hist_buffer.append(self_obs_vec)
if len(hist_buffer) < HIST_STEPS:
    # pad with zeros until buffer fills
    padded = [torch.zeros_like(self_obs_vec)]*(HIST_STEPS-len(hist_buffer)) + \
             list(hist_buffer)
else:
    padded = list(hist_buffer)

historical = torch.cat(padded + [torch.zeros(1, device=device)])  # +phase flag
```

Now `historical` is `(5385,)` and you can set

```python
dummy["historical_pose_obs"] = historical.unsqueeze(0).half()
```

---

## 4 · Putting it together – replace step 5 in your script

```python
def read_sensors():
    """Return a dict with your live sensor values."""
    return {
        "wheel_enc":  get_wheel_positions(),   # 2‑element list
        "imu_lin":    get_imu_linear_vel(),    # 3‑element
        "imu_ang":    get_imu_angular_vel(),   # 3‑element
        "height":     estimate_root_height(),  # float
    }

hist_buffer = deque(maxlen=HIST_STEPS)

for step in range(10_000):          # run forever
    s = read_sensors()

    # --- build self_obs ------------------------------------
    self_obs = torch.zeros(OBS_DIM, device=device)
    self_obs[0]  = s["height"]
    self_obs[1:4] = torch.tensor(s["imu_lin"], device=device)
    self_obs[4:7] = torch.tensor(s["imu_ang"], device=device)
    self_obs[7:11]= torch.tensor([1,0,0,0], device=device)  # no orientation
    self_obs[11:13]= torch.tensor(s["wheel_enc"], device=device)
    # rest already zero
    self_obs = self_obs.half()

    # --- historical buffer ---------------------------------
    hist_buffer.append(self_obs)
    padded = (HIST_STEPS-len(hist_buffer))*[torch.zeros_like(self_obs)] \
             + list(hist_buffer)
    historical = torch.cat(padded + [torch.zeros(1, device=device)])  # +phase

    # --- vae noise (zero = deterministic) -------------------
    vae_noise = torch.zeros(LATENT, device=device).half()

    # --- build input dict ----------------------------------
    inp = {
        "self_obs":                       self_obs.unsqueeze(0),
        "vae_noise":                      vae_noise.unsqueeze(0),
        "mimic_target_poses":             torch.zeros(1, MIMIC_DIM, device=device).half(),
        "masked_mimic_target_poses":      torch.zeros(1, SPARSE*FUT_N, device=device).half(),
        "masked_mimic_target_poses_masks":torch.zeros(1, FUT_N, device=device).half(),
        "historical_pose_obs":            historical.unsqueeze(0),
        "motion_text_embeddings":         torch.zeros(1, MOTION_DIM, device=device).half(),
        "motion_text_embeddings_mask":    torch.zeros(1, 1, device=device).half(),
        "terrain":                        torch.zeros(1, TERRAIN, device=device).half(),
    }

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device=="cuda"):
        act = model.act(inp, with_encoder=False)

    # take first two action entries for wheel speeds
    left_cmd, right_cmd = act[0, 0].item(), act[0, 1].item()
    set_motor_pwm(left_cmd, right_cmd)

    time.sleep(0.02)   # 50 Hz loop
```

---

## 5 · Mapping actions to motors

During training the actions are `tanh`‑clamped to ±1.  Map that to your
PWM range:

```python
def set_motor_pwm(left, right):
    pwm_l = int( 80 * left )   # scale to ±80 % duty
    pwm_r = int( 80 * right)
    motor_driver.set(pwm_l, pwm_r)
```

---

### Recap

1. **Fill `self_obs`** with your IMU + encoder data (358‑vector).
2. **Maintain a deque** of the last 15 `self_obs` to build
   `historical_pose_obs`.
3. Keep all conditioning tokens and masks **zeroed** until you implement
   a proper motion‑target generator.
4. Use the **first two entries of the 69‑D action** as left/right wheel
   commands (or whichever mapping you decide).

With those changes the same script that now prints dummy actions will
drive your real robot.  Let me know when you have sensor code ready and
we can fine‑tune the exact ordering of the 358‑D vector!



### Yes — that’s the clean architecture

```
sensors → Foundation‑Pose  ─┐
                            │  world‑frame chair pose (6 DoF)
                            ▼
               Goal‑Generator/Planner  ──►  humanoid‑frame “body goal”
                            │
                            ▼
           Masked‑Mimic actor (this node)
                            │
                            ▼
                     wheel PWMs / joints
```

| layer                        | purpose                                                                                                                                                                                                  | runs at                     |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| **Foundation‑Pose**          | Extracts the *current* 6‑DoF pose of the chair (or any object) from RGB/RGB‑D.                                                                                                                           | 5–10 Hz (camera‑limited)    |
| **Goal‑generator / planner** | Transforms the chair pose into a **body‑centric target**: e.g. desired root position/orientation and sitting posture over the next 2–3 s.  Can do collision checks, path smoothing, foot placement, etc. | 10–20 Hz                    |
| **Masked‑Mimic actor**       | Consumes that short‑horizon body goal plus local state & history, and outputs low‑level actions.                                                                                                         | 50 Hz (already in the node) |

Why this split works:

* **Interpretability** – you can visualise/intercept the high‑level goal trajectory without touching the RL policy.
* **Re‑usability** – the same actor can “sit”, “step over”, “duck” by just changing the goal generator + text token.
* **Safety** – the planner can keep constraints (avoid table edge) before the RL layer.

---

## How to wire it into the current node

1. **Remove the random chair pose code.**

```python
chair_pose = foundation_pose_api()          # returns pos+quat
body_goal  = goal_generator(chair_pose)     # your new module
inp["object_6d_pose"] = body_goal.unsqueeze(0).half()
inp["object_6d_pose_mask"].fill_(1.)
```

*`body_goal` could be a 7‑vector or a longer horizon flattened to match `masked_mimic_target_poses`.*

2. **Keep the text embedding.**
   The `motion_text_embeddings` token still provides the semantic cue (“sit on chair”), while the numeric goal gives geometry.

3. **Timestep alignment.**

   * Planner (say 10 Hz) publishes the latest goal.
   * The 50 Hz control loop just reads the most recent goal every cycle; hold last value between planner updates.

4. **Debug first with playback.**
   Record a chair pose → goal trajectory in a bag/file, feed it to the node, verify the robot lowers and rotates correctly, then switch to live.

---

## Existing open‑source pieces you can reuse

* **Foundation‑Pose** (object pose)
  → already gives you 6‑DoF in world frame.
* **`spatial‑ai/choreo‑planners`** (task‑space key‑pose planner)
  → takes an object pose and emits a spline of root targets for sitting/standing.
* **Your current node** (low‑level RL control).

Glue them with a tiny Python pub‑sub (ZeroMQ, DDS, or even files) before moving to full ROS2.

---

### Bottom line

> **Yes**—let Foundation‑Pose handle perception, a lightweight planner
> translate that into a *body‑goal trajectory*, and keep Masked‑Mimic
> strictly for reflex‑style tracking.
> This mirrors the hierarchy used in the original ProtoMotions papers
> (perception → mid‑level planning → RL control) and scales well to new
> tasks like “step onto stair” or “duck under bar” without retraining
> the low‑level policy.
