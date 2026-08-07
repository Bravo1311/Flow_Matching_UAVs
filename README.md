# Flow-Matching Landing Policy for a Simulated Drone

A from-scratch pipeline that trains a **Diffusion Transformer-style conditional flow-matching transformer** to imitate a classical **PD landing controller**, using synthetic data generated in a simplified **MuJoCo** quadrotor simulation. Built as a scoped proof-of-concept, not a production system. Every simplification below is deliberate and documented.

## Motivation

Modern VLA robot policies, e.g. Physical Intelligence's π0, use flow-matching "action experts" to generate short chunks of continuous control actions, conditioned on a semantic backbone. This project reproduces that architecture at small scale: no vision, no language, just pose-conditioned continuous control, to build and understand every piece of the pipeline end-to-end: simulation, control, synthetic data generation, and generative policy training.

**Explicit, non-overclaimed goal**: demonstrate a working-flow-matching imitation-learning pipeline that reproduces PD-level landing behaviour. This project does **not** claim the learned policy outperforms PD - pure behavior cloning from a PD teacher cannot exceed its teacher's behavior by construction.

## Pipeline Overview

```
MuJoCo sim (simplified dynamics)
        │
        ▼
PD landing controller  ──► synthetic data collection (randomized episodes)
        │                          │
        │                          ▼
        │                  data/episodes/*.json
        │                          │
        │                          ▼
        │                  train/val split (by episode)
        │                          │
        │                          ▼
        │             DiT-style flow-matching transformer
        │                          │
        │                          ▼
        │                    trained policy
        │                          │
        ▼                          ▼
  PD baseline eval        policy closed-loop eval (MuJoCo)
```
 

 ## Simulation Design

 - **Model**: MuJoCo Menagerie's Skydio X2 quadrotor, with a static landing-marker body added to the scene.
 - **Dynamcis**: deliberately simplified - no rotor thrust/torque mixer. The policy (PD or learned) outputs `[vx, vy, vz, yaw_rate]` directly; a first-order lag model (`v_actual += (cmd - v_actual) * dt / TAU`) approximates real quadrotor response timme.
 - **Gravity**: enabled, with a constant per-step hover-bias term added to `vz` to cancel gravity's effect on `qvel`, so commanding zero velocity requires the same "active hovering" real drones exhibit — a deliberate choice over the simpler (but less realistic) zero-gravity shortcut.
- **Attitude stabilization**: roll and pitch are *not* part of either policy's action space. A fixed, always-on proportional "leveling" controller holds roll/pitch near zero every step, independent of what generates `[vx,vy,vz,yaw_rate]` — structurally mirroring how a real flight controller's inner attitude loop runs underneath velocity-mode commands (e.g. PX4 offboard velocity control).
- **Known simplification**: no rotor-level physics, no thrust mixer, no wind/disturbance modeling. This is a controls-and-learning testbed, not a flight-dynamics-accurate simulator.

## PD baseline controller
 
- PD on x/y (shared gains) and z (separate gains), P-only on yaw — mirroring a real PX4-based ArUco-landing controller previously implemented in [ROS2-PX4 UAV Autonomy Project](https://github.com/Bravo1311/ROS2_PX4_Drone_Autonomy_POC).
- **Derivative terms use finite-differenced error**, not privileged simulator velocity — deliberately modeling what a real perception-only system would have access to (matches the real PX4 implementation's constraint of only receiving `PoseStamped` marker detections).
- Landing success: horizontal error `< 0.1m` and altitude within `0.05m` of a target landing height, sustained for 20 consecutive simulation steps (guards against false-triggering on a momentary pass-through).
## Synthetic data generation
 
- Episodes spawn the drone at a randomized offset (`±6m` xy, `4-6m` altitude) with a randomized yaw (uniform, full circle) relative to a fixed marker.
- The PD controller runs closed-loop to completion; `(relative_pos, relative_quat, cmd_vel)` is logged at every timestep.
- Only successful episodes are saved — failed/timed-out episodes are discarded so the policy never imitates failure.
## Flow-matching policy
 
**Input (condition)**: a short history of `H=4` past poses (`relative_pos` + `relative_quat`, 7-dim each), encoded by a small Transformer encoder and mean-pooled into a single conditioning vector. Velocity is deliberately *not* given directly — matching the PD controller's real-world perception constraint — the policy must learn to infer motion trends from the pose history itself.
 
**Output (generated)**: an action chunk of `C=8` future `[vx,vy,vz,yaw_rate]` commands, generated jointly via flow matching (not autoregressively).
 
**Architecture** (DiT-style, matching production flow-matching action-experts):
- Sinusoidal embedding for the flow-matching timestep `t ∈ [0,1]`.
- Learned positional embeddings for both the action-chunk tokens and the history tokens.
- **AdaLN (adaptive layer norm)** conditioning: the combined (history + timestep) embedding modulates each transformer block's LayerNorm scale/shift, rather than being concatenated into the token sequence.
- Several DiT blocks (self-attention + MLP, each preceded by AdaLN) process the action-chunk tokens; the history is fully compressed into the conditioning vector before this stage — the two token sequences never directly attend to each other.
**Training objective**: standard conditional flow matching — sample noise `x₀`, interpolate `x_t = (1-t)x₀ + t·x₁` against a real demonstrated action chunk `x₁`, and regress the network's output against the constant target velocity `x₁ - x₀`.
 
**Inference**: Euler integration from pure noise (10 steps by default), executing the first `STEPS_PER_CHUNK=3` actions of the generated chunk before replanning against fresh pose data (a middle ground between fully reactive per-step replanning and fully open-loop chunk execution).
 
## Repository structure
 
```
flow_matching_landing/
├── assets/                # MuJoCo scene assets
├── mujoco_menagerie/       # cloned drone model (external, gitignored)
├── sim/                    # dynamics, teleop, pose computation
├── control/                # PD controller
├── model/flow_matching_v1/ # dataset, transformer, flow matching, inference, config
├── scripts/                 # collect_data.py, train.py, evaluate_policy.py
├── data/episodes/            # synthetic dataset (gitignored)
└── checkpoints/               # trained weights (gitignored)
```
 
## Key engineering lessons from this build
 
- **Sign conventions must be re-derived, not assumed, whenever a coordinate frame changes.** Flipping `relative_pos` from drone-relative to marker-relative broke every downstream sign-dependent formula (x/y PD, z PD, yaw) simultaneously — a good example of why a single convention change needs a full audit, not a patch.
- **Piecewise control laws need continuity at their switch points.** An early z-controller that switched between PD and an exponential decay term introduced a real discontinuity in commanded velocity at the switch boundary — visible as a late-trajectory spike in logged data. Root-caused by comparing logged `cmd_vel` values directly rather than assuming the bug was elsewhere.
- **Train/val leakage happens silently at the framing level, not the code level.** Splitting a sliding-window dataset by individual `(episode, t)` examples — rather than by whole episodes — leaks near-identical adjacent-timestep windows across the split, producing artificially good validation numbers without any obviously "wrong" line of code.
- **Simulation timing must be explicitly decoupled from wall-clock time**, and re-coupled deliberately for visual debugging (`time.sleep` pacing) vs. bulk data collection (deliberately unthrottled).
## Explicit non-goals / future work
 
- **Not** vision-conditioned — pose input is privileged simulator state, not detected from images. A natural v2 would swap the condition encoder for a CNN over rendered camera frames.
- **Not** validated to outperform PD — pure imitation learning cannot exceed its teacher. RL fine-tuning on top of a BC-initialized policy is the documented path to a genuinely better-than-PD policy, deliberately deferred.
- **Not** tested under disturbance/wind or sensor noise.
- A pose-conditioned vs. vision-conditioned comparison study is a natural, well-scoped extension once both variants exist.
## Setup
 
```bash
python3 -m venv venv
source venv/bin/activate
pip install mujoco numpy scipy mujoco-python-viewer torch tqdm
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```
 
## Usage
 
```bash
# Collect synthetic PD demonstration data
python3 -m scripts.collect_data
 
# Train the flow-matching policy
python3 -m scripts.train
 
# Evaluate closed-loop, headless (success-rate statistics)
python3 -m scripts.evaluate_policy
 
# Watch a few episodes with the MuJoCo viewer
python3 -m scripts.evaluate_policy --visualize 5
```
 