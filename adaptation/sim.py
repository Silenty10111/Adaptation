#!/usr/bin/env python3
"""Isaac Gym simulation infrastructure — _RobotSimCtx + gait constants.

Extracted from batch_test.py for reuse across experiment scripts.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Gait constants ───────────────────────────────────────────────────────
GAIT_FREQUENCY    = 0.85
SWING_AMP         = 0.32   # ↑ 0.26→0.32: prevent near-zero motion on low-amp morphologies
SWING_LIFT_RATIO  = 0.78
STANCE_LIFT_RATIO = 0.05
SWING_DROP_RATIO  = 0.38
STANCE_DROP_RATIO = 0.90
BODY_HEIGHT       = 0.50
HOLD_STEPS        = 300

# ── _RobotSimCtx ─────────────────────────────────────────────────────────

class _RobotSimCtx:
    """One Isaac Gym simulator per robot, reused across all probe episodes.

    Root cause of segfault (Exit 139):
        Each run_gait_sim call previously did create_sim / destroy_sim.
        After ~100 such cycles in one process, PhysX GPU memory gets
        corrupted → segfault.

    Fix:
        _RobotSimCtx creates the simulator ONCE per robot, runs the sag-
        controller warm-up ONCE, then resets DOF positions + all velocities
        between probe episodes without touching create_sim / destroy_sim.
    """

    def __init__(self, description: dict, urdf_path: Path, use_gpu: bool = True):
        from isaacgym import gymapi as _ga
        self._ga = _ga

        gym = _ga.acquire_gym()
        sp = _ga.SimParams()
        sp.up_axis   = _ga.UP_AXIS_Z
        sp.gravity   = _ga.Vec3(0.0, 0.0, -9.81)
        sp.dt        = 1.0 / 60.0
        sp.substeps  = 2
        sp.physx.use_gpu = use_gpu
        sp.physx.num_position_iterations = 8
        sp.physx.num_velocity_iterations = 2
        gpu_dev = 0 if use_gpu else -1
        sim = gym.create_sim(0, gpu_dev, _ga.SIM_PHYSX, sp)
        if sim is None:
            raise RuntimeError("_RobotSimCtx: create_sim failed")

        pp = _ga.PlaneParams()
        pp.normal = _ga.Vec3(0.0, 0.0, 1.0)
        pp.static_friction = 1.8; pp.dynamic_friction = 1.6; pp.restitution = 0.0
        gym.add_ground(sim, pp)

        ao = _ga.AssetOptions()
        ao.default_dof_drive_mode = int(_ga.DOF_MODE_POS)
        ao.fix_base_link = False
        ao.collapse_fixed_joints = True
        urdf_path = urdf_path.resolve()
        asset = gym.load_asset(sim, str(urdf_path.parent), urdf_path.name, ao)
        if asset is None:
            gym.destroy_sim(sim)
            raise RuntimeError("_RobotSimCtx: load_asset failed")

        env   = gym.create_env(sim, _ga.Vec3(-3, -3, 0), _ga.Vec3(3, 3, 2), 1)
        pose  = _ga.Transform()
        pose.p = _ga.Vec3(0.0, 0.0, BODY_HEIGHT)
        actor = gym.create_actor(env, asset, pose, "r", 0, 1)

        dof_props = gym.get_actor_dof_properties(env, actor)
        dof_props["driveMode"].fill(_ga.DOF_MODE_POS)
        dof_props["stiffness"].fill(200.0)
        dof_props["damping"].fill(20.0)
        if "effort"   in dof_props.dtype.names: dof_props["effort"].fill(1000.0)
        if "armature" in dof_props.dtype.names: dof_props["armature"].fill(0.01)
        dof_names = gym.get_asset_dof_names(asset)
        for idx, name in enumerate(dof_names):
            if "_swing" in name:
                dof_props["stiffness"][idx] = 100.0; dof_props["damping"][idx] = 10.0
            elif "_drop" in name:
                dof_props["stiffness"][idx] = 250.0; dof_props["damping"][idx] = 25.0
        gym.set_actor_dof_properties(env, actor, dof_props)

        lower    = np.where(np.isfinite(dof_props["lower"]), dof_props["lower"], -0.5).astype(np.float32)
        upper    = np.where(np.isfinite(dof_props["upper"]), dof_props["upper"],  0.5).astype(np.float32)
        finite_lo = np.where(np.isfinite(dof_props["lower"]), dof_props["lower"], -1e9)
        finite_hi = np.where(np.isfinite(dof_props["upper"]), dof_props["upper"],  1e9)

        name2idx = {n: i for i, n in enumerate(dof_names)}
        triplets: Dict[int, dict] = {}
        for lid in range(int(description.get("num_legs", 0))):
            ln, sn, dn = f"leg_{lid}_lift", f"leg_{lid}_swing", f"leg_{lid}_drop"
            if ln not in name2idx:
                continue
            triplets[lid] = {
                "lift_idx":    name2idx[ln], "swing_idx": name2idx[sn], "drop_idx":  name2idx[dn],
                "lift_lower":  float(lower[name2idx[ln]]), "lift_upper":  float(upper[name2idx[ln]]),
                "swing_lower": float(lower[name2idx[sn]]), "swing_upper": float(upper[name2idx[sn]]),
                "drop_lower":  float(lower[name2idx[dn]]), "drop_upper":  float(upper[name2idx[dn]]),
            }

        foot_z_vals = [float(lk["default_world_origin"][2])
                       for lk in description.get("links", [])
                       if lk.get("role") == "foot" and lk.get("leg_id") is not None]
        mean_foot_z = float(np.mean(foot_z_vals)) if foot_z_vals else -0.35
        feet_at_ground = abs(mean_foot_z + BODY_HEIGHT) < 0.12

        def _rtj(lo, hi, r):
            return float(lo + max(0.0, min(1.0, r)) * (hi - lo))

        stand = 0.5 * (lower + upper).astype(np.float32)
        for lid, j in triplets.items():
            if feet_at_ground:
                _lr = max(0.0, min(1.0, (0.0 - j["lift_lower"]) / max(j["lift_upper"] - j["lift_lower"], 1e-9)))
                _dr = max(0.0, min(1.0, (0.0 - j["drop_lower"]) / max(j["drop_upper"] - j["drop_lower"], 1e-9)))
            else:
                _lr, _dr = 0.05, 0.995
            stand[j["lift_idx"]]  = _rtj(j["lift_lower"],  j["lift_upper"],  _lr)
            stand[j["drop_idx"]]  = _rtj(j["drop_lower"],  j["drop_upper"],  _dr)
            stand[j["swing_idx"]] = _rtj(j["swing_lower"], j["swing_upper"], 0.5)
        stand = np.clip(stand, finite_lo, finite_hi)

        ds = gym.get_actor_dof_states(env, actor, _ga.STATE_ALL)
        ds["pos"] = stand; ds["vel"].fill(0.0)
        gym.set_actor_dof_states(env, actor, ds, _ga.STATE_ALL)

        # Cache the complete articulated pose before any warm-up can tip an
        # asymmetric robot.  Resetting only rigid body 0 leaves child links in
        # their previous fallen poses and contaminates every later probe.
        initial_rb_states = np.copy(
            gym.get_actor_rigid_body_states(env, actor, _ga.STATE_ALL)
        )

        if feet_at_ground and triplets:
            fj   = next(iter(triplets.values()))
            _lrr = max(fj["lift_upper"] - fj["lift_lower"], 1e-9)
            _drr = max(fj["drop_upper"] - fj["drop_lower"], 1e-9)
            _sl  = max(0.0, min(1.0, (0.0  - fj["lift_lower"]) / _lrr))
            _swl = max(0.0, min(1.0, (-0.25 - fj["lift_lower"]) / _lrr))  # foot UP during swing
            _sd  = max(0.0, min(1.0, (0.0  - fj["drop_lower"]) / _drr))
            _swd = _sd
        else:
            _sl, _swl, _sd, _swd = STANCE_LIFT_RATIO, SWING_LIFT_RATIO, STANCE_DROP_RATIO, SWING_DROP_RATIO

        # ── Sag-controller warm-up (run ONCE, result cached) ─────────────────
        stand_ev = stand.copy()
        for _ in range(HOLD_STEPS):
            ds_pos = gym.get_actor_dof_states(env, actor, _ga.STATE_POS)
            jpos = np.asarray(ds_pos["pos"], dtype=np.float32)
            se   = stand_ev.copy()
            for idx, name in enumerate(dof_names):
                sag = se[idx] - jpos[idx]
                if "_drop" in name and sag > 0.004:
                    se[idx] = min(finite_hi[idx], se[idx] + min(0.012, 0.22 * sag))
                elif "_lift" in name and sag > 0.004:
                    se[idx] = max(finite_lo[idx], se[idx] - min(0.008, 0.16 * sag))
            stand_ev = np.clip(se, finite_lo, finite_hi)
            gym.set_actor_dof_position_targets(env, actor, stand_ev)
            gym.simulate(sim); gym.fetch_results(sim, True)

        # ── Foot position map ─────────────────────────────────────────────────
        fmap: Dict[int, np.ndarray] = {}
        for lk in description.get("links", []):
            if lk.get("role") == "foot" and lk.get("leg_id") is not None:
                fmap[int(lk["leg_id"])] = np.asarray(lk["default_world_origin"], dtype=float)[:2]
        if fmap:
            foot_pts = np.array(list(fmap.values()), dtype=float)
            body_length = float(np.max(foot_pts.max(axis=0) - foot_pts.min(axis=0)))
        else:
            body_length = 0.5

        self.gym = gym; self.sim = sim; self.env = env; self.actor = actor
        self.description = description
        self.urdf_path = urdf_path
        self.dof_names = list(dof_names)
        self.triplets  = triplets
        self.lower = lower; self.upper = upper
        self.finite_lo = finite_lo; self.finite_hi = finite_hi
        self.stand = stand; self.stand_ev = stand_ev.copy()
        self.initial_rb_states = initial_rb_states
        self._sl = _sl; self._swl = _swl; self._sd = _sd; self._swd = _swd
        self.fmap = fmap; self.body_length = body_length
        self._closed = False

    # ------------------------------------------------------------------
    def _reset(self, stand_targets=None):
        """Teleport robot back to origin, zero velocities and level its stance."""
        ga = self._ga
        gym, sim, env, actor = self.gym, self.sim, self.env, self.actor

        # Reset DOF positions and velocities
        ds = gym.get_actor_dof_states(env, actor, ga.STATE_ALL)
        reset_stand = self.stand_ev if stand_targets is None else np.asarray(stand_targets, dtype=np.float32)
        ds["pos"] = reset_stand
        ds["vel"].fill(0.0)
        gym.set_actor_dof_states(env, actor, ds, ga.STATE_ALL)

        # Reset root body pose + velocity
        try:
            rb = np.copy(self.initial_rb_states)
            rb["pose"]["p"][0]["x"] = 0.0
            rb["pose"]["p"][0]["y"] = 0.0
            rb["pose"]["p"][0]["z"] = BODY_HEIGHT
            rb["pose"]["r"][0]["x"] = 0.0
            rb["pose"]["r"][0]["y"] = 0.0
            rb["pose"]["r"][0]["z"] = 0.0
            rb["pose"]["r"][0]["w"] = 1.0
            rb["vel"]["linear"]["x"].fill(0.0)
            rb["vel"]["linear"]["y"].fill(0.0)
            rb["vel"]["linear"]["z"].fill(0.0)
            rb["vel"]["angular"]["x"].fill(0.0)
            rb["vel"]["angular"]["y"].fill(0.0)
            rb["vel"]["angular"]["z"].fill(0.0)
            gym.set_actor_rigid_body_states(env, actor, rb, ga.STATE_ALL)
        except Exception:
            pass   # If root-state setter is unavailable, DOF reset is sufficient

        def _roll_pitch(w, x, y, z):
            sinr = 2.0 * (w * x + y * z)
            cosr = 1.0 - 2.0 * (x * x + y * y)
            roll = math.atan2(sinr, cosr)
            sinp = 2.0 * (w * y - z * x)
            pitch = math.asin(max(-1.0, min(1.0, sinp)))
            return roll, pitch

        x_extent = max((abs(float(v[0])) for v in self.fmap.values()), default=1.0)
        y_extent = max((abs(float(v[1])) for v in self.fmap.values()), default=1.0)
        level_bias = {lid: 0.0 for lid in self.triplets}

        # Marginal amputated support polygons need an active levelling phase.
        # A five-step reset was too short: the gait started while the chassis
        # was already rolling toward the missing side.
        for _ in range(90):
            targets = reset_stand.copy()
            for lid, j in self.triplets.items():
                b = level_bias.get(lid, 0.0)
                targets[j["lift_idx"]] += 0.25 * b * (j["lift_upper"] - j["lift_lower"])
                targets[j["drop_idx"]] += 0.65 * b * (j["drop_upper"] - j["drop_lower"])
            targets = np.clip(targets, self.finite_lo, self.finite_hi)
            gym.set_actor_dof_position_targets(env, actor, targets)
            gym.simulate(sim); gym.fetch_results(sim, True)
            rb = gym.get_actor_rigid_body_states(env, actor, ga.STATE_POS)
            if rb is not None and len(rb) > 0:
                q = rb["pose"]["r"][0]
                roll, pitch = _roll_pitch(float(q["w"]), float(q["x"]), float(q["y"]), float(q["z"]))
                for lid, foot in self.fmap.items():
                    tilt = roll * float(foot[1]) / y_extent - pitch * float(foot[0]) / x_extent
                    raw = float(np.clip(0.70 * tilt, -0.18, 0.18))
                    level_bias[lid] = 0.85 * level_bias.get(lid, 0.0) + 0.15 * raw
        return level_bias

    # ------------------------------------------------------------------
    def run_episode(self, plan: dict, n_steps: int,
                    return_yaw_stats: bool = False,
                    min_travel: float = 0.0,
                    min_steps: int = 0) -> tuple:
        """Reset robot and run one gait episode.

        Returns (com_trail, forward_axis, yaw_stats_or_None).
        min_travel / min_steps: early-exit when both conditions met (for full sims).
        """
        ga = self._ga
        gym, sim, env, actor = self.gym, self.sim, self.env, self.actor

        forward_axis = list(plan.get("final_forward_axis", [1.0, 0.0]))
        topo    = plan["topology"]
        group_a = topo["groups"]["group_a"]
        group_b = topo["groups"]["group_b"]
        group_c = topo["groups"].get("group_c", [])

        cpg = plan.get("cpg", {})
        gait_mode = str(cpg.get("mode", "legacy_sine"))
        gait_frequency = float(cpg.get("frequency_hz", GAIT_FREQUENCY))
        duty_factor = float(np.clip(cpg.get("duty_factor", 0.5), 0.50, 0.92))
        phase_offsets = {
            int(k): float(v) for k, v in cpg.get("phase_offsets", {}).items()
        }

        episode_stand = self.stand_ev.copy()
        stance_override = dict(plan.get("stance_joint_positions", {}))
        if plan.get("use_stance_ik", False) and not stance_override and len(self.triplets) <= 4:
            compensation = plan.get("translational_compensation_xy", [0.0, 0.0])
            if float(np.linalg.norm(np.asarray(compensation[:2], dtype=float))) > 0.06:
                try:
                    from adaptation.kinematics import compensated_stance_ik
                    stance_override = compensated_stance_ik(
                        self.urdf_path, compensation, BODY_HEIGHT
                    )
                except Exception as exc:
                    print(f"[StanceIK] failed: {exc}")
        name_to_index = {name: idx for idx, name in enumerate(self.dof_names)}
        for name, value in stance_override.items():
            if name in name_to_index:
                episode_stand[name_to_index[name]] = float(value)
        episode_stand = np.clip(episode_stand, self.finite_lo, self.finite_hi)
        level_bias = self._reset(episode_stand)

        fwd = np.asarray(forward_axis, dtype=float)
        fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
        lat = np.array([-fwd[1], fwd[0]], dtype=float)

        per_amp = {str(k): float(v) for k, v in topo.get("per_leg_stride_amplitudes", {}).items()}
        if "_per_amp_override" in plan:
            per_amp = {str(k): float(v) for k, v in plan["_per_amp_override"].items()}

        def _rtj(lo, hi, r):
            return float(lo + max(0.0, min(1.0, r)) * (hi - lo))
        # Quintic smoothstep — C² continuous (acceleration vanishes at boundaries)
        def _ss(e0, e1, x):
            t = max(0.0, min(1.0, (x - e0) / max(e1 - e0, 1e-9)))
            return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)
        def _quat_to_yaw(w, x, y, z):
            """Extract yaw from quaternion."""
            siny = 2.0 * (w * z + x * y)
            cosy = 1.0 - 2.0 * (y * y + z * z)
            return float(np.arctan2(siny, cosy))
        def _quat_to_roll_pitch(w, x, y, z):
            sinr = 2.0 * (w * x + y * z)
            cosr = 1.0 - 2.0 * (x * x + y * y)
            roll = math.atan2(sinr, cosr)
            sinp = 2.0 * (w * y - z * x)
            return roll, math.asin(max(-1.0, min(1.0, sinp)))
        def _gait_wave(phase_rad, duty):
            """Return (fore/aft wave, lift alpha) with C2-continuous joins.

            During stance the foot moves slowly from front (+1) to rear (-1).
            During swing it returns quickly while following a bell lift profile.
            """
            q = (phase_rad / (2.0 * math.pi)) % 1.0
            if q < duty:
                s = q / max(duty, 1e-9)
                return 1.0 - 2.0 * _ss(0.0, 1.0, s), 0.0
            s = (q - duty) / max(1.0 - duty, 1e-9)
            return -1.0 + 2.0 * _ss(0.0, 1.0, s), math.sin(math.pi * s) ** 2

        _sl, _swl, _sd, _swd = self._sl, self._swl, self._sd, self._swd
        com_trail:      List[List[float]] = []
        yaw_acc:        List[float]       = []
        touchdown_ramp: Dict[int, int]    = {}
        sim_time = 0.0
        dt = 1.0 / 60.0

        # ── Online yaw correction state ────────────────────────────────────
        _base_amplitudes = dict(per_amp)
        _yaw_integral = 0.0
        # A robot can translate along a morphology-selected axis without
        # rotating its chassis to that axis (e.g. a damaged robot walking
        # sideways).  Keep the initial body orientation unless explicitly set.
        _planned_yaw = float(plan.get("body_yaw_target", 0.0))
        # ── Body height compensation state ─────────────────────────────────
        _height_target = BODY_HEIGHT
        _height_ie = 0.0
        x_extent = max((abs(float(v[0])) for v in self.fmap.values()), default=1.0)
        y_extent = max((abs(float(v[1])) for v in self.fmap.values()), default=1.0)

        for step in range(n_steps):
            phase_now = 2.0 * math.pi * gait_frequency * sim_time
            targets   = episode_stand.copy()
            stance_weights: Dict[int, float] = {}

            for lid, j in self.triplets.items():
                use_stance_override = bool(stance_override)
                base_lr = ((episode_stand[j["lift_idx"]] - j["lift_lower"])
                           / max(j["lift_upper"] - j["lift_lower"], 1e-9))
                base_dr = ((episode_stand[j["drop_idx"]] - j["drop_lower"])
                           / max(j["drop_upper"] - j["drop_lower"], 1e-9))
                base_sr = ((episode_stand[j["swing_idx"]] - j["swing_lower"])
                           / max(j["swing_upper"] - j["swing_lower"], 1e-9))
                if lid in group_c:
                    targets[j["lift_idx"]] = _rtj(j["lift_lower"], j["lift_upper"], base_lr if use_stance_override else _sl)
                    targets[j["drop_idx"]] = _rtj(j["drop_lower"], j["drop_upper"], base_dr if use_stance_override else _sd)
                    targets[j["swing_idx"]] = _rtj(j["swing_lower"], j["swing_upper"], base_sr if use_stance_override else 0.5)
                    continue

                if lid in phase_offsets:
                    lg_ph = phase_now + phase_offsets[lid]
                elif lid in group_b:
                    lg_ph = phase_now + math.pi
                elif lid in group_a:
                    lg_ph = phase_now
                else:
                    lg_ph = 0.0

                if gait_mode in ("tripod", "alternating", "wave"):
                    sw, alpha = _gait_wave(lg_ph, duty_factor)
                else:
                    sw = float(math.sin(lg_ph))
                    alpha = _ss(-0.35, 0.35, sw)
                stance_weights[lid] = 1.0 - alpha
                fv    = self.fmap.get(lid, np.zeros(2))
                dsign = 1.0 if float(np.dot(fv, lat)) > 0.0 else -1.0
                eff_amp = SWING_AMP * float(per_amp.get(str(lid), 1.0))

                if use_stance_override:
                    lr = base_lr + (_swl - self._sl) * alpha + (_sl - self._sl)
                    dr = base_dr + (_swd - self._sd) * alpha + (_sd - self._sd)
                    sr = base_sr + eff_amp * dsign * sw
                else:
                    lr = _sl + (_swl - _sl) * alpha
                    dr = _sd + (_swd - _sd) * alpha
                    sr = 0.5 + eff_amp * dsign * sw

                is_sw = alpha > 1e-6
                if not is_sw:
                    if lid in touchdown_ramp:
                        touchdown_ramp[lid] += 1
                        rp_raw = min(touchdown_ramp[lid] / 25, 1.0)
                        if rp_raw >= 1.0:
                            del touchdown_ramp[lid]
                        else:
                            rp = _ss(0.0, 1.0, rp_raw)  # quintic easing
                            dl  = _rtj(j["lift_lower"],  j["lift_upper"],  0.5)
                            dd  = _rtj(j["drop_lower"],  j["drop_upper"],  0.5)
                            ds_ = j["swing_lower"] + 0.5 * (j["swing_upper"] - j["swing_lower"])
                            targets[j["lift_idx"]]  = dl  + rp * (_rtj(j["lift_lower"],  j["lift_upper"],  lr) - dl)
                            targets[j["drop_idx"]]  = dd  + rp * (_rtj(j["drop_lower"],  j["drop_upper"],  dr) - dd)
                            targets[j["swing_idx"]] = ds_ + rp * (_rtj(j["swing_lower"], j["swing_upper"], sr) - ds_)
                            continue
                else:
                    touchdown_ramp[lid] = 0

                targets[j["lift_idx"]]  = _rtj(j["lift_lower"], j["lift_upper"], lr)
                targets[j["drop_idx"]]  = _rtj(j["drop_lower"], j["drop_upper"], dr)
                targets[j["swing_idx"]] = _rtj(j["swing_lower"], j["swing_upper"], sr)

            # Per-leg roll/pitch levelling.  Apply mainly to stance legs so it
            # does not erase swing clearance.
            for lid, j in self.triplets.items():
                b = level_bias.get(lid, 0.0) * stance_weights.get(lid, 1.0)
                targets[j["lift_idx"]] += 0.25 * b * (j["lift_upper"] - j["lift_lower"])
                targets[j["drop_idx"]] += 0.65 * b * (j["drop_upper"] - j["drop_lower"])
            targets = np.clip(targets, self.finite_lo, self.finite_hi)

            gym.set_actor_dof_position_targets(env, actor, targets)
            gym.simulate(sim); gym.fetch_results(sim, True)

            _sflag = ga.STATE_ALL if return_yaw_stats else ga.STATE_POS
            states = gym.get_actor_rigid_body_states(env, actor, ga.STATE_ALL)
            body_yaw = 0.0
            if states is not None and len(states) > 0:
                p = states["pose"]["p"][0]
                r = states["pose"]["r"][0]
                body_yaw = _quat_to_yaw(float(r["w"]), float(r["x"]), float(r["y"]), float(r["z"]))
                body_roll, body_pitch = _quat_to_roll_pitch(
                    float(r["w"]), float(r["x"]), float(r["y"]), float(r["z"])
                )
                for lid, foot in self.fmap.items():
                    tilt = (body_roll * float(foot[1]) / y_extent
                            - body_pitch * float(foot[0]) / x_extent)
                    raw = float(np.clip(0.70 * tilt, -0.18, 0.18))
                    level_bias[lid] = 0.90 * level_bias.get(lid, 0.0) + 0.10 * raw
                # Keep yaw as column 2 for backwards compatibility and append
                # height as column 3 for steady-locomotion validation.
                com_trail.append([float(p["x"]), float(p["y"]), body_yaw, float(p["z"])])
                if return_yaw_stats:
                    try:
                        yaw_acc.append(float(states["vel"]["angular"][0]["z"]))
                    except Exception:
                        if len(com_trail) >= 3:
                            _d1 = np.array(com_trail[-1][:2]) - np.array(com_trail[-2][:2])
                            _d0 = np.array(com_trail[-2][:2]) - np.array(com_trail[-3][:2])
                            _da = ((math.atan2(_d1[1], _d1[0]) - math.atan2(_d0[1], _d0[0]) + math.pi) % (2 * math.pi)) - math.pi
                            yaw_acc.append(_da / max(dt, 1e-9))

            # ── Online yaw correction (every 60 steps) ─────────────────────
            if (step + 1) % 60 == 0:
                yaw_err = body_yaw - _planned_yaw
                yaw_err = float(np.arctan2(np.sin(yaw_err), np.cos(yaw_err)))
                # Track the planned line as well as the planned heading.  Pure
                # yaw regulation cannot remove a steady lateral translation
                # caused by an amputated/asymmetric contact pattern.
                lateral_error = 0.0
                lateral_velocity = 0.0
                if len(com_trail) >= 2:
                    p0 = np.asarray(com_trail[0][:2], dtype=float)
                    pn = np.asarray(com_trail[-1][:2], dtype=float)
                    lateral_error = float(np.dot(pn - p0, lat))
                    if len(com_trail) >= 61:
                        p_prev = np.asarray(com_trail[-61][:2], dtype=float)
                        lateral_velocity = float(np.dot(pn - p_prev, lat))
                desired_yaw_offset = float(np.clip(
                    -2.0 * lateral_error - 0.8 * lateral_velocity,
                    -0.40, 0.40,
                ))
                control_err = yaw_err - desired_yaw_offset
                if abs(control_err) > 0.0087:  # ~0.5° or equivalent cross-track error
                    _yaw_integral += 0.02 * control_err
                    _yaw_integral = float(np.clip(_yaw_integral, -0.5, 0.5))
                    # PI: Kp=1.0, Ki=0.3, saturation at 0.3 rad (~17°)
                    yaw_p = float(np.clip(control_err / 0.3, -1.0, 1.0))
                    yaw_i = float(np.clip(_yaw_integral / 0.3, -1.0, 1.0))
                    yaw_signal = float(np.clip(yaw_p + 0.3 * yaw_i, -1.0, 1.0))
                    # Apply via yaw_lever model
                    yaw_levers = {int(k): float(v)
                                  for k, v in plan.get("yaw_balance", {}).get("yaw_levers", {}).items()}
                    if yaw_levers:
                        max_lv = max(abs(v) for v in yaw_levers.values())
                        if max_lv > 1e-9:
                            corr = {}
                            for lid_str, amp in _base_amplitudes.items():
                                lv = yaw_levers.get(int(lid_str), 0.0)
                                s = 1.0 - yaw_signal * lv / max_lv
                                s = float(np.clip(s, 0.40, 1.60))
                                corr[lid_str] = float(np.clip(float(amp) * s, 0.15, 0.85))
                            per_amp = corr

            # ── Body height compensation (every 120 steps) ────────────────
            if (step + 1) % 120 == 0 and states is not None and len(states) > 0:
                body_z = float(states["pose"]["p"][0]["z"])
                z_err = _height_target - body_z
                _height_ie += 0.03 * z_err
                _height_ie = float(np.clip(_height_ie, -0.15, 0.35))
                _sl  = self._sl  - 0.25 * _height_ie
                _sl  = float(np.clip(_sl,  0.05, 0.95))
                _sd  = self._sd  - 0.40 * _height_ie
                _sd  = float(np.clip(_sd,  0.55, 1.0))

            sim_time += dt

            # Early exit for full-sim mode
            if min_travel > 0.0 and step >= min_steps and len(com_trail) > 1:
                arr = np.array(com_trail, dtype=float)
                covered = abs(float(np.dot(arr[-1][:2] - arr[0][:2], fwd)))
                if covered >= min_travel:
                    break

        yaw_stats = None
        if return_yaw_stats:
            valid  = [v for v in yaw_acc if abs(v) < 20.0]
            mean_y = float(np.mean(valid)) if valid else 0.0
            yaw_stats = {"yaw_rate_mean": mean_y,
                         "yaw_rate_std":  float(np.std(valid)) if valid else 0.0}
        return com_trail, forward_axis, yaw_stats

    # ------------------------------------------------------------------
    def close(self):
        if not self._closed:
            self.gym.destroy_sim(self.sim)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
