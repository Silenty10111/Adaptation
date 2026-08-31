#!/usr/bin/env python3
"""Isaac Gym simulation infrastructure — _RobotSimCtx + gait constants.

Extracted from batch_test.py for reuse across experiment scripts.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from adaptation.phase import (
    ContactPhaseGateConfig,
    TWO_PI,
    contact_aware_swing_set,
    leg_phase_state,
    phase_state,
    resolve_duty_factors,
    resolve_phase_offsets,
    wrap_2pi,
)
from adaptation.diagnostics import (
    EpisodeDiagnosticAccumulator,
    EpisodeStepTelemetry,
    contact_support_ssm,
)

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
        foot_z_vals = [
            float(link["default_world_origin"][2])
            for link in description.get("links", [])
            if link.get("role") == "foot" and link.get("leg_id") is not None
        ]
        # Spawn close to the morphology's nominal standing height.  A fixed
        # 0.50 m release made short/irregular robots fall several centimetres
        # onto only their longest legs before every episode.
        nominal_body_height = (
            float(np.clip(-np.percentile(foot_z_vals, 75.0) + 0.005, 0.20, BODY_HEIGHT))
            if foot_z_vals else BODY_HEIGHT
        )
        pose  = _ga.Transform()
        pose.p = _ga.Vec3(0.0, 0.0, nominal_body_height)
        actor = gym.create_actor(env, asset, pose, "r", 0, 1)
        dof_force_sensors_enabled = False
        try:
            gym.enable_actor_dof_force_sensors(env, actor)
            dof_force_sensors_enabled = True
        except Exception:
            pass

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
        velocity_limits = (
            np.asarray(dof_props["velocity"], dtype=float)
            if "velocity" in dof_props.dtype.names else np.full(len(lower), np.nan)
        )
        effort_limits = (
            np.asarray(dof_props["effort"], dtype=float)
            if "effort" in dof_props.dtype.names else np.full(len(lower), np.nan)
        )

        name2idx = {n: i for i, n in enumerate(dof_names)}
        rigid_body_names = list(gym.get_actor_rigid_body_names(env, actor))
        terminal_body_by_leg: Dict[int, int] = {}
        body_leg_ids: Dict[int, int] = {}
        for body_index, body_name in enumerate(rigid_body_names):
            match = re.match(r"leg_(\d+)_", str(body_name))
            if match:
                body_leg_ids[body_index] = int(match.group(1))
            if match and str(body_name).endswith(("_foot", "_lower", "_knee")):
                terminal_body_by_leg[int(match.group(1))] = body_index
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

        mean_foot_z = float(np.mean(foot_z_vals)) if foot_z_vals else -0.35
        feet_at_ground = abs(mean_foot_z + nominal_body_height) < 0.12

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

        # Joint-limit ratios do not have a universal vertical direction.
        # Standard legs and some generated legs use opposite revolute axes.
        # Compute the sign of a small stance-extension command from the
        # zero-pose geometric Jacobian: dz/dq = (axis x (foot-joint))_z.
        link_origins = {
            str(link.get("name")): np.asarray(
                link.get("default_world_origin", [0.0, 0.0, 0.0]),
                dtype=float,
            )
            for link in description.get("links", [])
        }
        joints_by_name = {
            str(joint.get("name")): joint
            for joint in description.get("joints", [])
        }
        stance_extension_sign: Dict[int, float] = {}
        for lid, j in triplets.items():
            foot = link_origins.get(f"leg_{lid}_foot")
            vertical_sensitivity = 0.0
            if foot is not None:
                for kind, weight, joint_range in (
                    ("lift", 1.0, j["lift_upper"] - j["lift_lower"]),
                    ("drop", 0.70, j["drop_upper"] - j["drop_lower"]),
                ):
                    joint = joints_by_name.get(f"leg_{lid}_{kind}", {})
                    origin = link_origins.get(str(joint.get("parent")))
                    axis = np.asarray(joint.get("axis", [0.0, 0.0, 0.0]), dtype=float)
                    if origin is not None and len(axis) >= 3:
                        vertical_sensitivity += (
                            weight * float(joint_range)
                            * float(np.cross(axis[:3], foot[:3] - origin[:3])[2])
                        )
            stance_extension_sign[lid] = (
                -1.0 if vertical_sensitivity > 1e-9 else 1.0
            )

        self.gym = gym; self.sim = sim; self.env = env; self.actor = actor
        self.description = description
        self.urdf_path = urdf_path
        self.dof_names = list(dof_names)
        self.triplets  = triplets
        self.lower = lower; self.upper = upper
        self.finite_lo = finite_lo; self.finite_hi = finite_hi
        self.velocity_limits = velocity_limits
        self.effort_limits = effort_limits
        self.dof_force_sensors_enabled = dof_force_sensors_enabled
        self.rigid_body_names = rigid_body_names
        self.terminal_body_by_leg = terminal_body_by_leg
        self.body_leg_ids = body_leg_ids
        self.stand = stand; self.stand_ev = stand_ev.copy()
        self.initial_rb_states = initial_rb_states
        self._sl = _sl; self._swl = _swl; self._sd = _sd; self._swd = _swd
        self.fmap = fmap; self.body_length = body_length
        self.stance_extension_sign = stance_extension_sign
        self.body_height = nominal_body_height
        self._closed = False

    # ------------------------------------------------------------------
    def _reset(
        self,
        stand_targets=None,
        contact_calibration: bool = True,
        maximum_stance_extension: float = 0.06,
        calibration_step: float = 0.0015,
    ):
        """Teleport, level, and calibrate per-leg ground reach.

        Nominal foot coordinates can be coplanar while motor sag and unequal
        link leverage still leave several feet a few millimetres airborne.
        During the reset hold, extend only persistently non-contacting legs by
        a small bounded joint-range ratio.  The resulting offsets are carried
        into stance execution; this is contact calibration, not a global step
        amplitude reduction.
        """
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
            rb["pose"]["p"][0]["z"] = self.body_height
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
        stance_extension = {lid: 0.0 for lid in self.triplets}
        missing_streak = {lid: 0 for lid in self.triplets}
        calibrated_contacts = set()
        terminal_bodies = set(self.terminal_body_by_leg.values())

        def _terminal_ground_contacts():
            result = set()
            for contact in gym.get_env_rigid_contacts(env):
                if float(contact["lambda"]) <= 1e-6:
                    continue
                body0 = int(contact["body0"])
                body1 = int(contact["body1"])
                if body1 == -1 and body0 in terminal_bodies:
                    result.add(int(self.body_leg_ids[body0]))
                elif body0 == -1 and body1 in terminal_bodies:
                    result.add(int(self.body_leg_ids[body1]))
            return result

        # Marginal amputated support polygons need an active levelling phase.
        # A five-step reset was too short: the gait started while the chassis
        # was already rolling toward the missing side.
        for reset_step in range(90):
            targets = reset_stand.copy()
            for lid, j in self.triplets.items():
                b = level_bias.get(lid, 0.0)
                targets[j["lift_idx"]] += 0.25 * b * (j["lift_upper"] - j["lift_lower"])
                targets[j["drop_idx"]] += 0.65 * b * (j["drop_upper"] - j["drop_lower"])
                extension = stance_extension.get(lid, 0.0)
                signed_extension = self.stance_extension_sign.get(lid, 1.0) * extension
                targets[j["lift_idx"]] += signed_extension * (
                    j["lift_upper"] - j["lift_lower"]
                )
                targets[j["drop_idx"]] += 0.70 * signed_extension * (
                    j["drop_upper"] - j["drop_lower"]
                )
            targets = np.clip(targets, self.finite_lo, self.finite_hi)
            gym.set_actor_dof_position_targets(env, actor, targets)
            gym.simulate(sim); gym.fetch_results(sim, True)
            if contact_calibration and reset_step >= 15:
                try:
                    calibrated_contacts = _terminal_ground_contacts()
                    for lid in self.triplets:
                        if lid in calibrated_contacts:
                            missing_streak[lid] = 0
                        else:
                            missing_streak[lid] += 1
                            if missing_streak[lid] >= 3:
                                stance_extension[lid] = min(
                                    float(maximum_stance_extension),
                                    stance_extension[lid] + float(calibration_step),
                                )
                except Exception:
                    contact_calibration = False
            rb = gym.get_actor_rigid_body_states(env, actor, ga.STATE_POS)
            if rb is not None and len(rb) > 0:
                q = rb["pose"]["r"][0]
                roll, pitch = _roll_pitch(float(q["w"]), float(q["x"]), float(q["y"]), float(q["z"]))
                for lid, foot in self.fmap.items():
                    tilt = roll * float(foot[1]) / y_extent - pitch * float(foot[0]) / x_extent
                    raw = float(np.clip(0.70 * tilt, -0.18, 0.18))
                    level_bias[lid] = 0.85 * level_bias.get(lid, 0.0) + 0.15 * raw
        return level_bias, stance_extension, {
            "enabled": bool(contact_calibration),
            "contact_leg_ids_at_release": sorted(calibrated_contacts),
            "stance_extension_ratio_by_leg": {
                str(lid): float(self.stance_extension_sign.get(lid, 1.0) * value)
                for lid, value in stance_extension.items()
            },
            "maximum_extension_ratio": float(max(
                stance_extension.values(), default=0.0,
            )),
        }

    # ------------------------------------------------------------------
    def run_episode(self, plan: dict, n_steps: int,
                    return_yaw_stats: bool = False,
                    return_diagnostics: bool = False,
                    diagnostic_stride: int = 1,
                    min_travel: float = 0.0,
                    min_steps: int = 0) -> tuple:
        """Reset robot and run one gait episode.

        Returns ``(trail, axis, yaw_stats)`` and appends a diagnostic mapping
        when ``return_diagnostics=True``.
        min_travel / min_steps: early-exit when both conditions met (for full sims).
        """
        ga = self._ga
        gym, sim, env, actor = self.gym, self.sim, self.env, self.actor

        forward_axis = list(plan.get("final_forward_axis", [1.0, 0.0]))
        # Keep motor-map side assignment tied to the axis on which the gait
        # was planned.  A probe may calibrate a different realised course for
        # validation/cross-track control without silently rebuilding the gait.
        actuation_forward_axis = list(plan.get(
            "actuation_forward_axis", forward_axis,
        ))
        topo    = plan["topology"]
        group_a = topo["groups"]["group_a"]
        group_b = topo["groups"]["group_b"]
        group_c = topo["groups"].get("group_c", [])

        cpg = plan.get("cpg", {})
        gait_frequency = float(cpg.get("frequency_hz", GAIT_FREQUENCY))
        default_stride_direction = -1.0 if len(self.triplets) >= 7 else 1.0
        stride_direction = (
            1.0 if float(cpg.get(
                "stride_direction", default_stride_direction,
            )) >= 0.0 else -1.0
        )
        swing_sign_mode = str(cpg.get(
            "swing_sign_mode",
            "kinematic_jacobian" if len(self.triplets) >= 7 else "legacy_side",
        ))
        if swing_sign_mode not in {"legacy_side", "kinematic_jacobian"}:
            swing_sign_mode = "legacy_side"
        touchdown_settle_steps = max(int(cpg.get(
            "touchdown_settle_steps", 0 if len(self.triplets) >= 7 else 25,
        )), 0)
        touchdown_vertical_reset = float(np.clip(
            cpg.get("touchdown_vertical_reset", 0.35), 0.0, 1.0,
        ))
        phase_offsets = resolve_phase_offsets(plan, self.triplets)
        duty_factors, duty_diagnostics = resolve_duty_factors(plan, self.triplets)
        for message in duty_diagnostics:
            print(f"[Phase] {message}")
        raw_contact_feedback = cpg.get("contact_feedback", {})
        if not isinstance(raw_contact_feedback, dict):
            raw_contact_feedback = {}
        contact_feedback_enabled = bool(
            raw_contact_feedback.get("enabled", False)
        )
        phase_gate_enabled = (
            contact_feedback_enabled
            and bool(raw_contact_feedback.get("gate_liftoff", False))
        )
        contact_gate_config = ContactPhaseGateConfig(
            minimum_support_count=int(raw_contact_feedback.get(
                "minimum_support_count", 3,
            )),
            maximum_simultaneous_swing=int(raw_contact_feedback.get(
                "maximum_simultaneous_swing", 3,
            )),
            swing_start_window=float(raw_contact_feedback.get(
                "swing_start_window", 0.35,
            )),
            early_touchdown_progress=float(raw_contact_feedback.get(
                "early_touchdown_progress", 0.35,
            )),
        )
        stance_search_ratio = max(float(raw_contact_feedback.get(
            "stance_search_ratio", 0.07,
        )), 0.0)
        stance_search_steps = max(int(raw_contact_feedback.get(
            "stance_search_steps", 18,
        )), 1)
        latch_stance_search = bool(raw_contact_feedback.get(
            "latch_stance_search", False,
        ))
        stance_search_release_steps = max(int(raw_contact_feedback.get(
            "stance_search_release_steps", 144,
        )), 1)
        emergency_contact_recovery = bool(raw_contact_feedback.get(
            "emergency_contact_recovery", False,
        ))
        emergency_support_count = max(int(raw_contact_feedback.get(
            "emergency_support_count", 3,
        )), 0)

        episode_stand = self.stand_ev.copy()
        stance_override = dict(plan.get("stance_joint_positions", {}))
        if plan.get("use_stance_ik", False) and not stance_override and len(self.triplets) <= 4:
            compensation = plan.get("translational_compensation_xy", [0.0, 0.0])
            if float(np.linalg.norm(np.asarray(compensation[:2], dtype=float))) > 0.06:
                try:
                    from adaptation.kinematics import compensated_stance_ik
                    stance_override = compensated_stance_ik(
                        self.urdf_path, compensation, self.body_height
                    )
                except Exception as exc:
                    print(f"[StanceIK] failed: {exc}")
        name_to_index = {name: idx for idx, name in enumerate(self.dof_names)}
        for name, value in stance_override.items():
            if name in name_to_index:
                episode_stand[name_to_index[name]] = float(value)
        episode_stand = np.clip(episode_stand, self.finite_lo, self.finite_hi)
        reset_contact_calibration = bool(cpg.get(
            "reset_contact_calibration", len(self.triplets) >= 7,
        ))
        reset_maximum_extension = float(np.clip(cpg.get(
            "reset_maximum_stance_extension", 0.06,
        ), 0.0, 0.15))
        level_bias, stance_extension, reset_calibration = self._reset(
            episode_stand,
            contact_calibration=reset_contact_calibration,
            maximum_stance_extension=reset_maximum_extension,
        )

        fwd = np.asarray(forward_axis, dtype=float)
        fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
        lat = np.array([-fwd[1], fwd[0]], dtype=float)
        actuation_fwd = np.asarray(actuation_forward_axis, dtype=float)
        actuation_fwd = actuation_fwd / max(
            float(np.linalg.norm(actuation_fwd)), 1e-9,
        )
        actuation_lat = np.array(
            [-actuation_fwd[1], actuation_fwd[0]], dtype=float,
        )
        kinematic_swing_sign: Dict[int, float] = {}
        if swing_sign_mode == "kinematic_jacobian":
            link_origins = {
                str(link.get("name")): np.asarray(
                    link.get("default_world_origin", [0.0, 0.0, 0.0]),
                    dtype=float,
                )
                for link in self.description.get("links", [])
            }
            joints_by_name = {
                str(joint.get("name")): joint
                for joint in self.description.get("joints", [])
            }
            for lid in self.triplets:
                foot = link_origins.get(f"leg_{lid}_foot")
                joint = joints_by_name.get(f"leg_{lid}_swing", {})
                origin = link_origins.get(str(joint.get("parent")))
                axis = np.asarray(
                    joint.get("axis", [0.0, 0.0, 0.0]), dtype=float,
                )
                sensitivity = 0.0
                if foot is not None and origin is not None and len(axis) >= 3:
                    tangent = np.cross(axis[:3], foot[:3] - origin[:3])[:2]
                    sensitivity = float(np.dot(tangent, actuation_fwd))
                # Fall back only for a truly singular forward projection.
                legacy = (
                    1.0 if float(np.dot(self.fmap.get(lid, np.zeros(2)), actuation_lat)) > 0.0
                    else -1.0
                )
                kinematic_swing_sign[lid] = (
                    1.0 if sensitivity > 1e-8
                    else -1.0 if sensitivity < -1e-8
                    else legacy
                )

        per_amp = {str(k): float(v) for k, v in topo.get("per_leg_stride_amplitudes", {}).items()}
        if "_per_amp_override" in plan:
            per_amp = {str(k): float(v) for k, v in plan["_per_amp_override"].items()}

        def _rtj(lo, hi, r):
            return float(lo + max(0.0, min(1.0, r)) * (hi - lo))
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
        def _ground_contact_legs():
            contacts = gym.get_env_rigid_contacts(env)
            result = set()
            terminal_bodies = set(self.terminal_body_by_leg.values())
            for contact in contacts:
                if float(contact["lambda"]) <= 1e-6:
                    continue
                body0 = int(contact["body0"])
                body1 = int(contact["body1"])
                if body1 == -1 and body0 in terminal_bodies:
                    result.add(int(self.body_leg_ids[body0]))
                elif body0 == -1 and body1 in terminal_bodies:
                    result.add(int(self.body_leg_ids[body1]))
            return result
        _sl, _swl, _sd, _swd = self._sl, self._swl, self._sd, self._swd
        com_trail:      List[List[float]] = []
        yaw_acc:        List[float]       = []
        touchdown_ramp: Dict[int, int] = {}
        local_phases = {
            int(lid): wrap_2pi(float(phase_offsets.get(int(lid), 0.0)))
            for lid in self.triplets if lid not in group_c
        }
        previous_swing: set = set()
        swing_airborne: set = set()
        no_contact_stance_steps = {int(lid): 0 for lid in self.triplets}
        stance_search_offsets = {int(lid): 0.0 for lid in self.triplets}
        sim_time = 0.0
        dt = 1.0 / 60.0
        support_rescue_steps = 0

        # ── Online yaw correction state ────────────────────────────────────
        _base_amplitudes = dict(per_amp)
        _yaw_integral = 0.0
        # A robot can translate along a morphology-selected axis without
        # rotating its chassis to that axis (e.g. a damaged robot walking
        # sideways).  Keep the initial body orientation unless explicitly set.
        _planned_yaw = float(plan.get("body_yaw_target", 0.0))
        diagnostic_accumulator = None
        if return_diagnostics:
            diagnostic_accumulator = EpisodeDiagnosticAccumulator(
                forward_axis=fwd,
                active_leg_count=len([
                    leg_id for leg_id in self.triplets if leg_id not in group_c
                ]),
            )
            if not self.dof_force_sensors_enabled:
                diagnostic_accumulator.mark_unavailable(
                    "peak_joint_torque_ratio",
                    "Isaac Gym DOF force sensors could not be enabled",
                )
        # ── Body height compensation state ─────────────────────────────────
        _height_target = self.body_height
        _height_ie = 0.0
        x_extent = max((abs(float(v[0])) for v in self.fmap.values()), default=1.0)
        y_extent = max((abs(float(v[1])) for v in self.fmap.values()), default=1.0)

        for step in range(n_steps):
            phase_now = 2.0 * math.pi * gait_frequency * sim_time
            targets   = episode_stand.copy()
            stance_weights: Dict[int, float] = {}
            commanded_stance = {
                int(leg_id) for leg_id in group_c if int(leg_id) in self.triplets
            }
            feedback_contacts = None
            if contact_feedback_enabled:
                try:
                    feedback_contacts = _ground_contact_legs()
                except Exception:
                    feedback_contacts = None
            if phase_gate_enabled:
                command_states = {
                    int(lid): phase_state(local_phases[int(lid)], duty_factors[int(lid)])
                    for lid in local_phases
                }
                admitted_swing = contact_aware_swing_set(
                    command_states,
                    feedback_contacts,
                    previous_swing,
                    contact_gate_config,
                )
                # A rejected lift-off waits at the rear stance boundary.  It
                # retries on the next frame instead of skipping the swing and
                # dragging the foot rapidly from rear to front on the ground.
                for lid, state in list(command_states.items()):
                    if not state.is_stance and lid not in admitted_swing:
                        local_phases[lid] = max(
                            0.0, TWO_PI * duty_factors[lid] - 1e-6,
                        )
                        command_states[lid] = phase_state(
                            local_phases[lid], duty_factors[lid],
                        )
            else:
                command_states = {
                    int(lid): leg_phase_state(
                        phase_now, int(lid), phase_offsets, duty_factors,
                    )
                    for lid in self.triplets if lid not in group_c
                }
                admitted_swing = {
                    lid for lid, state in command_states.items() if not state.is_stance
                }
            support_rescue_active = bool(
                emergency_contact_recovery
                and feedback_contacts is not None
                and len(feedback_contacts) < emergency_support_count
            )
            if support_rescue_active:
                # This is a rare safety override, not a replacement timing
                # oscillator.  Put every leg down for this frame while the
                # global/local phases continue, then release automatically as
                # soon as measured support recovers.
                admitted_swing = set()
                support_rescue_steps += 1

            for lid, j in self.triplets.items():
                use_stance_override = bool(stance_override)
                base_lr = ((episode_stand[j["lift_idx"]] - j["lift_lower"])
                           / max(j["lift_upper"] - j["lift_lower"], 1e-9))
                base_dr = ((episode_stand[j["drop_idx"]] - j["drop_lower"])
                           / max(j["drop_upper"] - j["drop_lower"], 1e-9))
                base_sr = ((episode_stand[j["swing_idx"]] - j["swing_lower"])
                           / max(j["swing_upper"] - j["swing_lower"], 1e-9))
                if lid in group_c:
                    extension = (
                        self.stance_extension_sign.get(lid, 1.0)
                        * stance_extension.get(lid, 0.0)
                    )
                    targets[j["lift_idx"]] = _rtj(j["lift_lower"], j["lift_upper"], (base_lr if use_stance_override else _sl) + extension)
                    targets[j["drop_idx"]] = _rtj(j["drop_lower"], j["drop_upper"], (base_dr if use_stance_override else _sd) + 0.70 * extension)
                    targets[j["swing_idx"]] = _rtj(j["swing_lower"], j["swing_upper"], base_sr if use_stance_override else 0.5)
                    continue

                leg_state = command_states[lid]
                is_sw = lid in admitted_swing
                if not is_sw:
                    commanded_stance.add(int(lid))
                sw, alpha = stride_direction * leg_state.fore_aft, leg_state.lift
                if support_rescue_active:
                    alpha = 0.0
                if (
                    contact_feedback_enabled
                    and is_sw
                    and leg_state.swing_progress > 0.90
                    and feedback_contacts is not None
                    and lid not in feedback_contacts
                ):
                    # Touchdown search: preserve the front placement while
                    # removing clearance and extending the leg toward ground.
                    alpha = 0.0
                stance_weights[lid] = 1.0 - alpha
                fv    = self.fmap.get(lid, np.zeros(2))
                dsign = (
                    kinematic_swing_sign.get(lid, 1.0)
                    if swing_sign_mode == "kinematic_jacobian"
                    else (1.0 if float(np.dot(fv, actuation_lat)) > 0.0 else -1.0)
                )
                eff_amp = SWING_AMP * float(per_amp.get(str(lid), 1.0))

                if use_stance_override:
                    lr = base_lr + (_swl - self._sl) * alpha
                    dr = base_dr + (_swd - self._sd) * alpha
                    sr = base_sr + eff_amp * dsign * sw
                else:
                    lr = _sl + (_swl - _sl) * alpha
                    dr = _sd + (_swd - _sd) * alpha
                    sr = 0.5 + eff_amp * dsign * sw

                if not is_sw:
                    extension = (
                        self.stance_extension_sign.get(lid, 1.0)
                        * stance_extension.get(lid, 0.0)
                    )
                    lr += extension
                    dr += 0.70 * extension

                if not is_sw:
                    if touchdown_settle_steps > 0 and lid in touchdown_ramp:
                        touchdown_ramp[lid] += 1
                        raw = min(
                            touchdown_ramp[lid] / touchdown_settle_steps, 1.0,
                        )
                        if raw >= 1.0:
                            del touchdown_ramp[lid]
                        else:
                            blend = raw * raw * raw * (10.0 - 15.0 * raw + 6.0 * raw * raw)
                            lift_target = _rtj(j["lift_lower"], j["lift_upper"], lr)
                            drop_target = _rtj(j["drop_lower"], j["drop_upper"], dr)
                            lift_mid = _rtj(j["lift_lower"], j["lift_upper"], 0.5)
                            drop_mid = _rtj(j["drop_lower"], j["drop_upper"], 0.5)
                            dl = lift_target + touchdown_vertical_reset * (
                                lift_mid - lift_target
                            )
                            dd = drop_target + touchdown_vertical_reset * (
                                drop_mid - drop_target
                            )
                            ds = _rtj(j["swing_lower"], j["swing_upper"], 0.5)
                            targets[j["lift_idx"]] = dl + blend * (lift_target - dl)
                            targets[j["drop_idx"]] = dd + blend * (drop_target - dd)
                            targets[j["swing_idx"]] = ds + blend * (
                                _rtj(j["swing_lower"], j["swing_upper"], sr) - ds
                            )
                            continue
                else:
                    if touchdown_settle_steps > 0:
                        touchdown_ramp[lid] = 0
                    else:
                        touchdown_ramp.pop(lid, None)

                if contact_feedback_enabled and not is_sw:
                    if feedback_contacts is not None and lid in feedback_contacts:
                        no_contact_stance_steps[lid] = 0
                        if latch_stance_search:
                            stance_search_offsets[lid] = max(
                                0.0,
                                stance_search_offsets[lid]
                                - stance_search_ratio / stance_search_release_steps,
                            )
                        else:
                            stance_search_offsets[lid] = 0.0
                    else:
                        no_contact_stance_steps[lid] += 1
                        if latch_stance_search:
                            stance_search_offsets[lid] = min(
                                stance_search_ratio,
                                stance_search_offsets[lid]
                                + stance_search_ratio / stance_search_steps,
                            )
                    search = (
                        stance_search_offsets[lid]
                        if latch_stance_search
                        else stance_search_ratio * min(
                            no_contact_stance_steps[lid] / stance_search_steps,
                            1.0,
                        )
                    )
                    # This term acts under load, where compliance dominates
                    # the zero-pose geometric derivative used by reset
                    # calibration.  Keep the empirically validated direction.
                    lr += search
                    dr += 0.70 * search
                elif is_sw:
                    no_contact_stance_steps[lid] = 0
                    stance_search_offsets[lid] = 0.0

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

            if phase_gate_enabled:
                try:
                    post_step_contacts = _ground_contact_legs()
                except Exception:
                    post_step_contacts = feedback_contacts
                phase_increment = TWO_PI * gait_frequency * dt
                for lid, state in command_states.items():
                    if lid in admitted_swing:
                        if post_step_contacts is not None and lid not in post_step_contacts:
                            swing_airborne.add(lid)
                        touched_down = (
                            post_step_contacts is not None
                            and lid in post_step_contacts
                            and lid in swing_airborne
                            and state.swing_progress
                            >= contact_gate_config.early_touchdown_progress
                        )
                        if touched_down:
                            local_phases[lid] = 0.0
                            swing_airborne.discard(lid)
                            continue
                        next_phase = local_phases[lid] + phase_increment
                        if next_phase >= TWO_PI and (
                            post_step_contacts is None or lid not in post_step_contacts
                        ):
                            local_phases[lid] = TWO_PI - 1e-6
                        else:
                            local_phases[lid] = wrap_2pi(next_phase)
                    else:
                        swing_airborne.discard(lid)
                        local_phases[lid] = wrap_2pi(
                            local_phases[lid] + phase_increment
                        )
                previous_swing = set(admitted_swing)

            _sflag = ga.STATE_ALL if return_yaw_stats else ga.STATE_POS
            states = gym.get_actor_rigid_body_states(env, actor, ga.STATE_ALL)
            body_yaw = 0.0
            body_roll = None
            body_pitch = None
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

            if (
                diagnostic_accumulator is not None
                and step % max(int(diagnostic_stride), 1) == 0
            ):
                actual_contacts = None
                contact_speeds = None
                actual_ssm = None
                leg_body_collisions = None
                leg_leg_collisions = None
                try:
                    contacts = gym.get_env_rigid_contacts(env)
                    actual_set = set()
                    leg_body_collisions = 0
                    leg_leg_collisions = 0
                    for contact in contacts:
                        if float(contact["lambda"]) <= 1e-6:
                            continue
                        body0 = int(contact["body0"])
                        body1 = int(contact["body1"])
                        if body1 == -1 and body0 in self.terminal_body_by_leg.values():
                            actual_set.add(self.body_leg_ids[body0])
                        elif body0 == -1 and body1 in self.terminal_body_by_leg.values():
                            actual_set.add(self.body_leg_ids[body1])
                        if body0 == 0 and body1 in self.body_leg_ids:
                            leg_body_collisions += 1
                        elif body1 == 0 and body0 in self.body_leg_ids:
                            leg_body_collisions += 1
                        if body0 in self.body_leg_ids and body1 in self.body_leg_ids:
                            if self.body_leg_ids[body0] != self.body_leg_ids[body1]:
                                leg_leg_collisions += 1
                    actual_contacts = sorted(actual_set)
                    contact_speeds = {}
                    contact_positions = {}
                    if states is not None:
                        for leg_id in actual_contacts:
                            body_index = self.terminal_body_by_leg.get(leg_id)
                            if body_index is None or body_index >= len(states):
                                continue
                            velocity = states["vel"]["linear"][body_index]
                            contact_speeds[leg_id] = math.hypot(
                                float(velocity["x"]), float(velocity["y"])
                            )
                            position = states["pose"]["p"][body_index]
                            contact_positions[leg_id] = [
                                float(position["x"]), float(position["y"])
                            ]
                        if len(contact_positions) >= 3:
                            root = states["pose"]["p"][0]
                            actual_ssm = contact_support_ssm(
                                contact_positions, [float(root["x"]), float(root["y"])]
                            )
                        else:
                            actual_ssm = 0.0
                except Exception as exc:
                    diagnostic_accumulator.mark_unavailable(
                        "actual_contact_leg_ids", f"Isaac contact query failed: {exc}"
                    )
                    diagnostic_accumulator.mark_unavailable(
                        "contact_mismatch_ratio", "actual contacts unavailable"
                    )
                    diagnostic_accumulator.mark_unavailable(
                        "foot_slip_ratio", "actual contacts unavailable"
                    )
                    diagnostic_accumulator.mark_unavailable(
                        "leg_body_collision_count", "rigid contacts unavailable"
                    )
                    diagnostic_accumulator.mark_unavailable(
                        "leg_leg_collision_count", "rigid contacts unavailable"
                    )

                position_ratio = velocity_ratio = torque_ratio = None
                try:
                    dof_state = gym.get_actor_dof_states(env, actor, ga.STATE_ALL)
                    positions = np.asarray(dof_state["pos"], dtype=float)
                    velocities = np.asarray(dof_state["vel"], dtype=float)
                    half_range = 0.5 * (self.upper - self.lower)
                    center = 0.5 * (self.upper + self.lower)
                    valid_position = np.isfinite(half_range) & (half_range > 1e-9)
                    if np.any(valid_position):
                        position_ratio = float(np.max(
                            np.abs(positions[valid_position] - center[valid_position])
                            / half_range[valid_position]
                        ))
                    valid_velocity = (
                        np.isfinite(self.velocity_limits) & (self.velocity_limits > 1e-9)
                    )
                    if np.any(valid_velocity):
                        velocity_ratio = float(np.max(
                            np.abs(velocities[valid_velocity])
                            / self.velocity_limits[valid_velocity]
                        ))
                    if self.dof_force_sensors_enabled:
                        forces = np.asarray(
                            gym.get_actor_dof_forces(env, actor), dtype=float
                        )
                        valid_effort = (
                            np.isfinite(self.effort_limits) & (self.effort_limits > 1e-9)
                        )
                        if np.any(valid_effort):
                            torque_ratio = float(np.max(
                                np.abs(forces[valid_effort])
                                / self.effort_limits[valid_effort]
                            ))
                except Exception as exc:
                    diagnostic_accumulator.mark_unavailable(
                        "joint_state_ratios", f"Isaac DOF telemetry failed: {exc}"
                    )

                if states is not None and len(states) > 0:
                    root = states["pose"]["p"][0]
                    body_position = [
                        float(root["x"]), float(root["y"]), float(root["z"])
                    ]
                    body_rpy = [float(body_roll), float(body_pitch), float(body_yaw)]
                else:
                    body_position = None
                    body_rpy = None
                diagnostic_accumulator.update(EpisodeStepTelemetry(
                    time_s=sim_time,
                    body_position=body_position,
                    body_rpy=body_rpy,
                    yaw_reference=_planned_yaw,
                    commanded_stance_leg_ids=sorted(commanded_stance),
                    actual_contact_leg_ids=actual_contacts,
                    contact_foot_speed=contact_speeds,
                    leg_body_collision_count=leg_body_collisions,
                    leg_leg_collision_count=leg_leg_collisions,
                    joint_position_ratio=position_ratio,
                    joint_velocity_ratio=velocity_ratio,
                    joint_torque_ratio=torque_ratio,
                    actual_contact_ssm=actual_ssm,
                ))

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
                    # Course calibration must not change the probed motor
                    # controller.  Keep cross-track amplitude feedback in the
                    # original actuation frame; ``lat`` is reserved for
                    # evaluating the independently learned realised course.
                    lateral_error = float(np.dot(pn - p0, actuation_lat))
                    if len(com_trail) >= 61:
                        p_prev = np.asarray(com_trail[-61][:2], dtype=float)
                        lateral_velocity = float(np.dot(
                            pn - p_prev, actuation_lat,
                        ))
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
        if diagnostic_accumulator is not None:
            diagnostics = diagnostic_accumulator.finalize().to_dict()
            diagnostics["reset_contact_calibration"] = reset_calibration
            diagnostics["support_rescue"] = {
                "enabled": bool(emergency_contact_recovery),
                "minimum_contact_trigger": int(emergency_support_count),
                "active_steps": int(support_rescue_steps),
                "active_fraction": float(support_rescue_steps / max(n_steps, 1)),
            }
            return (
                com_trail,
                forward_axis,
                yaw_stats,
                diagnostics,
            )
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
