#!/usr/bin/env python3
"""Drive a REAL FR3 of the Duo with the sim's Cartesian impedance law: a step straight up.

    python real_arm_step.py --dry-run          # connect, read, compute torques, send NOTHING
    python real_arm_step.py                    # right arm: ramp 5 cm up, hold, stop
    python real_arm_step.py --return           # ... and ramp back down to where it started

Same controller as the sim. `RealArmImpedance` subclasses duo_control.CartesianImpedance and
replaces only where the state comes from: pose, Jacobian, mass matrix, q and dq are read from
libfranka (robot state + model) instead of MuJoCo. compute() -- stiffness, damping design, the
offset clamp, the null-space posture term and the joint-limit barrier -- is the sim's code,
unchanged, with the gains DuoSim runs the teleop sim at.

Frames. Everything runs in the arm's own base frame O (link0), which is what O_T_EE and the
zero Jacobian are expressed in. On the Duo that frame is tilted ~60 deg by the mount, so "up"
is NOT +z of O: the world-up direction is read out of the MuJoCo model instead. At the real
joint angles the sim's TCP pose and Jacobian match libfranka's to 1e-9.

Torque path. franka::Torques are sent WITHOUT gravity: the control box adds it, and on this
robot it already accounts for the tilted mount (tau_J at rest matches the sim's gravity to
0.6 Nm). libfranka's Model.gravity() assumes an upright arm and is off by ~30 Nm here -- never
add it. Coriolis comes from Model.coriolis(), as in the libfranka examples. Together that is
the hardware twin of the sim's `compute() + qfrc_bias`.

Safety. The target is ramped (minimum jerk), not stepped. The loop aborts -- ending the motion,
which hands the arm back to the robot's own position hold -- if the TCP strays more than
ABORT_TRACK from the ramped target, the orientation error exceeds ABORT_ROT, any joint moves
faster than ABORT_DQ, or on Ctrl-C. Collision reflex thresholds are the libfranka example
defaults. Keep a hand on the external stop.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

import mujoco
import numpy as np
import pylibfranka as fr

from duo_control import ARM, CartesianImpedance, ImpedanceGains, mat_to_quat

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, 'mobile_fr3_duo.xml')
IPS = {'right': '172.16.16.11', 'left': '172.16.16.12'}

MAX_TAU_RATE = 1000.0      # Nm/s, the FCI limit on the commanded torque derivative
ABORT_TRACK = 0.03         # m, TCP distance from the ramped target
ABORT_ROT = 0.15           # rad, orientation error
ABORT_DQ = 1.0             # rad/s, any joint

# libfranka examples_common.cpp setDefaultBehavior().
COLLISION_TAU = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
COLLISION_F = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]

# FR3 joint limits as the control box enforces them. Its joint velocity limit depends on
# position, v_max = -a + sqrt(k * (c - q)), and reaches zero at these positions (libfranka
# rate_limiting.h, computeUpper/LowerLimitsJointVelocity). They are narrower than the URDF and
# MuJoCo ranges (joint 1: 2.743 against 2.901), so a barrier built on those let the arm reach the
# real limit first and trip joint_velocity_violation at only 0.05 rad/s.
_V_A = np.array([0.30, 0.20, 0.20, 0.30, 0.35, 0.35, 0.35])
_V_K = np.array([12.0, 5.17, 7.00, 8.00, 34.0, 11.0, 34.0])
FR3_Q_LO = np.array([-2.7501, -1.7918, -2.9065, -3.0481, -2.8101, 0.54092, -3.0196]) + _V_A**2 / _V_K
FR3_Q_HI = np.array([2.7501, 1.7918, 2.9065, -0.1458, 2.8101, 4.5205, 3.0196]) - _V_A**2 / _V_K


def arm_frame_from_sim(prefix):
    """World-up expressed in the arm's base frame, plus the joint limits, from the MuJoCo model.

    Invariant under base yaw and spine height: the base is planar (yaw about world-up only) and
    the spine is a pure translation, so neither changes where "up" points in link0.
    """
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'{prefix}_{ARM}_link0')
    up = d.xmat[b].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'{prefix}_{ARM}_joint{i}')
           for i in range(1, 8)]
    return up, m.jnt_range[jid][:, 0].copy(), m.jnt_range[jid][:, 1].copy()


class RealArmImpedance(CartesianImpedance):
    """CartesianImpedance with its state read from libfranka instead of MuJoCo.

    Does not call super().__init__ -- there is no MjData behind this arm. Call update() with
    each new RobotState before compute().
    """

    def __init__(self, franka_model, q_lo, q_hi, gains: ImpedanceGains):
        self.fm, self.gains = franka_model, gains
        self.q_lo, self.q_hi = np.maximum(q_lo, FR3_Q_LO), np.minimum(q_hi, FR3_Q_HI)
        self.vel_d = np.zeros(6)
        self.last_error = np.zeros(6)

    def update(self, state):
        self._q = np.array(state.q)
        self._dq = np.array(state.dq)
        T = np.array(state.O_T_EE).reshape(4, 4, order='F')     # libfranka is column-major
        self._pos, self._quat = T[:3, 3].copy(), mat_to_quat(T[:3, :3])
        self._J = np.array(self.fm.zero_jacobian(state)).reshape(6, 7, order='F')
        self._M = np.array(self.fm.mass(state)).reshape(7, 7, order='F')

    @property
    def q(self):
        return self._q

    @property
    def dq(self):
        return self._dq

    def tcp_pose(self):
        return self._pos.copy(), self._quat.copy()

    def jacobian(self):
        return self._J

    def task_inertia(self, J):
        Minv = np.linalg.inv(self._M + 1e-9 * np.eye(7))
        return np.linalg.inv(J @ Minv @ J.T + 1e-4 * np.eye(6)), Minv


def min_jerk(t, T):
    """Normalised position and velocity (per second) of a minimum-jerk ramp of length T."""
    s = min(max(t / T, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5, (30 * s**2 - 60 * s**3 + 30 * s**4) / T


class Profile:
    """Piecewise target: ramp p0 -> goal, hold, and optionally ramp back and hold."""

    def __init__(self, p0, goal, ramp, hold, back):
        self.segs = [(p0, goal, ramp), (goal, goal, hold)]
        if back:
            self.segs += [(goal, p0, ramp), (p0, p0, 1.0)]
        self.duration = sum(s[2] for s in self.segs)

    def __call__(self, t):
        for a, b, T in self.segs:
            if t < T:
                s, ds = min_jerk(t, T)
                return a + s * (b - a), ds * (b - a)
            t -= T
        return self.segs[-1][1], np.zeros(3)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--arm', choices=('right', 'left'), default='right')
    ap.add_argument('--ip', help='override the arm IP')
    ap.add_argument('--height', type=float, default=0.05, help='m to move up (default 0.05)')
    ap.add_argument('--ramp', type=float, default=3.0, help='s for each ramp')
    ap.add_argument('--hold', type=float, default=3.0, help='s to hold at the top')
    ap.add_argument('--return', dest='back', action='store_true', help='ramp back down after')
    ap.add_argument('--translational', type=float, default=900.0, help='N/m (DuoSim: 900)')
    ap.add_argument('--rotational', type=float, default=60.0, help='Nm/rad (DuoSim: 60)')
    ap.add_argument('--dry-run', action='store_true', help='read and compute only; no motion')
    ap.add_argument('--log', help='save the run to this .npz')
    args = ap.parse_args()
    if not 0.0 < args.height <= 0.10:
        sys.exit('--height must be in (0, 0.10] m')

    up, q_lo, q_hi = arm_frame_from_sim(args.arm)
    gains = ImpedanceGains(translational=args.translational, rotational=args.rotational,
                           damping_design='inertia', velocity_feedforward=True)

    robot = fr.Robot(args.ip or IPS[args.arm], fr.RealtimeConfig.kIgnore)
    robot.set_collision_behavior(COLLISION_TAU, COLLISION_TAU, COLLISION_F, COLLISION_F)
    model = robot.load_model()
    arm = RealArmImpedance(model, q_lo, q_hi, gains)

    s = robot.read_once()
    if s.robot_mode != fr.RobotMode.Idle:
        sys.exit(f'robot mode is {s.robot_mode}, need Idle (joints unlocked, FCI on, stop released)')
    arm.update(s)
    p0, quat0 = arm.tcp_pose()
    arm.q_null = arm.q.copy()
    goal = p0 + args.height * up
    prof = Profile(p0, goal, args.ramp, args.hold, args.back)

    np.set_printoptions(precision=4, suppress=True)
    print(f'{args.arm} arm @ {args.ip or IPS[args.arm]}   q = {arm.q}')
    print(f'  TCP now   {p0}  (arm base frame)')
    print(f'  target    {goal}  = +{args.height * 100:.1f} cm along world-up {up}')
    print(f'  gains     K = {gains.translational:.0f} N/m / {gains.rotational:.0f} Nm/rad, '
          f'inertia-shaped damping, velocity feedforward')
    print(f'  profile   {args.ramp:.1f} s ramp, {args.hold:.1f} s hold'
          + (f', {args.ramp:.1f} s back down' if args.back else '') + f'  ({prof.duration:.1f} s)')
    arm.set_target(p0, quat0)
    tau0 = arm.compute() + np.array(model.coriolis(s))
    print(f'  tau at start (target = current pose): {tau0}')

    if args.dry_run:
        arm.set_target(goal, quat0)
        tau_goal = arm.compute()
        print(f'  tau if the full step were applied at once (arm static): {tau_goal}')
        print('dry run: nothing was sent to the robot')
        return

    n_max = int((prof.duration + 5.0) * 1000)
    log = {k: np.zeros((n_max, w)) for k, w in
           (('t', 1), ('dt', 1), ('pos', 3), ('pos_d', 3), ('tau', 7), ('q', 7))}
    n, t, tau_prev, reason = 0, 0.0, np.zeros(7), 'completed'
    step = MAX_TAU_RATE * 1e-3

    ctl = robot.start_torque_control()
    gc.disable()            # a GC pause inside the 1 ms loop is a missed packet
    try:
        while True:
            s, dur = ctl.readOnce()
            dt = dur.to_sec()
            t += dt
            arm.update(s)
            pos_d, vel_d = prof(t)
            arm.set_target(pos_d, quat0, np.concatenate([vel_d, np.zeros(3)]))
            tau = arm.compute() + np.array(model.coriolis(s))
            tau = tau_prev + np.clip(tau - tau_prev, -step, step)       # as franka::limitRate
            tau_prev = tau

            pos, _ = arm.tcp_pose()
            if np.linalg.norm(pos - pos_d) > ABORT_TRACK:
                reason = f'tracking error {np.linalg.norm(pos - pos_d) * 1000:.0f} mm'
            elif np.linalg.norm(arm.last_error[3:]) > ABORT_ROT:
                reason = f'orientation error {np.linalg.norm(arm.last_error[3:]):.3f} rad'
            elif np.abs(arm.dq).max() > ABORT_DQ:
                reason = f'joint speed {np.abs(arm.dq).max():.2f} rad/s'
            aborted = reason != 'completed'
            done = aborted or t >= prof.duration

            cmd = fr.Torques(tau.tolist())
            cmd.motion_finished = done
            ctl.writeOnce(cmd)

            if n < n_max:
                for k, v in (('t', t), ('dt', dt), ('pos', pos), ('pos_d', pos_d),
                             ('tau', tau), ('q', arm.q)):
                    log[k][n] = v
                n += 1
            if done:
                break
    except KeyboardInterrupt:
        reason = 'Ctrl-C'
        robot.stop()
    except fr.FrankaException as e:
        reason = f'libfranka: {e}'
    finally:
        gc.enable()

    log = {k: v[:n] for k, v in log.items()}
    after = robot.read_once()
    err = np.linalg.norm(log['pos'] - log['pos_d'], axis=1) if n else np.zeros(1)
    top = (log['t'][:, 0] > args.ramp + args.hold - 0.5) & (log['t'][:, 0] <= args.ramp + args.hold)
    print(f'\nresult: {reason}   ({n} cycles, {t:.2f} s)')
    if n:
        moved = (log['pos'][top].mean(axis=0) - p0) if top.any() else log['pos'][-1] - p0
        print(f'  moved along world-up {moved @ up * 1000:6.1f} mm   '
              f'(off-axis {np.linalg.norm(moved - (moved @ up) * up) * 1000:.1f} mm)')
        print(f'  tracking error: max {err.max() * 1000:.1f} mm, '
              f'end of hold {err[top].mean() * 1000 if top.any() else float("nan"):.1f} mm')
        print(f'  max |tau| {np.abs(log["tau"]).max(axis=0)}')
        late = int((log['dt'][1:] > 0.0015).sum())
        print(f'  loop: {late} of {n - 1} cycles took >1.5 ms (max {log["dt"][1:].max() * 1000:.1f} ms)'
              if n > 1 else '')
    print(f'  robot mode now {after.robot_mode}, current errors: {after.current_errors}, '
          f'last motion errors: {after.last_motion_errors}')
    if args.log:
        np.savez(args.log, p0=p0, goal=goal, up=up, **log)
        print(f'  log -> {args.log}')


if __name__ == '__main__':
    main()
