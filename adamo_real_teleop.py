#!/usr/bin/env python3
"""Teleoperate the REAL Duo arms with Adamo XR controllers -- the hardware twin of adamo_teleop.py.

    export ADAMO_API_KEY=ak_...
    python adamo_real_teleop.py --name duo-real                  # both arms, grippers, live cameras
    python adamo_real_teleop.py --name duo-real --dry-run        # Adamo live, arms READ-ONLY
    python adamo_real_teleop.py --dry-run --synthetic            # no network, arms read-only

Operator interface is the sim's (see adamo_teleop.py), per hand:

    grip / squeeze      CLUTCH. Hold to move that arm; release to freeze it. Engaging anchors at
                        the arm's current attractor, so there is no jump on engage.
    index trigger       That hand's gripper: the width follows the pull; a full pull grasps.
    thumbstick click    Drop the clutch.
    thumbsticks         Ignored -- this script does not drive the real base or spine.

Video comes from the ZED-M (side-by-side stereo, 720p per eye at 30 fps) and the two D405
wrist RGB cameras (848x480 at 30 fps), with the confirmed left/right mapping.

Controller. Each arm runs duo_control.CartesianImpedance.compute() -- the sim's control law,
unchanged -- on libfranka state, via real_arm_step.RealArmImpedance, with the gains DuoSim uses
for teleop (900 N/m, 60 Nm/rad, inertia-shaped damping, velocity feedforward). Torques are sent
without gravity (the control box adds it, and knows about the tilted mount) plus
Model.coriolis(). See real_arm_step.py for how that was checked against the sim.

Frames. XR motion is mapped into the ROBOT BASE frame exactly as in the sim (xr_to_base_*), then
rotated into each arm's own base frame (link0, tilted ~60 deg by the Duo mount) with the fixed
rotation read from the MuJoCo model. Motion is relative (clutched), so only rotations matter.

Processes. pylibfranka does not release the GIL while it blocks in readOnce(), so two 1 kHz
loops in one process would starve each other, and so would Adamo's network threads. Each arm
therefore runs in its own process, as does each gripper (whose move/grasp calls block for up to
a second). The main process runs Adamo, the clutch logic and the video, and hands each arm a
target pose through shared memory.

Safety.
  * The attractor only moves while the clutch is held AND operator samples are fresh (--lease).
    The arm process also holds if the main process stops refreshing its command.
  * The attractor is rate limited (--max-speed, --max-rot-speed), so a tracking glitch or a
    re-anchor cannot slam the arm; the controller's own offset clamp bounds force to ~54 N.
  * Torque commands are low-passed at 100 Hz, as libfranka's own control loop does
    (pylibfranka's writeOnce does not), and rate limited to 500 Nm/s against the last command
    the robot actually received (tau_J_d) as well as our own. The target's acceleration is
    capped (--max-accel). Arm loops run at SCHED_FIFO 80 with memory locked. On an abnormal
    stop the last 3 s of commands are saved under /tmp/adamo_real_teleop/.
  * Targets are kept within --max-reach of the pose each arm had at startup.
  * An arm stops (hands back to the robot's position hold) if any joint exceeds --abort-dq, on a
    libfranka reflex, or on Ctrl-C. Collision thresholds are the libfranka example defaults.
  * Keep a hand on the external stop.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time

import mujoco
import numpy as np

from adamo_teleop import (BTN_GRIP, BTN_THUMB, BTN_TRIGGER, FPS, _analog,
                          xr_to_base_pos, xr_to_base_quat)
from duo_control import ARM, ImpedanceGains, mat_to_quat, quat_conj, quat_mul, quat_to_mat
from real_arm_step import IPS, XML, arm_frame_from_sim

SIDES = ('left', 'right')
REF_HZ = 3.0                    # natural frequency of the attractor's reference filter
TAU_CUTOFF_HZ = 100.0           # low-pass on torque commands: libfranka's kDefaultCutoffFrequency
TAU_RATE = 500.0                # Nm/s; half the FCI limit (1000), leaving room for a lost packet
RING_N = 3000                   # cycles of commands kept for the post-mortem dump (3 s)
ARM_RT_PRIORITY = 80            # SCHED_FIFO for the arm loops (this account's rtprio cap is 95)
GRASP_AT = 0.9                  # trigger pull above which the gripper grasps instead of moving

# Shared-memory layouts (doubles).
CMD_N = 9                       # [stamp, active, pos(3), quat(4)] in the arm's base frame
ST_STATUS, ST_BEAT, ST_POS, ST_QUAT, ST_APOS, ST_AQUAT, ST_LATE, ST_CYC, ST_TAU = \
    0, 1, slice(2, 5), slice(5, 9), slice(9, 12), slice(12, 16), 16, 17, 18
ST_EXT = slice(19, 26)          # filtered external joint torques (Nm)
STATE_N = 26
S_START, S_RUN, S_DRY, S_STOPPED = 0, 1, 2, 3
STATUS_NAME = {S_START: 'starting', S_RUN: 'TORQUE CONTROL', S_DRY: 'dry-run (read-only)',
               S_STOPPED: 'STOPPED'}


def base_from_arm_rotation(prefix):
    """R such that v_base = R @ v_link0, from the MuJoCo model (fixed: the mount is rigid)."""
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    def rot(name):
        return d.xmat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)].reshape(3, 3)
    return rot('base_link').T @ rot(f'{prefix}_{ARM}_link0')


def snapshot(arr):
    with arr.get_lock():
        return np.array(arr[:])


# ---------------------------------------------------------------------------------------------
# Arm process: owns one robot connection and its 1 kHz torque loop.


class AttractorFollower:
    """Moves the impedance attractor toward the operator command, smoothly.

    Position runs through a critically damped second-order filter (natural frequency REF_HZ)
    whose velocity is capped at --max-speed. That velocity is also the feedforward velocity
    -- the real-robot counterpart of MobileDuo differentiating the operator stream -- and it
    must be continuous: it is multiplied by the damping matrix D (~100-200 N.s/m here), and
    the robot rejects torque commands that are not smooth (controller_torque_discontinuity).
    The first version was a bare speed limit, whose velocity jumped between 0 and max_speed
    at every 72 Hz operator sample; D*v turned each jump into a torque step at the rate
    limit and tripped both arms right after clutch-in.

    Its acceleration is capped too (--max-accel), because D*a is the rate at which the
    feedforward moves the torque. Uncapped, the filter still steps its acceleration at every
    operator sample -- up to 10 m/s^2 on the real right arm, ~900 Nm/s of torque demand.

    When inactive, the attractor bleeds its velocity off (time constant 1/(4*pi*REF_HZ),
    ~10 ms) rather than stopping dead, for the same reason.

    Orientation stays a rate-limited slerp: it is not fed forward, and at 60 Nm/rad the
    rotational spring changes by under 0.1 Nm per cycle at --max-rot-speed.
    """

    def __init__(self, cmd, pos, quat, opts):
        self.cmd, self.opts = cmd, opts
        self.pos, self.quat = pos.copy(), quat.copy()
        self.vel = np.zeros(3)
        self.active = False

    def step(self, dt):
        dt = min(max(dt, 1e-4), 0.005)
        c = snapshot(self.cmd)
        self.active = c[1] > 0.5 and (time.monotonic() - c[0]) < self.opts['lease']
        w = 2.0 * np.pi * REF_HZ
        pull = w * w * (c[2:5] - self.pos) if self.active else 0.0
        acc = pull - 2.0 * w * self.vel
        na = np.linalg.norm(acc)
        if na > self.opts['max_accel']:
            acc *= self.opts['max_accel'] / na
        self.vel = self.vel + acc * dt
        n = np.linalg.norm(self.vel)
        if n > self.opts['max_speed']:
            self.vel *= self.opts['max_speed'] / n
        self.pos = self.pos + self.vel * dt
        if self.active:
            q_cmd = c[5:9] / np.linalg.norm(c[5:9])
            qe = quat_mul(q_cmd, quat_conj(self.quat))
            if qe[0] < 0.0:
                qe = -qe
            ang = 2.0 * np.arccos(np.clip(qe[0], -1.0, 1.0))
            lim = self.opts['max_rot_speed'] * dt
            if ang > lim:
                axis = qe[1:] / np.linalg.norm(qe[1:])
                qe = np.concatenate([[np.cos(lim / 2)], np.sin(lim / 2) * axis])
            self.quat = quat_mul(qe, self.quat)
            self.quat /= np.linalg.norm(self.quat)
        return np.concatenate([self.vel, np.zeros(3)])


def arm_process(side, ip, opts, cmd, state, stop, msgs):
    signal.signal(signal.SIGINT, signal.SIG_IGN)     # the main process owns shutdown, via `stop`
    import ctypes
    import pylibfranka as fr
    from real_arm_step import COLLISION_F, COLLISION_TAU, RealArmImpedance

    # Realtime scheduling. libfranka's own attempt fails here only because it asks for
    # priority 99 and this account is capped at 95, so do it ourselves. Without it, camera
    # encoding in the main process can preempt this loop past its 1 ms command window.
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(ARM_RT_PRIORITY))
        rt_note = f'SCHED_FIFO {ARM_RT_PRIORITY}'
        if ctypes.CDLL('libc.so.6', use_errno=True).mlockall(3) == 0:   # MCL_CURRENT|FUTURE
            rt_note += ', memory locked'
        else:
            rt_note += f', memory NOT locked ({os.strerror(ctypes.get_errno())})'
    except OSError as e:
        rt_note = f'NO realtime priority ({e}) -- expect dropped packets'

    def say(text):
        msgs.put((side, text))

    def publish(status, arm, fol, late, cycles, tau, external_torque):
        p, q = arm.tcp_pose()
        with state.get_lock():
            state[ST_STATUS], state[ST_BEAT] = status, time.monotonic()
            state[ST_POS], state[ST_QUAT] = list(p), list(q)
            state[ST_APOS], state[ST_AQUAT] = list(fol.pos), list(fol.quat)
            state[ST_LATE], state[ST_CYC], state[ST_TAU] = late, cycles, float(np.abs(tau).max())
            state[ST_EXT] = external_torque

    try:
        robot = fr.Robot(ip, fr.RealtimeConfig.kIgnore)
        model = robot.load_model()
        gains = ImpedanceGains(translational=opts['translational'], rotational=opts['rotational'],
                               damping_design='inertia', velocity_feedforward=True)
        arm = RealArmImpedance(model, opts['q_lo'], opts['q_hi'], gains)
        s = robot.read_once()
        if s.robot_mode != fr.RobotMode.Idle:
            raise RuntimeError(f'robot mode is {s.robot_mode}, need Idle '
                               '(joints unlocked, FCI on, external stop released)')
        arm.update(s)
        arm.q_null = arm.q.copy()
        fol = AttractorFollower(cmd, *arm.tcp_pose(), opts)
        with cmd.get_lock():                          # start out holding exactly here
            cmd[1] = 0.0
            cmd[2:5], cmd[5:9] = list(fol.pos), list(fol.quat)
    except Exception as e:                            # noqa: BLE001 -- report anything
        say(f'failed to start: {e}')
        state[ST_STATUS] = S_STOPPED
        return

    late = cycles = 0
    tau = np.zeros(7)
    if opts['dry_run']:
        # Read-only: the same follower and control law, torques computed and discarded.
        last = time.monotonic()
        try:
            while not stop.is_set():
                s = robot.read_once()
                now = time.monotonic()
                dt, last = now - last, now
                arm.update(s)
                arm.set_target(fol.pos, fol.quat, fol.step(dt))
                tau = arm.compute() + np.array(model.coriolis(s))
                cycles += 1
                if cycles % 5 == 0:
                    publish(S_DRY, arm, fol, late, cycles, tau, s.tau_ext_hat_filtered)
        except Exception as e:                        # noqa: BLE001
            say(f'dry run STOPPED by {type(e).__name__}: {e}')
        state[ST_STATUS] = S_STOPPED
        return

    reason, abnormal = 'stopped by operator', False
    step = TAU_RATE * 1e-3
    # libfranka's own control loop runs every torque command through a first-order low-pass
    # (100 Hz) and franka::limitRate before sending it. pylibfranka's writeOnce() does
    # neither, and without the filter the robot trips controller_torque_discontinuity.
    gain = 1e-3 / (1e-3 + 1.0 / (2.0 * np.pi * TAU_CUTOFF_HZ))   # libfranka lowpassFilter
    filt = np.zeros(7)
    saturated = 0
    ring = {k: np.zeros((RING_N, w)) for k, w in
            (('dt', 1), ('tau_raw', 7), ('tau', 7), ('tau_J_d', 7), ('tcp', 3), ('attractor', 3),
             ('vel_ff', 3), ('q', 7), ('dq', 7), ('success', 1), ('loop_us', 1))}
    tau_th = [opts['coll_tau']] * 7 if opts['coll_tau'] else COLLISION_TAU
    f_th = [opts['coll_f']] * 6 if opts['coll_f'] else COLLISION_F
    # Warm up the per-cycle maths once: the first pass took ~5 ms (first-call allocations and
    # page faults under mlockall), which would make the first command miss its window.
    s = robot.read_once()
    arm.update(s)
    arm.set_target(fol.pos, fol.quat, np.zeros(6))
    warm = arm.compute() + np.array(model.coriolis(s))
    np.clip(warm, np.maximum(warm, tau) - step, np.minimum(warm, tau) + step)
    try:
        robot.set_collision_behavior(tau_th, tau_th, f_th, f_th)
        ctl = robot.start_torque_control()
        say(f'torque control running ({rt_note}) -- holding current pose')
        gc.disable()                                  # a GC pause here is a missed packet
        while True:
            s, dur = ctl.readOnce()
            t_loop = time.perf_counter()
            dt = dur.to_sec()
            late += dt > 0.0015
            arm.update(s)
            vel = fol.step(dt)
            arm.set_target(fol.pos, fol.quat, vel)
            raw = arm.compute() + np.array(model.coriolis(s))
            filt = filt + gain * (raw - filt)
            # The robot computes the torque rate against the last command it RECEIVED, which
            # it echoes back as tau_J_d (libfranka docs/overview.rst). After a dropped packet
            # that is older than our own last command, so limiting against ours alone let one
            # drop double the step -- the right arm's controller_torque_discontinuity with
            # every logged step at 0.5 Nm. Stay within `step` of both.
            tjd = np.array(s.tau_J_d)
            lo, hi = np.maximum(tau, tjd) - step, np.minimum(tau, tjd) + step
            apart = lo > hi                           # only after several drops in a row
            lo[apart], hi[apart] = tjd[apart] - step, tjd[apart] + step
            new = np.clip(filt, lo, hi)
            saturated += bool(np.any(new != filt))
            tau = new

            fault = np.abs(arm.dq).max() > opts['abort_dq']
            if fault:
                reason, abnormal = f'ABORT: joint speed {np.abs(arm.dq).max():.2f} rad/s', True
            finish = fault or stop.is_set()
            out = fr.Torques(tau.tolist())
            out.motion_finished = finish
            ctl.writeOnce(out)

            i = cycles % RING_N
            ring['loop_us'][i] = (time.perf_counter() - t_loop) * 1e6
            ring['dt'][i], ring['tau_raw'][i], ring['tau'][i] = dt, raw, tau
            ring['tau_J_d'][i], ring['success'][i] = tjd, s.control_command_success_rate
            ring['tcp'][i], ring['attractor'][i] = arm._pos, fol.pos
            ring['vel_ff'][i], ring['q'][i], ring['dq'][i] = vel[:3], arm.q, arm.dq
            cycles += 1
            if cycles % 5 == 0:
                publish(S_RUN, arm, fol, late, cycles, tau, s.tau_ext_hat_filtered)
            if finish:
                break
    except Exception as e:                            # noqa: BLE001
        # Catch everything: pylibfranka's ControlException (reflexes) and NetworkException are
        # RuntimeErrors, NOT FrankaExceptions. Catching only FrankaException let a reflex kill
        # this process silently while the status still read TORQUE CONTROL.
        reason, abnormal = f'STOPPED by {type(e).__name__}: {e}', True
    finally:
        gc.enable()
    state[ST_STATUS] = S_STOPPED
    try:
        s = robot.read_once()
        errs = f'\n    robot mode {s.robot_mode}, last motion errors: {s.last_motion_errors}'
    except Exception:                                 # noqa: BLE001
        errs = ''
    dump = ''
    if cycles:
        order = np.arange(max(0, cycles - RING_N), cycles) % RING_N
        dump = (f'\n    last {len(order) / 1000:.1f} s: command success rate min '
                f'{ring["success"][order].min():.2f}, loop compute max '
                f'{ring["loop_us"][order].max():.0f} us')
    if abnormal and cycles:
        path = os.path.join('/tmp/adamo_real_teleop', f'{side}-{time.strftime("%Y%m%d-%H%M%S")}.npz')
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, **{k: v[order] for k, v in ring.items()})
            dump += f'\n    last {len(order) / 1000:.1f} s of commands saved to {path}'
        except OSError as e:
            dump += f'\n    could not save the command log: {e}'
    hint = ''
    if 'torque_discontinuity' in reason:
        hint = ('\n    -> the robot rejected a torque command that changed too abruptly. Send the '
                'saved log for analysis; clear the error in Desk or restart this script.')
    elif 'reflex' in reason.lower():
        hint = ('\n    -> safety reflex. For cartesian_reflex / joint_reflex (the external-force '
                'estimate crossed the collision threshold) try --collision-force 60 '
                '--collision-torque 40; clear the error in Desk or restart this script.')
    say(f'{reason}; {cycles} cycles, {late} late, {saturated} rate-limited{errs}{dump}{hint}')


def gripper_process(side, ip, gcmd, stop, msgs, force):
    """Latest-wins gripper driver: trigger 0..1 -> width; a full pull grasps with `force` N."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import pylibfranka as fr
    try:
        g = fr.Gripper(ip)
        max_w = g.read_once().max_width
    except Exception as e:                            # noqa: BLE001
        msgs.put((side, f'gripper unavailable: {e}'))
        return
    msgs.put((side, f'gripper ready, max width {max_w * 1000:.0f} mm'))
    last = None
    while not stop.is_set():
        trig = snapshot(gcmd)[0]
        try:
            if np.isnan(trig):
                pass
            elif trig >= GRASP_AT:
                if last != 'grasp':
                    g.grasp(0.0, 0.1, force, 0.005, max_w)   # accepts any object width
                    last = 'grasp'
            else:
                w = max_w * (1.0 - trig / GRASP_AT)
                if last is None or last == 'grasp' or abs(w - last) > 0.005:
                    g.move(w, 0.1)
                    last = w
        except Exception as e:                        # noqa: BLE001 -- e.g. blocked by an object
            msgs.put((side, f'gripper: {e}'))
            last = None
            time.sleep(0.2)
        time.sleep(0.01)


# ---------------------------------------------------------------------------------------------
# Main process: operator side.


class RealArmChannel:
    """Clutched operator -> arm mapping, as adamo_teleop.ArmChannel but for a real arm.

    Operator motion is measured in the robot BASE frame and applied in the arm's link0 frame:
      p = p_anchor + R_ab * d_base
      R = (R_ab * dR_base * R_ab^T) * R_anchor         (delta on the left, as in the sim)
    """

    def __init__(self, side, R_base_arm, cmd, state, gcmd, opts):
        self.side, self.cmd, self.state, self.gcmd, self.opts = side, cmd, state, gcmd, opts
        self.R_ab = R_base_arm.T                     # base -> arm
        st = snapshot(state)
        self.home = st[ST_POS].copy()
        self.engaged = False
        self.anchor_ctrl = self.anchor_tgt = self.target = None
        self.last_pose = None
        self.last_rx = 0.0

    def set_clutch(self, engaged):
        if engaged and not self.engaged:
            if self.last_pose is None:
                return
            st = snapshot(self.state)
            if st[ST_STATUS] not in (S_RUN, S_DRY):
                return
            self.anchor_ctrl = self.last_pose
            self.anchor_tgt = (st[ST_APOS].copy(), st[ST_AQUAT].copy())
            self.target = self.anchor_tgt
            self.engaged = True
            print(f'[{self.side}] clutch IN')
        elif not engaged and self.engaged:
            self.engaged = False
            print(f'[{self.side}] clutch out')

    def update(self, pos_base, quat_base):
        self.last_rx = time.monotonic()
        self.last_pose = (np.asarray(pos_base, float), np.asarray(quat_base, float))
        if not self.engaged:
            return
        d = self.R_ab @ ((self.last_pose[0] - self.anchor_ctrl[0]) * self.opts['scale'])
        p = self.anchor_tgt[0] + d
        off = p - self.home
        n = np.linalg.norm(off)
        if n > self.opts['max_reach']:
            p = self.home + off * self.opts['max_reach'] / n
        dq = quat_mul(self.last_pose[1], quat_conj(self.anchor_ctrl[1]))
        dR = self.R_ab @ quat_to_mat(dq) @ self.R_ab.T
        self.target = (p, mat_to_quat(dR @ quat_to_mat(self.anchor_tgt[1])))

    def set_trigger(self, value):
        if self.gcmd is not None:
            with self.gcmd.get_lock():
                self.gcmd[0], self.gcmd[1] = value, time.monotonic()

    def push(self):
        fresh = (time.monotonic() - self.last_rx) < self.opts['lease']
        active = self.engaged and fresh and self.target is not None
        with self.cmd.get_lock():
            self.cmd[0], self.cmd[1] = time.monotonic(), float(active)
            if active:
                self.cmd[2:5], self.cmd[5:9] = list(self.target[0]), list(self.target[1])


def handle_xr_buttons(ch, axes, buttons, ax_trigger, ax_grip):
    grip, _ = _analog(axes, buttons, ax_grip, BTN_GRIP)
    ch.set_clutch(grip > 0.5)
    trig, _ = _analog(axes, buttons, ax_trigger, BTN_TRIGGER)
    ch.set_trigger(trig)
    if len(buttons) > BTN_THUMB and buttons[BTN_THUMB]:
        ch.set_clutch(False)


def wire_adamo(robot, name, channels, lock, args):
    from adamo import xr
    seen = {'n': 0}

    def side_of(topic):
        t = topic.lower()
        return 'left' if 'left' in t else ('right' if 'right' in t else None)

    def on_xr(sample):
        side, msg = side_of(sample.topic), sample.message
        if args.debug_input and seen['n'] < 40:
            seen['n'] += 1
            extra = (f'axes={list(getattr(msg, "axes", []))} '
                     f'buttons={list(getattr(msg, "buttons", []))}'
                     if isinstance(msg, xr.XRJoy) else '')
            print(f'[xr] topic={sample.topic!r} side={side} {type(msg).__name__} {extra}')
        ch = channels.get(side)
        if ch is None:
            return
        with lock:
            if isinstance(msg, xr.PoseStamped):
                ch.update(xr_to_base_pos(msg.position.as_tuple()),
                          xr_to_base_quat(msg.orientation.as_tuple()))
            elif isinstance(msg, xr.XRJoy):
                handle_xr_buttons(ch, list(msg.axes), list(msg.buttons),
                                  args.trigger_axis, args.grip_axis)

    xr.subscribe_xr_control(robot.session, name, on_xr, max_age_seconds=0.25)


def synthetic_operator(channels, lock, stop, t_end):
    """--synthetic: clutch in, then 5 cm up, 5 cm forward, 5 cm left (WebXR frame), and back.

    Each leg is along one robot-base axis, so the status line shows whether operator axes land
    on the right robot axes.
    """
    legs = [np.array([0.0, 0.05, 0.0]),      # WebXR +y = up
            np.array([0.0, 0.0, -0.05]),     # WebXR -z = forward
            np.array([-0.05, 0.0, 0.0])]     # WebXR -x = left
    q_xr = np.array([0.0, 0.0, 0.0, 1.0])
    t0 = time.monotonic()
    while not stop.is_set() and time.monotonic() < t_end:
        t = time.monotonic() - t0
        k, u = int((t - 1.0) // 3.0), ((t - 1.0) % 3.0) / 3.0
        p = np.zeros(3)
        if 0 <= k < len(legs):
            p = legs[k] * np.sin(np.pi * u)             # out and back over 3 s
        with lock:
            for ch in channels.values():
                ch.update(xr_to_base_pos(p), xr_to_base_quat(q_xr))
                ch.set_clutch(t > 1.0)
        time.sleep(1.0 / 72.0)


def camera_robot(args):
    """Validate the three cameras and register their tracks before starting any arms."""
    from pathlib import Path
    import subprocess
    import adamo

    serials = ['133323070779', '133323070232']  # confirmed left, right
    wrists = [
        (name, Path('/dev/v4l/by-id') / (
            'usb-Intel_R__RealSense_TM__Depth_Camera_405_'
            f'Intel_R__RealSense_TM__Depth_Camera_405_{serial}-video-index4'))
        for name, serial in zip(('wrist_left', 'wrist_right'), serials)
    ]
    zed = Path('/dev/v4l/by-id/usb-Technologies__Inc._ZED-M-video-index0')
    for name, device in [*wrists, ('stereo', zed)]:
        if not device.exists() or not os.access(device, os.R_OK | os.W_OK):
            raise RuntimeError(f'Cannot access {name}: {device}. Check the USB cable and device permissions.')
    usb = (Path('/sys/class/video4linux') / zed.resolve().name / 'device').resolve()
    speed = next((p / 'speed' for p in usb.parents if (p / 'speed').exists()), None)
    if speed is None or float(speed.read_text()) < 5000:
        raise RuntimeError('ZED video is not at SuperSpeed. Reconnect it with a USB 3 cable and port.')
    try:
        plugin = subprocess.run(['gst-inspect-1.0', 'zedsrc'], capture_output=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError('Cannot inspect zedsrc. Check the GStreamer tools and ZED SDK installation.') from error
    if plugin.returncode:
        raise RuntimeError('zedsrc could not load. Run `gst-inspect-1.0 zedsrc` and check the ZED SDK installation.')

    robot = adamo.Robot(api_key=args.api_key, name=args.name)
    try:
        options = dict(encoder=None if args.encoder == 'default' else args.encoder,
                       backend=None if args.video_backend == 'default' else args.video_backend)
        for name, device in wrists:
            robot.attach_video(name, device=str(device), width=848, height=480, fps=30,
                               pixel_format='YUY2', bitrate_kbps=1000, **options)
            print(f'{name}: {device.name} (848x480 at 30 fps)', flush=True)
        # Keep the existing stereo track name and register it last as the main view.
        robot.attach_video('stereo', pipeline=(
            'zedsrc camera-sn=17145159 stream-type=5 camera-resolution=3 '
            'camera-fps=30 depth-mode=0 ! queue max-size-buffers=1 leaky=downstream'),
            width=2560, height=720, fps=30, pixel_format='BGRA', stereo=True,
            bitrate_kbps=args.bitrate, **options)
        print('stereo: ZED-M 17145159 (2560x720 side-by-side at 30 fps)', flush=True)
    except BaseException:
        robot.close()
        raise
    return robot


def contact_rumble_intensity(st, now):
    """Initial tuning: 2 Nm deadband, linear rise to 80% at 8 Nm.

    Uses filtered external joint torque, not commanded torque or raw motor load.
    This is contact feedback, not a calibrated force measurement or safety check.
    """
    if st[ST_STATUS] != S_RUN or not 0 <= now - st[ST_BEAT] <= 0.25:
        return 0.0
    torques = st[ST_EXT]
    if len(torques) != 7 or not all(math.isfinite(t) for t in torques):
        return 0.0
    peak = max(abs(t) for t in torques)
    return min(0.8, max(0.0, (peak - 2.0) * (0.8 / 6.0)))


def stream_contact_rumble(robot, arms, quit_evt):
    """Publish short variable pulses at 20 Hz, outside the realtime arm processes."""
    from adamo._native import fabric_now_us

    publisher = None
    failed = False
    announced = False
    while not quit_evt.is_set():
        try:
            if publisher is None:
                publisher = robot.publish('feedback/haptics', reliable=False, express=True)
            for side, (_, state) in arms.items():
                # Never wait for the arm process to release its telemetry lock.
                if not state.get_lock().acquire(False):
                    continue
                try:
                    st = state[:]
                finally:
                    state.get_lock().release()
                intensity = contact_rumble_intensity(st, time.monotonic())
                publisher.put(json.dumps(dict(hand=side, intensity=intensity,
                                              duration_ms=100, ts_us=fabric_now_us())))
            if not announced:
                message = 'Contact rumble publishing: external torque 2-8 Nm maps to 0-80%, per hand at 20 Hz.'
                print(message, flush=True)
                robot.log(message)
                announced = True
            if failed:
                robot.log('Contact rumble feedback restored.')
                failed = False
            quit_evt.wait(0.05)
        except Exception as error:
            if not failed:
                message = f'Contact rumble unavailable: {error}. Retrying automatically; arm control continues.'
                print(message, file=sys.stderr, flush=True)
                try:
                    robot.log(message, level='warning')
                except Exception:
                    pass
                failed = True
            if publisher is not None:
                try:
                    publisher.close()
                except Exception:
                    pass
                publisher = None
            quit_evt.wait(2.0)
    if publisher is not None:
        try:
            publisher.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--name', default='duo-real', help='robot name registered with Adamo')
    ap.add_argument('--api-key', default=os.environ.get('ADAMO_API_KEY'))
    ap.add_argument('--arms', nargs='+', choices=SIDES, default=list(SIDES))
    ap.add_argument('--dry-run', action='store_true',
                    help='arms read-only: torques are computed and never sent; no gripper')
    ap.add_argument('--synthetic', action='store_true',
                    help='with --dry-run: scripted operator, no Adamo connection')
    ap.add_argument('--seconds', type=float, default=12.0, help='--synthetic duration')
    ap.add_argument('--no-gripper', dest='gripper', action='store_false')
    ap.add_argument('--grasp-force', type=float, default=20.0, help='N, on a full trigger pull')
    ap.add_argument('--translational', type=float, default=900.0, help='N/m (DuoSim: 900)')
    ap.add_argument('--rotational', type=float, default=60.0, help='Nm/rad (DuoSim: 60)')
    ap.add_argument('--scale', type=float, default=1.0, help='operator-to-robot motion scale')
    ap.add_argument('--max-speed', type=float, default=0.3, help='m/s attractor speed limit')
    ap.add_argument('--max-rot-speed', type=float, default=1.5, help='rad/s attractor limit')
    ap.add_argument('--max-accel', type=float, default=3.0,
                    help='m/s^2 attractor acceleration limit (bounds the feedforward torque rate)')
    ap.add_argument('--max-reach', type=float, default=0.25,
                    help='m, furthest a target may get from the startup TCP position')
    ap.add_argument('--abort-dq', type=float, default=2.5,
                    help='rad/s, per-joint abort speed (FR3 limits: 2.62 on joints 1-4)')
    ap.add_argument('--collision-torque', type=float, metavar='NM',
                    help='joint collision-reflex threshold, all joints (default: libfranka '
                         'example values, 12-20 Nm; its Cartesian impedance example uses 100)')
    ap.add_argument('--collision-force', type=float, metavar='N',
                    help='Cartesian collision-reflex threshold (default 20-25)')
    ap.add_argument('--status', type=float, default=0.0, metavar='S',
                    help='print a status line per arm every S seconds (default: only on changes)')
    ap.add_argument('--lease', type=float, default=0.3,
                    help='s; hold the arm if no operator sample arrives within this')
    ap.add_argument('--bitrate', type=int, default=6000, help='kbps for the ZED stereo track')
    ap.add_argument('--video-backend', default='gstreamer', help="adamo video backend "
                    "('default' for the SDK's own choice, 'gstreamer', or 'hw_pipeline')")
    # Use this host's Intel encoder: adamo 0.4.59 has a bitrate-unit bug in its NVENC path.
    ap.add_argument('--encoder', default='vah264enc', help="H.264 encoder ('default': SDK picks)")
    ap.add_argument('--trigger-axis', type=int, default=4, metavar='I')
    ap.add_argument('--grip-axis', type=int, default=5, metavar='I')
    ap.add_argument('--debug-input', action='store_true')
    args = ap.parse_args()
    if args.synthetic and not args.dry_run:
        sys.exit('--synthetic only runs with --dry-run')
    if not args.synthetic and not args.api_key:
        sys.exit('set ADAMO_API_KEY or pass --api-key (get one at operate.adamohq.com)')

    robot = None
    if not args.synthetic:
        try:
            robot = camera_robot(args)
        except Exception as error:
            sys.exit(f'Camera setup failed before starting the arms: {error}')

    ctx = mp.get_context('spawn')
    stop, msgs = ctx.Event(), ctx.Queue()
    base_opts = dict(dry_run=args.dry_run, lease=args.lease, max_speed=args.max_speed,
                     max_rot_speed=args.max_rot_speed, max_accel=args.max_accel,
                     abort_dq=args.abort_dq,
                     translational=args.translational, rotational=args.rotational,
                     scale=args.scale, max_reach=args.max_reach,
                     coll_tau=args.collision_torque, coll_f=args.collision_force)
    procs, arms = [], {}
    for side in args.arms:
        _, q_lo, q_hi = arm_frame_from_sim(side)
        cmd, state = ctx.Array('d', CMD_N), ctx.Array('d', STATE_N)
        p = ctx.Process(target=arm_process, name=f'arm-{side}', daemon=True,
                        args=(side, IPS[side], dict(base_opts, q_lo=q_lo, q_hi=q_hi),
                              cmd, state, stop, msgs))
        p.start()
        procs.append(p)
        arms[side] = (cmd, state)

    def drain():
        try:
            while True:
                side, text = msgs.get_nowait()
                print(f'[{side}] {text}')
                if robot is not None:
                    robot.log(f'{side}: {text}', level='warn' if 'ABORT' in text else 'info')
        except queue.Empty:
            pass

    deadline = time.monotonic() + 15.0
    while any(snapshot(st)[ST_STATUS] == S_START for _, st in arms.values()):
        drain()
        if time.monotonic() > deadline or not all(p.is_alive() for p in procs):
            break
        time.sleep(0.05)
    drain()
    if not all(snapshot(st)[ST_STATUS] in (S_RUN, S_DRY) for _, st in arms.values()):
        stop.set()
        if robot is not None:
            robot.close()
        sys.exit('an arm failed to start; nothing is moving')

    gcmds = {s: None for s in arms}
    if args.gripper and not args.dry_run:
        for side in arms:
            gcmds[side] = ctx.Array('d', [np.nan, 0.0])
            gp = ctx.Process(target=gripper_process, name=f'gripper-{side}', daemon=True,
                             args=(side, IPS[side], gcmds[side], stop, msgs, args.grasp_force))
            gp.start()
            procs.append(gp)

    lock = threading.Lock()
    channels = {s: RealArmChannel(s, base_from_arm_rotation(s), cmd, st, gcmds[s], base_opts)
                for s, (cmd, st) in arms.items()}

    quit_evt = threading.Event()

    def pusher():
        while not quit_evt.is_set():
            with lock:
                for ch in channels.values():
                    ch.push()
            time.sleep(0.005)
    threading.Thread(target=pusher, daemon=True).start()

    rumble_thread = None
    video_errors = []
    t_end = time.monotonic() + args.seconds if args.synthetic else float('inf')
    if args.synthetic:
        threading.Thread(target=synthetic_operator, args=(channels, lock, quit_evt, t_end),
                         daemon=True).start()
    else:
        wire_adamo(robot, args.name, channels, lock, args)
        rumble_thread = threading.Thread(target=stream_contact_rumble,
                                         args=(robot, arms, quit_evt), daemon=True)
        rumble_thread.start()
        robot.log(f'real Duo teleop up ({", ".join(arms)}), '
                  + ('DRY RUN: arms read-only' if args.dry_run else 'arms under torque control'))
        def stream_cameras():
            try:
                robot.run()
                if not quit_evt.is_set():
                    raise RuntimeError('camera pipeline ended unexpectedly')
            except Exception as error:
                message = f'Camera streaming failed: {error}. Stopping teleop; check cameras and network, then restart.'
                video_errors.append(message)
                print(message, file=sys.stderr, flush=True)
                stop.set()
                quit_evt.set()
                try:
                    robot.log(message, level='error')
                except Exception:
                    pass
        threading.Thread(target=stream_cameras, daemon=True).start()
        print(f'streaming as "{args.name}" -- open operate.adamohq.com. Ctrl-C to stop.')

    np.set_printoptions(precision=1, suppress=True)

    def status_line(s, ch):
        st = snapshot(ch.state)
        off = ch.R_ab.T @ (st[ST_APOS] - ch.home) * 1000          # base frame, mm
        err = np.linalg.norm(st[ST_APOS] - st[ST_POS]) * 1000
        fresh = (time.monotonic() - ch.last_rx) < args.lease
        return (f'[{s:5}] {STATUS_NAME[int(st[ST_STATUS])]:20} '
                f'clutch={"IN " if ch.engaged else "out"} op={"live" if fresh else "----"}  '
                f'attractor from start (base fwd/left/up) {off} mm  '
                f'|attractor-TCP| {err:5.1f} mm  max|tau| {st[ST_TAU]:4.1f} Nm  '
                f'ext|tau| {np.max(np.abs(st[ST_EXT])):4.2f} Nm  '
                f'rumble {contact_rumble_intensity(st, time.monotonic()):.0%}  '
                f'late {int(st[ST_LATE])}/{int(st[ST_CYC])}')

    # Report changes, not a heartbeat: operator link up/down, an arm stopping, or an arm
    # process that stopped updating (a dead loop must never look like a running one).
    arm_procs = dict(zip(arms, procs))                # the arm processes were started first
    seen = {}
    next_check = next_status = time.monotonic()
    try:
        while time.monotonic() < t_end and not quit_evt.is_set():
            tick = time.monotonic()
            drain()
            if tick >= next_check:
                next_check = tick + 0.1
                for s, ch in channels.items():
                    st = snapshot(ch.state)
                    status = int(st[ST_STATUS])
                    fresh = (tick - ch.last_rx) < args.lease
                    alive = status == S_STOPPED or (arm_procs[s].is_alive()
                                                    and tick - st[ST_BEAT] < 1.0)
                    prev = seen.get(s)
                    seen[s] = (status, fresh, alive)
                    if prev is None or prev == seen[s]:
                        continue
                    if prev[1] != fresh:
                        print(f'[{s}] operator {"live" if fresh else "link lost -- arm holding"}')
                    if prev[0] != status and status == S_STOPPED:
                        print(f'[{s}] arm STOPPED')
                    if prev[2] and not alive:
                        print(f'[{s}] arm process NOT RESPONDING -- arm is no longer controlled')
                if seen and all(v[0] == S_STOPPED or not v[2] for v in seen.values()):
                    print('all arms stopped')
                    break
            if args.status > 0 and tick >= next_status:
                next_status = tick + args.status
                for s, ch in channels.items():
                    print(status_line(s, ch))
            time.sleep(max(0.0, 1.0 / FPS - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        print('\nstopping...')
    finally:
        quit_evt.set()
        stop.set()
        if rumble_thread is not None:
            rumble_thread.join(timeout=1.0)
        for p in procs:
            p.join(timeout=3.0)
        drain()
        for s, ch in channels.items():
            print(status_line(s, ch))
        def final(s, st):
            status = int(snapshot(st)[ST_STATUS])
            died = status != S_STOPPED and not arm_procs[s].is_alive()
            return f'{s} {"DIED (never reported STOPPED)" if died else STATUS_NAME[status]}'
        print('arms: ' + ', '.join(final(s, st) for s, (_, st) in arms.items())
              + ('' if not any(p.is_alive() for p in procs) else '  (a worker is still alive!)'))
        if robot is not None:
            robot.close()
            print('adamo session closed')
    if video_errors:
        sys.exit(video_errors[0])


if __name__ == '__main__':
    code = 0
    try:
        main()
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    sys.stdout.flush()
    sys.stderr.flush()
    # Adamo's native threads abort ("FATAL: exception not rethrown") during interpreter
    # finalization -- after the arms have stopped and the session is closed. Skip finalization.
    os._exit(code)
