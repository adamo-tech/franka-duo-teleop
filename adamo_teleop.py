#!/usr/bin/env python3
"""Run the Mobile FR3 Duo sim as an Adamo robot: stereo video out, XR control in.

    pip install 'adamo[video]'
    export ADAMO_API_KEY=ak_...
    python adamo_teleop.py --name duo-sim

    python adamo_teleop.py --dry-run        # no network; synthetic operator, verifies the loop

What this does:

  * Publishes the ZED Mini stereo pair as ONE side-by-side track with ``stereo=True``, which is
    the format Adamo's viewer expects to split into left/right eyes for a headset.
  * Subscribes to XR controller poses via ``adamo.xr.subscribe_xr_control`` and drives each
    arm's Cartesian impedance attractor with them.
  * Uses relative (clutched) motion: the operator's absolute headset pose is meaningless in the
    robot's workspace, so squeezing the grip anchors the current controller pose to the current
    TCP pose and motion is measured from there. Releasing the grip freezes the arm and lets the
    operator reposition their hand -- the same pattern Quest2ROS uses.
  * Opens a local MuJoCo window alongside the stream (--no-view to suppress it).

Controls, per hand (WebXR xr-standard mapping):

    grip / squeeze      CLUTCH. Hold to control that arm; release to freeze it and reposition
                        your hand. Engaging anchors at wherever the arm already is, so there
                        is never a jump on engage.
    index trigger       That hand's gripper, proportional: the width follows the pull, so
                        released = open 80 mm, half-pulled = 40 mm, fully pulled = closed.
                        The analog pull has to arrive in Joy.axes to work -- XRJoy's `buttons`
                        are integers, so a trigger read from there can only be open or closed.
                        Defaults to axes[4]; --trigger-axis moves it, --debug-input finds it,
                        and the exit summary says which of the two paths you actually got.
    thumbstick (left)   Drive the base: push/pull = forward/back, left/right = strafe.
    thumbstick (right)  Left/right = yaw the base; push/pull = raise/lower the spine.
    thumbstick click    Re-home that arm and drop the clutch.

If the mapping comes through differently on your headset, run with --debug-input: it prints the
topic, axes and buttons of the first 40 XR samples so you can correct BTN_* / AX_* below.
  * Feeds the mobile base pose to ``robot.set_pose`` so the platform shows up on Adamo's map,
    and mirrors joint state and warnings to ``robot.log``.

Frames. XR poses arrive in the WebXR convention (x right, y UP, z BACKWARD) and are mapped into
the ROBOT BASE frame (x forward, y left, z up) by `xr_to_base_*`. Deliberately not the ZED
optical frame: that one is pitched 41 deg down by the head bracket, so a straight-forward push
of the controller would drive the hand forward AND down -- 0.20 m of hand motion becoming
0.151 m forward and 0.131 m down. The base frame keeps the robot's heading without the pitch,
and the planar base has no roll or pitch DoF, so it stays gravity-aligned while driving.

Safety. Motion requires a live deadman: no XR sample newer than `--lease` seconds means the
attractor is frozen where it is. That is on top of `MobileDuo.STALE_S`, which holds the
attractor inside the control loop if the operator link goes quiet.

STATUS: the sim, the frame math, the clutch gating and the trigger->gripper mapping are
exercised by `--dry-run` and by a direct clutch test. The Adamo transport calls are written
against the installed SDK's signatures (adamo 0.4.58) but have NOT been run against the live
service from here -- that needs an API key.
"""
from __future__ import annotations

import argparse
import os
import threading
import time

import mujoco
import numpy as np

from duo_control import ImpedanceGains, MobileDuo, mat_to_quat, quat_mul, quat_to_mat

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, 'mobile_fr3_duo.xml')

HOME_BASE = {}   # home target per arm, in the robot base frame; filled when DuoSim is built
EYE_W, EYE_H = 640, 360          # per eye; the published frame is 2*EYE_W x EYE_H
FPS = 30


# WebXR is x right, y UP, z BACKWARD. The robot base frame is x FORWARD, y LEFT, z UP.
# Rows of C give each base axis as a combination of XR axes: forward = -z, left = -x, up = +y.
#
# Note this maps into the BASE frame, not the ZED optical frame. The optical frame is pitched
# 41 deg down by the head bracket, so feeding gravity-aligned hand motion into it tilts every
# command by 41 deg: a 0.20 m straight-forward push becomes 0.151 m forward and 0.131 m down.
# The base frame carries the robot's heading without the pitch, and the planar base has no roll
# or pitch DoF, so it stays gravity-aligned as the robot drives.
C_XR_TO_BASE = np.array([[0.0, 0.0, -1.0],
                         [-1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]])


def xr_to_base_pos(p):
    return C_XR_TO_BASE @ np.asarray(p, float)


def xr_to_base_quat(q_xyzw):
    """Change of basis for orientation: R_base = C R_xr C^T, returned as (w, x, y, z)."""
    x, y, z, w = q_xyzw
    R = quat_to_mat(np.array([w, x, y, z], float))
    return mat_to_quat(C_XR_TO_BASE @ R @ C_XR_TO_BASE.T)


# WebXR "xr-standard" gamepad mapping, which is what the browser reports for a Quest
# controller and what Adamo forwards in its XR Joy messages.
BTN_TRIGGER, BTN_GRIP, BTN_THUMB = 0, 1, 3      # index trigger, side grip, thumbstick click
AX_THUMB = (2, 3)                               # thumbstick x, y

# Where the ANALOG pull of the trigger and squeeze live in Joy.axes.
#
# This is the part that decides whether a half-pulled trigger half-closes the gripper. In
# WebXR's xr-standard mapping the analog pull is `GamepadButton.value` on buttons 0 and 1 --
# but a sensor_msgs/Joy has no room for it: `adamo.xr.XRJoy` types `buttons` as
# `Tuple[int, ...]`, so anything arriving through the button array is already rounded to 0/1
# and no amount of scaling downstream can recover the pull. Bridges that want to keep the
# analog value append it to `axes` after the four stick axes, which is where these indices
# point. axes[0..1] are the *touchpad*, not the trigger: reading the trigger there (as this
# did) gets you a binary gripper and, on a controller with a touchpad, phantom pulls.
#
# If your bridge lays them out differently, --trigger-axis / --grip-axis move them and
# --debug-input prints the raw arrays. -1 means "no analog axis", i.e. fall back to the button.
AX_TRIGGER, AX_GRIP = 4, 5

# Triggers rest a little off zero and often stop short of full scale, so rescale the usable
# span to 0..1. Without this the gripper never quite opens or never quite closes.
TRIGGER_DEADZONE, TRIGGER_FULL = 0.04, 0.95

GRIPPER_OPEN = 0.08                             # m, Franka Hand full opening


def _analog(axes, buttons, axis_idx, button_idx):
    """Read a trigger as ``(value in 0..1, came_from_an_analog_axis)``.

    The axis wins whenever the stream carries one, *including when it reads exactly zero* --
    a released analog trigger is a real 0.0, not a missing sample, and falling through to the
    button on it is how a working analog axis still ends up feeling binary at the extremes.
    """
    if axis_idx is not None and axis_idx >= 0 and len(axes) > axis_idx:
        raw = min(1.0, abs(float(axes[axis_idx])))
        span = max(TRIGGER_FULL - TRIGGER_DEADZONE, 1e-6)
        return float(np.clip((raw - TRIGGER_DEADZONE) / span, 0.0, 1.0)), True
    if len(buttons) > button_idx:
        return float(min(1.0, abs(buttons[button_idx]))), False
    return 0.0, False


class ArmChannel:
    """Per-arm clutched mapping from operator controller pose to a camera-frame target.

    The clutch is explicit and is driven ONLY by the grip button. Poses arriving while
    disengaged are remembered but do not move the arm -- that is the whole point of a clutch,
    and auto-engaging on the first pose (as an earlier version did) silently defeats it.
    """

    def __init__(self, duo, prefix, scale=1.0):
        self.duo, self.prefix, self.scale = duo, prefix, scale
        self.engaged = False
        self.anchor_ctrl = None
        self.anchor_tgt = None
        self.target = HOME_BASE[prefix].copy()
        self.quat = None
        self.last_pose = None          # most recent controller pose, engaged or not
        self.last_rx = 0.0

    def set_clutch(self, engaged):
        """Grip pressed -> anchor here and start following. Released -> freeze."""
        if engaged and not self.engaged:
            if self.last_pose is None or self.quat is None:
                return                 # no controller pose yet; nothing to anchor to
            self.anchor_ctrl = self.last_pose
            self.anchor_tgt = (self.target.copy(), self.quat.copy())
            self.engaged = True
        elif not engaged and self.engaged:
            self.engaged = False

    def release(self):
        self.engaged = False

    def update(self, ctrl_pos, ctrl_quat):
        """Record the controller pose; move the target only while clutched in."""
        self.last_rx = time.monotonic()
        self.last_pose = (np.asarray(ctrl_pos, float), np.asarray(ctrl_quat, float))
        if not self.engaged or self.anchor_ctrl is None:
            return
        d = (self.last_pose[0] - self.anchor_ctrl[0]) * self.scale
        self.target = self.anchor_tgt[0] + d

        # Orientation delta, taken in the BASE frame and applied on the left:
        #     dq = q_ctrl * q_anchor_ctrl^-1        (rotation as seen in the base frame)
        #     q_target = dq * q_anchor_tgt
        # Composing on the right instead (q_anchor_tgt * q_anchor_ctrl^-1 * q_ctrl) applies the
        # delta in the TARGET HAND's body frame, and the home hand frame has its approach axis
        # along world-forward -- so an operator yaw about world-up came out as a roll about the
        # gripper's approach axis. Left-composition keeps operator axes and robot axes aligned.
        inv_anchor = self.anchor_ctrl[1] * np.array([1.0, -1.0, -1.0, -1.0])
        dq = quat_mul(self.last_pose[1], inv_anchor)
        self.quat = quat_mul(dq, self.anchor_tgt[1])

    def apply(self, lease_s):
        """Push the current target into the arm's impedance attractor."""
        if self.quat is None:
            return
        fresh = (time.monotonic() - self.last_rx) < lease_s
        if not (self.engaged and fresh):
            return                                  # frozen: attractor stays where it is
        self.duo.arms[self.prefix].set_target(
            *self.duo.base_to_world(self.target, self.quat))


class DuoSim:
    """The simulation, stepped at 1 kHz on its own thread, paced to wall-clock."""

    def __init__(self, gains=None, realtime=True):
        self.duo = MobileDuo(XML, gains or ImpedanceGains(
            translational=900, rotational=60, damping_design='inertia',
            velocity_feedforward=True))
        self.realtime = realtime
        self.lock = threading.Lock()
        self.stop = threading.Event()
        for p in ('left', 'right'):
            HOME_BASE[p] = self.duo.world_to_base(*self.duo.home_world(p))[0]
        # Per arm: the home pose is mirror-symmetric, so left and right have DIFFERENT
        # orientations. Sharing one quaternion leaves the right ghost gripper (and the right
        # arm's attractor) at the left arm's attitude.
        self.home_quat = {p: self.duo.world_to_base(*self.duo.home_world(p))[1]
                          for p in ('left', 'right')}
        for p in ('left', 'right'):
            self.duo.arms[p].set_target(*self.duo.home_world(p))
        self.channels = {p: ArmChannel(self.duo, p) for p in ('left', 'right')}
        for p, ch in self.channels.items():
            ch.quat = self.home_quat[p].copy()
        self.base_cmd = np.zeros(3)
        self.spine_cmd = self.duo.spine.height
        self.trigger = {p: 0.0 for p in ('left', 'right')}
        # False until a sample proves an analog trigger axis exists; reported at exit,
        # because 'my trigger is on/off' is otherwise indistinguishable from a stiff hand.
        self.trigger_analog = {p: False for p in ('left', 'right')}

    def settle(self, seconds=2.0):
        for _ in range(int(seconds / self.duo.dt)):
            self.duo.step()

    def loop(self, lease_s):
        next_t = time.perf_counter()
        while not self.stop.is_set():
            with self.lock:
                for ch in self.channels.values():
                    ch.apply(lease_s)
                self.duo.base.set_twist(*self.base_cmd)
                self.duo.spine.move_absolute(self.spine_cmd)
                self.duo.step()
            if self.realtime:
                next_t += self.duo.dt
                slip = next_t - time.perf_counter()
                if slip > 0:
                    time.sleep(slip)
                elif slip < -0.05:
                    next_t = time.perf_counter()    # fell behind; resynchronise

    def sync_markers(self):
        """Park the translucent ghost grippers on each arm's commanded target pose."""
        for p, ch in self.channels.items():
            if ch.quat is None:
                continue
            mid = self.duo.m.body_mocapid[
                mujoco.mj_name2id(self.duo.m, mujoco.mjtObj.mjOBJ_BODY, f'target_{p}')]
            pos, quat = self.duo.base_to_world(ch.target, ch.quat)
            self.duo.d.mocap_pos[mid] = pos
            self.duo.d.mocap_quat[mid] = quat

    def render_stereo(self, renderer_l, renderer_r, out):
        """Render both eyes into one side-by-side RGB frame."""
        with self.lock:
            self.sync_markers()
            renderer_l.update_scene(self.duo.d, camera='zed_left')
            renderer_r.update_scene(self.duo.d, camera='zed_right')
        out[:, :EYE_W] = renderer_l.render()
        out[:, EYE_W:] = renderer_r.render()
        return out


def wire_control(sim, robot, name, lease_s, debug=False):
    """Subscribe to XR controller poses and buttons from the Adamo operator."""
    from adamo import xr

    seen = {'n': 0}

    def side_of(topic):
        t = topic.lower()
        return 'left' if 'left' in t else ('right' if 'right' in t else None)

    def on_xr(sample):
        side = side_of(sample.topic)
        msg = sample.message
        if debug and seen['n'] < 40:
            seen['n'] += 1
            kind = type(msg).__name__
            extra = (f'axes={list(getattr(msg, "axes", []))} '
                     f'buttons={list(getattr(msg, "buttons", []))}'
                     if kind == 'XRJoy' else '')
            print(f'[xr] topic={sample.topic!r} side={side} {kind} {extra}')
        if isinstance(msg, xr.PoseStamped):
            if side is None:
                return
            with sim.lock:
                sim.channels[side].update(xr_to_base_pos(msg.position.as_tuple()),
                                          xr_to_base_quat(msg.orientation.as_tuple()))
        elif isinstance(msg, xr.XRJoy):
            handle_xr_buttons(sim, side, list(msg.axes), list(msg.buttons))

    # max_age_seconds drops samples whose ROS source timestamp is already stale, so a burst
    # arriving after a network stall does not get replayed into the arm.
    xr.subscribe_xr_control(robot.session, name, on_xr, max_age_seconds=0.25)

    # A plain gamepad, if one is connected, drives the platform only.
    @robot.on(name, 'control/joy', decode='control', priority=250)
    def on_joy(msg):                                     # noqa: ARG001 (registered by decorator)
        handle_pad(sim, list(getattr(msg, 'axes', [])), list(getattr(msg, 'buttons', [])))


def handle_xr_buttons(sim, side, axes, buttons):
    """VR controller buttons for one hand.

      grip (squeeze)  -> CLUTCH: hold to engage control of that arm, release to freeze it
      index trigger   -> that hand's gripper, analog: released = open, squeezed = closed
      thumbstick      -> drives the platform (left hand: translate, right hand: yaw + spine)
    """
    if side is None:
        return
    with sim.lock:
        ch = sim.channels[side]
        grip, _ = _analog(axes, buttons, AX_GRIP, BTN_GRIP)
        ch.set_clutch(grip > 0.5)          # the clutch only ever wanted a boolean
        trigger, analog = _analog(axes, buttons, AX_TRIGGER, BTN_TRIGGER)
        # Proportional: trigger 0 -> 80 mm open, 0.5 -> 40 mm, 1 -> closed. The Gripper's own
        # position servo does the rest, so a partial pull is a partial width, not a partial
        # squeeze force.
        sim.duo.grippers[side].set_width(GRIPPER_OPEN * (1.0 - trigger))
        sim.trigger[side] = trigger
        sim.trigger_analog[side] = analog

        def dz(v, d=0.15):
            return 0.0 if abs(v) < d else float(np.sign(v) * (abs(v) - d) / (1 - d))

        tx = axes[AX_THUMB[0]] if len(axes) > AX_THUMB[0] else 0.0
        ty = axes[AX_THUMB[1]] if len(axes) > AX_THUMB[1] else 0.0
        if side == 'left':                       # translate the base
            sim.base_cmd[0] = -dz(ty) * 0.6
            sim.base_cmd[1] = -dz(tx) * 0.4
        else:                                    # yaw the base, raise/lower the spine
            sim.base_cmd[2] = -dz(tx) * 0.8
            sim.spine_cmd = float(np.clip(sim.spine_cmd - dz(ty) * 0.004, 0.0, 0.85))
        if len(buttons) > BTN_THUMB and buttons[BTN_THUMB]:   # thumbstick click: re-home
            ch.target = HOME_BASE[side].copy()
            ch.quat = sim.home_quat[side].copy()
            ch.release()


def handle_pad(sim, axes, buttons):
    """Optional flat gamepad: platform only, so it never fights the VR controllers."""
    axes = list(axes) + [0.0] * 6
    buttons = list(buttons) + [0] * 21

    def dz(v, d=0.12):
        return 0.0 if abs(v) < d else float(np.sign(v) * (abs(v) - d) / (1 - d))

    with sim.lock:
        sim.base_cmd = np.array([-dz(axes[1]) * 0.6, -dz(axes[0]) * 0.4, -dz(axes[2]) * 0.8])
        if buttons[4] or buttons[5]:
            sim.spine_cmd = float(np.clip(
                sim.spine_cmd + (0.004 if buttons[5] else -0.004), 0.0, 0.85))


def synthetic_operator(sim, lease_s, seconds):
    """--dry-run: drive the channels locally so the whole pipeline can be exercised offline.

    Emits the same shape of input a Quest does: XR-frame poses plus xr-standard button arrays,
    so the clutch, the trigger->gripper mapping and the frame conversion are all really tested.
    """
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds and not sim.stop.is_set():
        t = time.monotonic() - t0
        for i, p in enumerate(sim.channels):
            ph = i * np.pi / 2
            # A WebXR-frame hand motion, so the frame conversion is genuinely exercised.
            pos_xr = np.array([0.10 * np.sin(2 * np.pi * 0.3 * t + ph),
                               0.07 * np.sin(2 * np.pi * 0.5 * t + ph),    # up in WebXR
                               -0.06 * np.sin(2 * np.pi * 0.2 * t + ph)])  # forward in WebXR
            with sim.lock:
                sim.channels[p].update(xr_to_base_pos(pos_xr),
                                       np.array([1.0, 0.0, 0.0, 0.0]))
            # Clutch in after 1 s and stay engaged; squeeze the trigger on a slow cycle.
            grip = 1.0 if t > 1.0 else 0.0
            trig = 0.5 * (1.0 - np.cos(2 * np.pi * 0.25 * t + ph))
            # xr-standard order, with the analog button values appended after the stick axes,
            # so the dry run exercises the same path a real analog trigger takes. buttons stay
            # integers, as XRJoy declares them -- that is exactly why the axes carry the pull.
            axes = [0.0, 0.0, 0.0, 0.0, trig, grip]
            buttons = [int(trig > 0.5), int(grip > 0.5), 0, 0]
            handle_xr_buttons(sim, p, axes, buttons)
        time.sleep(1.0 / 72.0)


def main():
    global AX_TRIGGER, AX_GRIP           # --trigger-axis / --grip-axis retarget the analog read
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='duo-sim', help='robot name registered with Adamo')
    ap.add_argument('--api-key', default=os.environ.get('ADAMO_API_KEY'))
    ap.add_argument('--bitrate', type=int, default=6000, help='kbps for the stereo track')
    ap.add_argument('--lease', type=float, default=0.5,
                    help='freeze the arms if no operator sample arrives within this many seconds')
    ap.add_argument('--dry-run', action='store_true',
                    help='no network: synthetic operator, frames rendered and discarded')
    ap.add_argument('--seconds', type=float, default=20.0, help='dry-run duration')
    ap.add_argument('--no-view', dest='view', action='store_false',
                    help='do not open the local MuJoCo window')
    ap.add_argument('--trigger-axis', type=int, default=AX_TRIGGER, metavar='I',
                    help='Joy.axes index carrying the analog trigger pull (default %(default)s; '
                         '-1 to use the on/off button instead). Run --debug-input to find it.')
    ap.add_argument('--grip-axis', type=int, default=AX_GRIP, metavar='I',
                    help='Joy.axes index carrying the analog squeeze (default %(default)s; '
                         '-1 to use the on/off button instead)')
    ap.add_argument('--debug-input', action='store_true',
                    help='print the first XR samples (topic, axes, buttons) to check the mapping')
    args = ap.parse_args()
    AX_TRIGGER, AX_GRIP = args.trigger_axis, args.grip_axis

    sim = DuoSim()
    sim.settle()
    print(f'sim ready: {sim.duo.m.nv} DoF, camera at '
          f'{np.round(sim.duo.camera_pose()[0], 3)}')

    rl = mujoco.Renderer(sim.duo.m, EYE_H, EYE_W)
    rr = mujoco.Renderer(sim.duo.m, EYE_H, EYE_W)
    frame = np.zeros((EYE_H, 2 * EYE_W, 3), dtype=np.uint8)

    thread = threading.Thread(target=sim.loop, args=(args.lease,), daemon=True)
    thread.start()

    track = None
    robot = None
    if not args.dry_run:
        if not args.api_key:
            raise SystemExit('set ADAMO_API_KEY or pass --api-key (get one at operate.adamohq.com)')
        import adamo
        robot = adamo.Robot(api_key=args.api_key, name=args.name)
        # One side-by-side track, flagged stereo so the viewer splits it per eye. Raw RGB frames
        # go straight into the Rust encoder; nothing touches a v4l2 device.
        track = robot.video('stereo', width=2 * EYE_W, height=EYE_H, pixel_format='RGB',
                            codec='h264', fps=FPS, bitrate_kbps=args.bitrate, stereo=True)
        wire_control(sim, robot, args.name, args.lease, debug=args.debug_input)
        robot.log(f'mobile_fr3_duo sim up: stereo {2 * EYE_W}x{EYE_H} @ {FPS}fps')
        threading.Thread(target=robot.run, daemon=True).start()
        print(f'streaming as "{args.name}" -- open operate.adamohq.com. Ctrl-C to stop.')
    else:
        threading.Thread(target=synthetic_operator,
                         args=(sim, args.lease, args.seconds), daemon=True).start()

    # Local window. Bound to the main thread, which is where MuJoCo wants its viewer, so the
    # video render/send loop runs inside the viewer loop rather than the other way round.
    viewer = None
    if args.view:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(sim.duo.m, sim.duo.d,
                                          show_left_ui=False, show_right_ui=False)
        viewer.cam.lookat[:] = sim.duo.d.xpos[mujoco.mj_name2id(
            sim.duo.m, mujoco.mjtObj.mjOBJ_BODY, 'fr3_duo_mount_origin')]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.6, 150, -18
        print('local viewer open -- close the window or Ctrl-C to stop')

    n, t0, deadline = 0, time.perf_counter(), time.perf_counter() + args.seconds
    try:
        while True:
            tick = time.perf_counter()
            sim.render_stereo(rl, rr, frame)
            n += 1
            if track is not None:
                track.send(frame)
                if n % 6 == 0:
                    x, y, yaw = sim.duo.base.pose
                    robot.set_pose(float(x), float(y), heading=float(yaw))
            if viewer is not None:
                if not viewer.is_running():
                    break
                with sim.lock:
                    viewer.sync()
            if args.dry_run and tick > deadline:
                break
            time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        sim.stop.set()
        if viewer is not None:
            viewer.close()
        if robot is not None:
            robot.close()

    if args.dry_run:
        errs = {p: np.linalg.norm(sim.duo.world_to_base(*sim.duo.arms[p].tcp_pose())[0]
                                  - sim.channels[p].target) for p in sim.channels}
        print(f'dry run: {n} stereo frames ({n / (time.perf_counter() - t0):.1f} fps), '
              f'frame {frame.shape} {frame.dtype}, sim time {sim.duo.t:.1f} s')
        print('  clutch engaged: '
              + '  '.join(f'{p}={sim.channels[p].engaged}' for p in sim.channels))
        print('  final base-frame tracking error: '
              + '  '.join(f'{p} {e * 1000:.1f} mm' for p, e in errs.items()))
        print('  gripper width: ' + '  '.join(
            f'{p} {GRIPPER_OPEN * (1 - sim.trigger[p]) * 1000:.0f} mm '
            f'(trigger {sim.trigger[p]:.2f})' for p in sim.channels))
        if any(sim.trigger_analog.values()):
            print(f'  trigger: analog, from axes[{AX_TRIGGER}] -- partial pull, partial close')
        else:
            print(f'  trigger: ON/OFF, from buttons[{BTN_TRIGGER}] -- nothing analog arrived on '
                  f'axes[{AX_TRIGGER}], so the gripper can only be fully open or fully closed.\n'
                  f'           Run --debug-input to see the raw axes, then point --trigger-axis '
                  f'at the one that moves with the pull.')
        print(f'  base at {np.round(sim.duo.base.pose, 3)}, spine {sim.duo.spine.height:.3f} m')


if __name__ == '__main__':
    main()
