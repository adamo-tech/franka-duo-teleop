#!/usr/bin/env python3
"""Control interfaces for the Mobile FR3 Duo MuJoCo model.

Each class mirrors how the corresponding subsystem is actually commanded on hardware:

  CartesianImpedance  <-> your own libfranka torque controller (franka::Torques at 1 kHz).
                          Same control law as libfranka's examples/cartesian_impedance_control.cpp:
                          tau = J^T (-K e - D J dq) + coriolis, with a null-space posture term.
  Spine               <-> franka_spine_msgs/MoveAbsolute action server (non-realtime, rate limited).
  BaseTwist           <-> the TMR hardware component's cartesian_velocity GPIO (vx, vy, wz).

Gravity note: libfranka's franka::Torques are documented as "without gravity and friction" --
the control box adds gravity internally, and the example adds model.coriolis() by hand. MuJoCo's
data.qfrc_bias is exactly coriolis + gravity, so applying tau_task + qfrc_bias to the arm DoFs
reproduces the hardware torque path term for term.
"""
from __future__ import annotations

import dataclasses
from collections import deque

import mujoco
import numpy as np

ARM = 'fr3v2_1'
ARM_PREFIXES = ('left', 'right')
# FR3 joint torque limits, from the URDF <limit effort=...>; also the actuator ctrlrange.
TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

# ---------------------------------------------------------------------------
# Home pose, given explicitly as the TCP pose of each arm in the ROBOT BASE frame
# (x forward, y left, z up, origin at base_link), together with the joint configuration that
# realises it and the spine height it assumes.
#
# Base-frame rather than camera-frame: the ZED optical frame is pitched 41 deg down by the head
# bracket, and the left lens sits 14 mm off the robot centreline, so camera-frame coordinates
# make a symmetric pose look asymmetric and tilt every operator command.
HOME_SPINE = 0.376934        # m, spine extension that puts the TCPs at z = 0.7105

HOME_TCP_BASE = {
    'left':  (np.array([1.0537, 0.0892, 0.7105]),
              np.array([-0.1549, -0.4507, 0.7232, -0.4998])),
    'right': (np.array([1.0537, -0.0892, 0.7105]),
              np.array([-0.1549, 0.4507, 0.7232, 0.4998])),
}

# The joint configuration that realises the above. Exactly mirror-symmetric apart from the
# wrist roll, which differs by 90 deg because the two hands are rolled differently about a
# common approach axis.
HOME_JOINTS = {
    'left':  np.radians([43.63, 48.40, 10.48, -89.37, 121.90, 215.83, 60.61]),
    'right': np.radians([-43.63, 48.40, -10.48, -89.37, -121.90, 215.83, 29.39]),
}

def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def mat_to_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def quat_conj(q):
    out = np.zeros(4)
    mujoco.mju_negQuat(out, q)
    return out


def pose_to_mat(pos, quat):
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(quat)
    T[:3, 3] = pos
    return T


def mat_to_pose(T):
    return T[:3, 3].copy(), mat_to_quat(T[:3, :3])


# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ImpedanceGains:
    """Cartesian impedance gains, in the robot base frame.

    libfranka's shipped example uses 150 N/m and 10 Nm/rad, which is deliberately soft so you
    can push the arm around by hand. Teleoperation wants considerably stiffer; the defaults
    here are a starting point and `translational`/`rotational` are the snappiness dial.

    damping_design:
      'sqrt'    D = 2*zeta*sqrt(K), exactly as in libfranka's cartesian_impedance_control.cpp.
                That formula implicitly assumes unit apparent mass, so on a real arm -- whose
                task-space inertia is a few kg, dominated by reflected rotor inertia -- it
                under-damps and the step response rings.
      'inertia' D = 2*zeta*sqrt(Lambda)*sqrt(K) with Lambda = (J M^-1 J^T)^-1 the task-space
                inertia. Costs one 7x7 inverse and one 6x6 eigendecomposition per cycle and
                actually delivers the damping ratio you asked for.

    velocity_feedforward: add D*v_d, using the target velocity differentiated from the operator
    stream. The damping term otherwise fights the motion you are commanding, and cancelling it
    removes a large share of the arm's own contribution to end-to-end lag.
    """
    translational: float = 600.0     # N/m
    rotational: float = 40.0         # Nm/rad
    damping_ratio: float = 1.0
    nullspace_stiffness: float = 8.0
    max_offset: float = 0.06         # m; attractor-to-actual clamp, bounds peak force at K*d
    damping_design: str = 'sqrt'
    velocity_feedforward: bool = False
    joint_damping: float = 2.0        # Nm.s/rad, always on
    limit_margin: float = 0.20        # rad from a joint limit where the barrier engages
    limit_stiffness: float = 60.0     # Nm/rad barrier

    def stiffness(self):
        return np.diag([self.translational] * 3 + [self.rotational] * 3)

    def sqrt_damping(self):
        return np.diag([2.0 * self.damping_ratio * np.sqrt(self.translational)] * 3
                       + [2.0 * self.damping_ratio * np.sqrt(self.rotational)] * 3)


def _sqrtm_sym(A):
    """Matrix square root of a symmetric positive-(semi)definite matrix."""
    w, V = np.linalg.eigh((A + A.T) / 2.0)
    return V @ np.diag(np.sqrt(np.maximum(w, 0.0))) @ V.T


class CartesianImpedance:
    """One arm's task-space controller. Setpoint is a pose; output is 7 joint torques."""

    def __init__(self, model, data, prefix, gains: ImpedanceGains | None = None):
        self.m, self.d, self.prefix = model, data, prefix
        self.gains = gains or ImpedanceGains()
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f'{prefix}_{ARM}_hand_tcp')
        self.jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                          f'{prefix}_{ARM}_joint{i}') for i in range(1, 8)]
        self.dof = np.array([model.jnt_dofadr[j] for j in self.jnt_ids])
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jnt_ids])
        self.act = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                               f'{prefix}_{ARM}_joint{i}') for i in range(1, 8)])
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))
        self.q_lo = model.jnt_range[self.jnt_ids][:, 0].copy()
        self.q_hi = model.jnt_range[self.jnt_ids][:, 1].copy()
        self.q_null = self.q.copy()
        self.pos_d, self.quat_d = self.tcp_pose()
        self.vel_d = np.zeros(6)
        self.last_error = np.zeros(6)

    # -- state ------------------------------------------------------------
    @property
    def q(self):
        return self.d.qpos[self.qadr]

    @property
    def dq(self):
        return self.d.qvel[self.dof]

    def tcp_pose(self):
        return (self.d.site_xpos[self.site].copy(),
                mat_to_quat(self.d.site_xmat[self.site].reshape(3, 3)))

    def jacobian(self):
        mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, self.site)
        return np.vstack([self._jp[:, self.dof], self._jr[:, self.dof]])

    # -- setpoint ---------------------------------------------------------
    def set_target(self, pos, quat, vel=None):
        """Move the attractor. There is no trajectory and no goal state: overwriting this
        mid-motion is the normal case, exactly as on hardware.

        `vel` is an optional 6-vector task-space target velocity in the base frame, used only
        when gains.velocity_feedforward is set.
        """
        self.pos_d, self.quat_d = np.asarray(pos, float), np.asarray(quat, float)
        self.vel_d = np.zeros(6) if vel is None else np.asarray(vel, float)

    def reset_target_to_current(self):
        self.pos_d, self.quat_d = self.tcp_pose()
        self.vel_d = np.zeros(6)

    def task_inertia(self, J):
        """Lambda = (J M^-1 J^T)^-1, the apparent inertia at the TCP, and M^-1 alongside it."""
        M = np.zeros((self.m.nv, self.m.nv))
        mujoco.mj_fullM(self.m, self.d, M)
        Ma = M[np.ix_(self.dof, self.dof)]
        Minv = np.linalg.inv(Ma + 1e-9 * np.eye(7))
        # Damped inverse: near a singularity J M^-1 J^T loses rank and Lambda blows up.
        return np.linalg.inv(J @ Minv @ J.T + 1e-4 * np.eye(6)), Minv

    def joint_limit_torque(self):
        """Repulsion that switches on inside `limit_margin` of a joint limit.

        The posture term alone does not prevent a limit: it pulls toward one posture, and a
        target outside the workspace can pull harder. This is the explicit barrier.
        """
        margin = self.gains.limit_margin
        k = self.gains.limit_stiffness
        q = self.q
        return k * (np.maximum(0.0, (self.q_lo + margin) - q)
                    - np.maximum(0.0, q - (self.q_hi - margin)))

    # -- control law ------------------------------------------------------
    def compute(self):
        pos, quat = self.tcp_pose()
        K = self.gains.stiffness()
        J = self.jacobian()
        Lam, Minv = self.task_inertia(J)
        if self.gains.damping_design == 'inertia':
            sqrtK = np.diag(np.sqrt(np.diag(K)))
            DK = _sqrtm_sym(Lam) @ sqrtK
            D = self.gains.damping_ratio * (DK + DK.T)     # 2*zeta*sqrt(Lambda)*sqrt(K)
        else:
            D = self.gains.sqrt_damping()

        delta = self.pos_d - pos
        n = np.linalg.norm(delta)
        if n > self.gains.max_offset:          # clamp the spring extension, not the target
            delta = delta / n * self.gains.max_offset

        # Orientation error, same construction as the libfranka example: the "difference"
        # quaternion, sign-corrected for the double cover, rotated into the base frame.
        q_cur = quat.copy()
        if float(q_cur @ self.quat_d) < 0.0:
            q_cur = -q_cur
        q_err = quat_mul(quat_conj(q_cur), self.quat_d)
        err_rot = quat_to_mat(q_cur) @ q_err[1:]

        error = np.concatenate([-delta, -err_rot])
        self.last_error = error

        v = J @ self.dq
        v_ref = self.vel_d if self.gains.velocity_feedforward else np.zeros(6)
        tau_task = J.T @ (-K @ error + D @ (v_ref - v))

        # Null-space terms: posture (the torque-control counterpart of commanding elbow_command
        # on hardware) plus an explicit joint-limit barrier.
        #
        # The projector is the dynamically consistent one, N = I - J^T (M^-1 J^T Lambda)^T,
        # built from the *damped* Lambda above. A plain pinv(J^T, rcond=...) is not safe here:
        # driving the target past the reach limit parks the arm at a singularity where the
        # smallest singular value sits right around any fixed rcond, so the pseudo-inverse
        # produces enormous entries, the null-space torque saturates the joints, and the arm
        # stays stuck even after the target returns inside the workspace.
        Jbar = Minv @ J.T @ Lam                      # dynamically consistent inverse (7x6)
        null_proj = np.eye(7) - J.T @ Jbar.T
        ks = self.gains.nullspace_stiffness
        tau_null = null_proj @ (ks * (self.q_null - self.q) - 2.0 * np.sqrt(ks) * self.dq
                                + self.joint_limit_torque())

        # Small always-on joint damping. Costs a little bandwidth but bleeds energy in
        # configurations where the task-space damping term has no authority, which is exactly
        # what lets the arm walk back out of a singular pose.
        tau_joint_damp = -self.gains.joint_damping * self.dq

        tau = tau_task + tau_null + tau_joint_damp
        return np.clip(tau, -TAU_MAX, TAU_MAX)

    def apply(self):
        """Write torques for this arm. qfrc_bias supplies coriolis (which libfranka makes you
        add) plus gravity (which the control box adds internally)."""
        self.d.ctrl[self.act] = self.compute() + self.d.qfrc_bias[self.dof]


class Spine:
    """Prismatic torso lift, 0 - 0.85 m.

    On hardware this is franka_spine_server's MoveAbsolute action, outside the 1 kHz loop, so
    the model here is a rate-limited position setpoint rather than a streaming interface. The
    0.1 m/s cap is the URDF velocity limit.
    """
    V_MAX = 0.1
    RANGE = (0.0, 0.85)

    def __init__(self, model, data):
        self.m, self.d = model, data
        self.act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                     'franka_spine_vertical_joint')
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'franka_spine_vertical_joint')
        self.qadr = model.jnt_qposadr[jid]
        self.target = float(data.qpos[self.qadr])

    @property
    def height(self):
        return float(self.d.qpos[self.qadr])

    def move_absolute(self, z):
        self.target = float(np.clip(z, *self.RANGE))

    def apply(self, dt):
        cmd = self.d.ctrl[self.act]
        self.d.ctrl[self.act] = cmd + np.clip(self.target - cmd, -self.V_MAX * dt, self.V_MAX * dt)


class BaseTwist:
    """Omnidirectional base, commanded as a twist in the base frame.

    Matches the TMR's cartesian_velocity command interface (vx, vy, vz, wx, wy, wz); only the
    three planar components are meaningful for a ground vehicle. Top speed 1.75 m/s (datasheet).
    """
    V_MAX, W_MAX = 1.75, 2.0

    def __init__(self, model, data):
        self.d = data
        self.act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                    for n in ('base_x', 'base_y', 'base_yaw')]
        jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
               for n in ('base_x', 'base_y', 'base_yaw')]
        self.qadr = [model.jnt_qposadr[j] for j in jid]

    @property
    def pose(self):
        """(x, y, yaw) of the base in the world frame."""
        return self.d.qpos[self.qadr].copy()

    def set_twist(self, vx, vy, wz):
        yaw = self.d.qpos[self.qadr[2]]
        c, s = np.cos(yaw), np.sin(yaw)
        v = np.clip([vx, vy], -self.V_MAX, self.V_MAX)
        # Commanded twist is body-frame; the planar joints are world-frame.
        self.d.ctrl[self.act[0]] = c * v[0] - s * v[1]
        self.d.ctrl[self.act[1]] = s * v[0] + c * v[1]
        self.d.ctrl[self.act[2]] = np.clip(wz, -self.W_MAX, self.W_MAX)


class Gripper:
    def __init__(self, model, data, prefix):
        self.d = data
        self.act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                      f'{prefix}_{ARM}_finger_joint{i}') for i in (1, 2)]

    def set_width(self, width):
        self.d.ctrl[self.act] = np.clip(width / 2.0, 0.0, 0.04)


# ---------------------------------------------------------------------------


class TeleopLink:
    """Models the operator -> robot transport: fixed rate, fixed latency, latest-wins.

    This is the piece that decides how the system feels. Samples arrive at `rate_hz` (Quest
    controller pose rate) and become visible to the 1 kHz loop only after `latency_s`. Nothing
    queues: the control loop always reads the newest sample that has arrived, so a target the
    arm never reached is simply overwritten.
    """

    def __init__(self, rate_hz=72.0, latency_s=0.040, jitter_s=0.0, dropout_prob=0.0, seed=0):
        self.period = 1.0 / rate_hz
        self.latency, self.jitter, self.dropout = latency_s, jitter_s, dropout_prob
        self.rng = np.random.default_rng(seed)
        self.inflight = deque()
        self.latest = None
        self.latest_stamp = -np.inf
        self._next_send = 0.0

    def send(self, t, payload):
        """Offer a new operator sample; only accepted on the transport's own sample grid."""
        if t + 1e-12 < self._next_send:
            return False
        self._next_send += self.period
        if self.dropout and self.rng.random() < self.dropout:
            return False
        delay = self.latency + (self.rng.normal(0, self.jitter) if self.jitter else 0.0)
        self.inflight.append((t + max(delay, 0.0), t, payload))
        return True

    def poll(self, t):
        """Latest sample that has arrived by time t, plus the age of that sample."""
        while self.inflight and self.inflight[0][0] <= t:
            _, sent, payload = self.inflight.popleft()
            self.latest, self.latest_stamp = payload, sent
        return self.latest, (t - self.latest_stamp if self.latest is not None else np.inf)


class MobileDuo:
    """Everything bolted together, stepped at the model timestep (1 ms)."""

    STALE_S = 0.15         # hold the attractor if the operator link goes quiet this long
    VEL_FF_CUTOFF_HZ = 8.0  # low-pass on the differentiated operator stream

    def __init__(self, xml_path, gains: ImpedanceGains | None = None):
        self.m = mujoco.MjModel.from_xml_path(str(xml_path))
        self.d = mujoco.MjData(self.m)
        self.dt = self.m.opt.timestep
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        mujoco.mj_forward(self.m, self.d)

        self.arms = {p: CartesianImpedance(self.m, self.d, p, gains) for p in ARM_PREFIXES}
        self.grippers = {p: Gripper(self.m, self.d, p) for p in ARM_PREFIXES}
        self.spine = Spine(self.m, self.d)
        self.base = BaseTwist(self.m, self.d)
        self.cam_site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, 'zed_left_optical')
        self.stale = {p: False for p in ARM_PREFIXES}
        self._last_sample = {}
        self._vel_filt = {p: np.zeros(3) for p in ARM_PREFIXES}

    # -- home pose --------------------------------------------------------
    def home_world(self, prefix):
        """The home TCP pose for one arm, in the world frame."""
        pos, quat = HOME_TCP_BASE[prefix]
        return self.base_to_world(pos, quat / np.linalg.norm(quat))

    def home_camera(self, prefix):
        """The same pose expressed in the ZED left optical frame."""
        return self.world_to_camera(*self.home_world(prefix))

    # -- frames -----------------------------------------------------------
    def base_pose(self):
        """The robot's own base frame: x forward, y left, z up, origin at base_link.

        Use this, not the camera frame, for operator motion. The ZED optical frame is pitched
        41 deg down by the head bracket, so mapping a gravity-aligned hand motion into it tilts
        every command by 41 deg -- pushing the controller straight forward drives the hand
        forward AND down, and only 0.151 m of a 0.20 m push survives as forward travel. The
        base frame carries the robot's heading without that pitch, and the planar base has no
        roll or pitch DoF, so it is gravity-aligned by construction.
        """
        bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        return self.d.xpos[bid].copy(), self.d.xmat[bid].reshape(3, 3).copy()

    def base_to_world(self, pos, quat):
        p_b, R_b = self.base_pose()
        return p_b + R_b @ np.asarray(pos, float), mat_to_quat(R_b @ quat_to_mat(quat))

    def world_to_base(self, pos, quat):
        p_b, R_b = self.base_pose()
        return R_b.T @ (np.asarray(pos, float) - p_b), mat_to_quat(R_b.T @ quat_to_mat(quat))

    def camera_pose(self):
        """Left ZED optical frame (x right, y down, z forward) in the world frame."""
        return (self.d.site_xpos[self.cam_site].copy(),
                self.d.site_xmat[self.cam_site].reshape(3, 3).copy())

    def camera_to_world(self, pos_cam, quat_cam):
        """Map a Quest target from camera frame into the world frame.

        The head rides on the spine and the base, so this transform is re-evaluated every
        control cycle: raising the spine or driving the base moves the operator's frame of
        reference, and the arm target must follow it.
        """
        p_c, R_c = self.camera_pose()
        return p_c + R_c @ np.asarray(pos_cam, float), mat_to_quat(R_c @ quat_to_mat(quat_cam))

    def world_to_camera(self, pos_w, quat_w):
        p_c, R_c = self.camera_pose()
        return R_c.T @ (np.asarray(pos_w, float) - p_c), mat_to_quat(R_c.T @ quat_to_mat(quat_w))

    # -- stepping ---------------------------------------------------------
    def apply_teleop(self, prefix, link: TeleopLink, t):
        """Read the newest operator sample and move that arm's attractor.

        Also finite-differences successive operator samples to get a target velocity for the
        feedforward term. The stream is discrete (72 Hz) so the raw difference is steppy; it is
        low-passed here, which is the one place a filter belongs -- off the critical path of
        the position setpoint.

        That difference is taken in CAMERA frame, not world frame, and this matters. The
        damping term it cancels acts on J*dq, and J spans only the arm's 7 joints, so it is the
        TCP velocity relative to the arm base -- not relative to the world. The camera and the
        arm bases are rigidly connected (both ride the same mount), so a camera-frame difference
        is the right quantity. Differencing in world frame instead folds in the mobile base's
        own velocity, which the arm gets for free by being carried along; feeding that forward
        double-counts it and drives the arm into a joint limit as soon as the base moves.
        """
        sample, age = link.poll(t)
        if sample is None:
            return
        if age > self.STALE_S:
            self.stale[prefix] = True
            return          # hold the last equilibrium; damping settles the arm
        self.stale[prefix] = False

        pos_cam, quat_cam = sample
        pos_w, quat_w = self.camera_to_world(pos_cam, quat_cam)
        prev = self._last_sample.get(prefix)
        vel = np.zeros(6)
        if prev is not None and link.latest_stamp > prev[1] + 1e-9:
            dt = link.latest_stamp - prev[1]
            raw = (np.asarray(pos_cam, float) - prev[0]) / dt
            a = 1.0 - np.exp(-2 * np.pi * self.VEL_FF_CUTOFF_HZ * dt)
            self._vel_filt[prefix] = (1 - a) * self._vel_filt[prefix] + a * raw
        if link.latest_stamp != (prev[1] if prev else None):
            self._last_sample[prefix] = (np.asarray(pos_cam, float), link.latest_stamp)
        vel[:3] = self.camera_pose()[1] @ self._vel_filt[prefix]   # orient, don't add base motion
        self.arms[prefix].set_target(pos_w, quat_w, vel)

    def step(self):
        for arm in self.arms.values():
            arm.apply()
        self.spine.apply(self.dt)
        mujoco.mj_step(self.m, self.d)

    @property
    def t(self):
        return self.d.time
