"""
Backend abstraction for so101_scene primitives.

Splits primitive-level arm operations (move, grasp, release, pose/velocity
reads) behind a protocol so primitives_so101.py's task-level functions
don't talk to MuJoCo directly. SimBackend wraps the existing MuJoCo/mink
code -- pure refactor, no logic changes except wait_for_settle, which is
new (see run_handoff_task fix). RealBackend is a stub until SO-101
hardware exists (target: LeRobot's SO101Follower/BiSOFollower --
connect/get_observation/send_action/disconnect -- per the hardware
bring-up chat's research).
"""
from abc import ABC, abstractmethod
import numpy as np
import mujoco
import mink

ARM_JOINT_NAMES = {
    "left": ["left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
              "left_wrist_flex", "left_wrist_roll"],
    "right": ["right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
               "right_wrist_flex", "right_wrist_roll"],
}
GRIPPER_JOINT_NAME = {"left": "left_gripper", "right": "right_gripper"}
GRIPPER_OPEN = 1.2
GRIPPER_CLOSED = -0.3
DOWN_QUAT = np.array([0, 1, 0, 0])


def _jnt_qpos_adr(model, name):
    return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]


def _act_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


class ArmBackend(ABC):
    @abstractmethod
    def reset_to_neutral(self): ...
    @abstractmethod
    def move_to(self, arm, target_pos, target_quat=DOWN_QUAT, settle_steps=800, verbose=True): ...
    @abstractmethod
    def within_reach(self, arm, target_pos, threshold=0.09): ...
    @abstractmethod
    def set_gripper(self, arm, opening, settle_steps=250): ...
    @abstractmethod
    def grasp(self, arm, obj_name, settle_steps=250, verbose=True): ...
    @abstractmethod
    def release(self, arm, obj_name, settle_steps=250, verbose=True): ...
    @abstractmethod
    def damp_object(self, obj_name, steps=400, verbose=True): ...
    @abstractmethod
    def get_object_pos(self, obj_name): ...
    @abstractmethod
    def get_site_pos(self, site_name): ...
    @abstractmethod
    def object_velocity(self, obj_name): ...
    @abstractmethod
    def wait_for_settle(self, obj_name, vel_threshold=0.02, max_steps=2000, check_every=50, verbose=True): ...
    @abstractmethod
    def verify_grasp_held(self, arm, obj_name): ...
    @abstractmethod
    def set_drawer(self, target_opening, settle_steps=6500): ...
    @abstractmethod
    def verify_drawer_open(self, min_open=0.08): ...


class SimBackend(ArmBackend):
    def __init__(self, model, data, configuration):
        self.model, self.data, self.configuration = model, data, configuration

    def reset_to_neutral(self):
        model, data = self.model, self.data
        NEUTRAL_ARM = np.array([0, -1.5, 1.8, 0.6, 0])
        NEUTRAL_GRIPPER = 0.6
        for side in ("left", "right"):
            for jname, val in zip(ARM_JOINT_NAMES[side], NEUTRAL_ARM):
                data.qpos[_jnt_qpos_adr(model, jname)] = val
                data.ctrl[_act_id(model, jname)] = val
            gjname = GRIPPER_JOINT_NAME[side]
            data.qpos[_jnt_qpos_adr(model, gjname)] = NEUTRAL_GRIPPER
            data.ctrl[_act_id(model, gjname)] = NEUTRAL_GRIPPER
        data.ctrl[_act_id(model, "drawer_actuator")] = 0.0
        mujoco.mj_forward(model, data)
        for _ in range(400):
            mujoco.mj_step(model, data)
        for side in ("left", "right"):
            for jname, val in zip(ARM_JOINT_NAMES[side], NEUTRAL_ARM):
                data.qpos[_jnt_qpos_adr(model, jname)] = val
                data.ctrl[_act_id(model, jname)] = val
        mujoco.mj_forward(model, data)

    def move_to(self, arm, target_pos, target_quat=DOWN_QUAT, settle_steps=800, verbose=True):
        model, data, configuration = self.model, self.data, self.configuration
        site_name = f"{arm}_gripperframe"
        configuration.update(data.qpos.copy())
        task = mink.FrameTask(frame_name=site_name, frame_type="site",
                               position_cost=1.0, orientation_cost=0.0, lm_damping=1.0)
        task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(target_quat), np.array(target_pos)))
        posture_task = mink.PostureTask(model=model, cost=1e-2)
        posture_task.set_target(configuration.q)
        pos_err = np.inf
        for _ in range(500):
            vel = mink.solve_ik(configuration, [task, posture_task], 0.01, solver="daqp", damping=1e-6)
            configuration.integrate_inplace(vel, 0.01)
            pos_err = np.linalg.norm(task.compute_error(configuration)[:3])
            if pos_err < 0.006:
                break
        if pos_err >= 0.006 and verbose:
            print(f"  [move_to] WARNING: IK did not fully converge (err={pos_err:.4f}m)")
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINT_NAMES[arm]]
        target_angles = configuration.q[[model.jnt_qposadr[j] for j in joint_ids]].copy()
        for jname, val in zip(ARM_JOINT_NAMES[arm], target_angles):
            data.ctrl[_act_id(model, jname)] = val
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
        actual_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)].copy()
        final_err = np.linalg.norm(actual_pos - np.array(target_pos))
        if verbose:
            print(f"  [move_to] {arm} -> target {np.round(target_pos,3)} | actual {np.round(actual_pos,3)} | error {final_err:.4f}m")
        return final_err < 0.03, final_err

    def within_reach(self, arm, target_pos, threshold=0.09):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_gripperframe")
        dist = np.linalg.norm(self.data.site_xpos[site_id] - np.array(target_pos))
        return dist < threshold, dist

    def set_gripper(self, arm, opening, settle_steps=250):
        self.data.ctrl[_act_id(self.model, GRIPPER_JOINT_NAME[arm])] = opening
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)

    def grasp(self, arm, obj_name, settle_steps=250, verbose=True):
        model, data = self.model, self.data
        self.set_gripper(arm, GRIPPER_CLOSED, settle_steps)
        mujoco.mj_forward(model, data)
        eq_id = None
        for i in range(model.neq):
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.eq_obj1id[i])
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.eq_obj2id[i])
            if {n1, n2} == {f"{arm}_gripper", obj_name}:
                eq_id = i
                break
        assert eq_id is not None, f"no weld defined for {arm}_gripper <-> {obj_name}"
        body1, body2 = model.eq_obj1id[eq_id], model.eq_obj2id[eq_id]
        b1_pos, b1_quat = data.xpos[body1].copy(), data.xquat[body1].copy()
        b2_pos, b2_quat = data.xpos[body2].copy(), data.xquat[body2].copy()
        b2_quat_inv = np.zeros(4)
        mujoco.mju_negQuat(b2_quat_inv, b2_quat)
        rel_pos = np.zeros(3)
        mujoco.mju_sub3(rel_pos, b1_pos, b2_pos)
        rel_pos_in_b2 = np.zeros(3)
        mujoco.mju_rotVecQuat(rel_pos_in_b2, rel_pos, b2_quat_inv)
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, b2_quat_inv, b1_quat)
        model.eq_data[eq_id, 0:3] = 0.0
        model.eq_data[eq_id, 3:6] = rel_pos_in_b2
        model.eq_data[eq_id, 6:10] = rel_quat
        data.eq_active[eq_id] = 1
        mujoco.mj_forward(model, data)
        if verbose:
            print(f"  [grasp] {arm} closed gripper on '{obj_name}'")
        return True

    def release(self, arm, obj_name, settle_steps=250, verbose=True):
        model, data = self.model, self.data
        for i in range(model.neq):
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.eq_obj1id[i])
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.eq_obj2id[i])
            if {n1, n2} == {f"{arm}_gripper", obj_name}:
                data.eq_active[i] = 0
        obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        free_vadr = None
        for j in range(model.njnt):
            if model.jnt_bodyid[j] == obj_id and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                free_vadr = model.jnt_dofadr[j]
                break
        if free_vadr is not None:
            data.qvel[free_vadr:free_vadr + 6] = 0
        mujoco.mj_forward(model, data)

        self.set_gripper(arm, GRIPPER_OPEN, settle_steps)
        if verbose:
            print(f"  [release] {arm} released '{obj_name}'")

    def damp_object(self, obj_name, steps=400, verbose=True):
        """Re-zero obj_name's free-joint velocity every physics step for
        `steps` steps. Used after release+retreat's own damping window to
        absorb residual drift before the retreat moves run -- confirmed
        in-session that ~60 steps of damping inside release() wasn't
        enough; leftover drift carried through the retreat sequence and
        walked the cup off the table edge on ~half of randomized seeds."""
        model, data = self.model, self.data
        obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        free_vadr = None
        for j in range(model.njnt):
            if model.jnt_bodyid[j] == obj_id and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                free_vadr = model.jnt_dofadr[j]
                break
        for _ in range(steps):
            mujoco.mj_step(model, data)
            if free_vadr is not None:
                data.qvel[free_vadr:free_vadr + 6] = 0
        if verbose:
            print(f"  [damp] {obj_name} damped for {steps} steps")

    def get_object_pos(self, obj_name):
        return self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)].copy()

    def get_site_pos(self, site_name):
        return self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)].copy()

    def object_velocity(self, obj_name):
        model, data = self.model, self.data
        obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        for j in range(model.njnt):
            if model.jnt_bodyid[j] == obj_id and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                vadr = model.jnt_dofadr[j]
                return float(np.linalg.norm(data.qvel[vadr:vadr + 6]))
        return 0.0

    def wait_for_settle(self, obj_name, pos_threshold=0.003, max_steps=3000, check_every=100,
                         min_z=0.15, verbose=True):
        """Steps physics until obj_name's position stops changing
        meaningfully between checks, or max_steps elapses. Uses POSITION
        stability, not velocity. Fails fast if z drops below min_z,
        meaning the object fell off the table onto the floor -- confirmed
        in-session that a floor-level object can drift indefinitely
        without ever satisfying a position-stability check."""
        prev_pos = self.get_object_pos(obj_name)
        for step in range(0, max_steps, check_every):
            for _ in range(check_every):
                mujoco.mj_step(self.model, self.data)
            pos = self.get_object_pos(obj_name)
            if pos[2] < min_z:
                if verbose:
                    print(f"  [settle] {obj_name} fell off table (z={pos[2]:.4f}) after {step+check_every} steps")
                return False, step + check_every
            delta = float(np.linalg.norm(pos - prev_pos))
            if delta < pos_threshold:
                if verbose:
                    print(f"  [settle] {obj_name} settled after {step+check_every} steps (delta={delta:.5f})")
                return True, step + check_every
            prev_pos = pos
        if verbose:
            print(f"  [settle] WARNING: {obj_name} did not settle within {max_steps} steps (still drifting)")
        return False, max_steps

    def verify_grasp_held(self, arm, obj_name):
        model, data = self.model, self.data
        eq_id = None
        for i in range(model.neq):
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.eq_obj1id[i])
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.eq_obj2id[i])
            if {n1, n2} == {f"{arm}_gripper", obj_name}:
                eq_id = i
                break
        held = bool(data.eq_active[eq_id]) if eq_id is not None else False
        return held, {f"{arm}_holding_{obj_name}": held}

    def set_drawer(self, target_opening, settle_steps=6500):
        self.data.ctrl[_act_id(self.model, "drawer_actuator")] = target_opening
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)

    def verify_drawer_open(self, min_open=0.08):
        pos = self.data.qpos[_jnt_qpos_adr(self.model, "drawer_slide")]
        return pos >= min_open, {"drawer_pos": float(pos)}


class RealBackend(ArmBackend):
    """Stub. Target: LeRobot SO101Follower/BiSOFollower. Every method
    raises -- no silent fallback to sim behavior."""
    def __init__(self, *a, **kw):
        raise NotImplementedError("RealBackend: no hardware yet")
    def reset_to_neutral(self): raise NotImplementedError
    def move_to(self, *a, **kw): raise NotImplementedError
    def within_reach(self, *a, **kw): raise NotImplementedError
    def set_gripper(self, *a, **kw): raise NotImplementedError
    def grasp(self, *a, **kw): raise NotImplementedError
    def release(self, *a, **kw): raise NotImplementedError
    def damp_object(self, *a, **kw): raise NotImplementedError
    def get_object_pos(self, *a, **kw): raise NotImplementedError
    def get_site_pos(self, *a, **kw): raise NotImplementedError
    def object_velocity(self, *a, **kw): raise NotImplementedError
    def wait_for_settle(self, *a, **kw): raise NotImplementedError
    def verify_grasp_held(self, *a, **kw): raise NotImplementedError
    def set_drawer(self, *a, **kw): raise NotImplementedError
    def verify_drawer_open(self, *a, **kw): raise NotImplementedError


def make_backend(kind, model=None, data=None, configuration=None):
    if kind == "sim":
        return SimBackend(model, data, configuration)
    if kind == "real":
        return RealBackend()
    raise ValueError(f"unknown backend kind: {kind}")
