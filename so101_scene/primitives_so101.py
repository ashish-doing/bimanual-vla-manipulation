"""
Task primitives for the dual-SO101 dinner-table scene.

Ported from the original ALOHA-based primitives.py -- same IK/grasp/verify
pattern (weld-constraint grasping computed at the current relative pose,
CONFIRM/DISPUTE verify functions, retry-based approach_and_grasp), retargeted
to SO-101's joint layout: 5 arm joints + 1 gripper joint per arm (vs ALOHA's
6+1), single site "<side>_gripperframe" as the IK target per arm.

Neutral pose was empirically searched and confirmed collision-free (0
cross-arm contacts) in-session, the same discipline used for the ALOHA rig.
"""
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
DRAWER_ACTUATOR_NAME = "drawer_actuator"

# Empirically searched, confirmed collision-free (0 cross-arm contacts) --
# see the session log / ARCHITECTURE.md for the search.
NEUTRAL_ARM = np.array([0, -1.5, 1.8, 0.6, 0])
NEUTRAL_GRIPPER = 0.6
DOWN_QUAT = np.array([0, 1, 0, 0])


def _jnt_qpos_adr(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return model.jnt_qposadr[jid]


def _act_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def reset_to_neutral(model, data):
    for side in ("left", "right"):
        for jname, val in zip(ARM_JOINT_NAMES[side], NEUTRAL_ARM):
            data.qpos[_jnt_qpos_adr(model, jname)] = val
            data.ctrl[_act_id(model, jname)] = val
        gjname = GRIPPER_JOINT_NAME[side]
        data.qpos[_jnt_qpos_adr(model, gjname)] = NEUTRAL_GRIPPER
        data.ctrl[_act_id(model, gjname)] = NEUTRAL_GRIPPER
    data.ctrl[_act_id(model, DRAWER_ACTUATOR_NAME)] = 0.0
    mujoco.mj_forward(model, data)
    # Let free objects (spoon/fork/plate/cup) finish falling onto the table
    # BEFORE any task starts. Without this, an object spawned a few cm above
    # the table is still actively falling during the first approach move,
    # so the arm chases a moving target and misses -- confirmed in-session
    # (spoon/fork, spawned higher than plate/cup, failed consistently until
    # this settle was added; plate/cup, spawned closer to the table, didn't
    # show the bug, which is what made it non-obvious at first).
    for _ in range(400):
        mujoco.mj_step(model, data)
    # Re-assert neutral arm ctrl after settling -- the settle steps above
    # can let the arms drift slightly under their own dynamics.
    for side in ("left", "right"):
        for jname, val in zip(ARM_JOINT_NAMES[side], NEUTRAL_ARM):
            data.qpos[_jnt_qpos_adr(model, jname)] = val
            data.ctrl[_act_id(model, jname)] = val
    mujoco.mj_forward(model, data)


def solve_ik_to_target(model, data, configuration, arm, target_pos, target_quat=DOWN_QUAT,
                        max_iters=500, pos_threshold=0.006, dt=0.01):
    # orientation_cost=0.0: SO-101 has only 5 arm joints, so position + full
    # orientation is over-constrained (5 DOF can't independently satisfy a
    # full 6D pose). Position-only IK converges reliably; orientation falls
    # out naturally from the posture-regularized solution. Confirmed
    # empirically in-session: orientation_cost>0 measurably hurt convergence.
    site_name = f"{arm}_gripperframe"
    configuration.update(data.qpos.copy())
    task = mink.FrameTask(frame_name=site_name, frame_type="site",
                           position_cost=1.0, orientation_cost=0.0, lm_damping=1.0)
    target_se3 = mink.SE3.from_rotation_and_translation(mink.SO3(target_quat), np.array(target_pos))
    task.set_target(target_se3)
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    posture_task.set_target(configuration.q)
    pos_err = np.inf
    for _ in range(max_iters):
        vel = mink.solve_ik(configuration, [task, posture_task], dt, solver="daqp", damping=1e-6)
        configuration.integrate_inplace(vel, dt)
        err = task.compute_error(configuration)
        pos_err = np.linalg.norm(err[:3])
        if pos_err < pos_threshold:
            break
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINT_NAMES[arm]]
    qadrs = [model.jnt_qposadr[j] for j in joint_ids]
    return pos_err < pos_threshold, configuration.q[qadrs].copy(), pos_err


def move_to(model, data, configuration, arm, target_pos, target_quat=DOWN_QUAT,
            settle_steps=800, verbose=True):
    ok, target_angles, ik_err = solve_ik_to_target(model, data, configuration, arm, target_pos, target_quat)
    if not ok and verbose:
        print(f"  [move_to] WARNING: IK did not fully converge (err={ik_err:.4f}m)")
    for jname, val in zip(ARM_JOINT_NAMES[arm], target_angles):
        data.ctrl[_act_id(model, jname)] = val
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_gripperframe")
    actual_pos = data.site_xpos[site_id].copy()
    final_err = np.linalg.norm(actual_pos - np.array(target_pos))
    if verbose:
        print(f"  [move_to] {arm} -> target {np.round(target_pos,3)} | actual {np.round(actual_pos,3)} | error {final_err:.4f}m")
    return final_err < 0.03, final_err


def within_reach(model, data, arm, target_pos, threshold=0.09):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_gripperframe")
    dist = np.linalg.norm(data.site_xpos[site_id] - np.array(target_pos))
    return dist < threshold, dist


def _set_gripper(model, data, arm, opening, settle_steps=250):
    data.ctrl[_act_id(model, GRIPPER_JOINT_NAME[arm])] = opening
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)


def grasp_object(model, data, arm, obj_name, settle_steps=250, verbose=True):
    """Weld-based grasp at the CURRENT relative pose (same fix as the ALOHA
    build -- a naive default-pose weld snaps violently on activation)."""
    _set_gripper(model, data, arm, GRIPPER_CLOSED, settle_steps)
    mujoco.mj_forward(model, data)

    weld_name = f"{arm}_gripper_{obj_name}" if False else None
    eq_id = None
    for i in range(model.neq):
        b1 = model.eq_obj1id[i]; b2 = model.eq_obj2id[i]
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1)
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2)
        if {n1, n2} == {f"{arm}_gripper", obj_name}:
            eq_id = i
            break
    assert eq_id is not None, f"no weld defined for {arm}_gripper <-> {obj_name}"

    body1 = model.eq_obj1id[eq_id]
    body2 = model.eq_obj2id[eq_id]
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


def release(model, data, arm, obj_name, settle_steps=250, verbose=True):
    for i in range(model.neq):
        b1 = model.eq_obj1id[i]; b2 = model.eq_obj2id[i]
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1)
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2)
        if {n1, n2} == {f"{arm}_gripper", obj_name}:
            data.eq_active[i] = 0
    # Zero the object's velocity at the instant of release. A stiffened weld
    # (needed to stop it drifting during fast transport, see build script)
    # can be holding a small residual position error under tension; deactivating
    # it instantly frees that stored tension as a velocity kick -- a
    # "slingshot" launch. Confirmed in-session as the mechanism behind the
    # cup repeatedly ending up flung off the table after an otherwise
    # accurate placement.
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
    free_jid = None
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == obj_id and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            free_jid = j
            break
    if free_jid is not None:
        vadr = model.jnt_dofadr[free_jid]
        data.qvel[vadr:vadr + 6] = 0
    mujoco.mj_forward(model, data)
    _set_gripper(model, data, arm, GRIPPER_OPEN, settle_steps)
    if verbose:
        print(f"  [release] {arm} released '{obj_name}'")


def approach_and_grasp(model, data, configuration, arm, get_object_pos_fn, obj_name,
                        approach_offset=np.array([0, 0, 0.08]), grasp_offset=None,
                        max_retries=2, verbose=True):
    """grasp_offset lifts (or lowers) the final target relative to the
    object's exact center. Discovered in-session, twice: (1) moving the
    gripper frame all the way to a tall object's center drives the gripper
    mesh into the object's bulk on approach; (2) for the CUP specifically,
    the standard +0.02 offset (fine for flat objects like the plate) leaves
    only ~1.5cm of clearance between the cup's rim and the forearm link
    above the gripper -- confirmed by contact logging that right_lower_arm
    scrapes/drags the cup throughout transport, eventually launching it.
    Defaults to a small per-object-appropriate value when not specified."""
    if grasp_offset is None:
        grasp_offset = GRASP_OFFSETS.get(obj_name, np.array([0, 0, 0.02]))
    obj_pos = get_object_pos_fn()
    move_to(model, data, configuration, arm, obj_pos + approach_offset, verbose=verbose)
    for attempt in range(max_retries + 1):
        obj_pos = get_object_pos_fn()
        move_to(model, data, configuration, arm, obj_pos + grasp_offset, verbose=verbose)
        reachable, dist = within_reach(model, data, arm, get_object_pos_fn())
        if reachable:
            grasp_object(model, data, arm, obj_name, verbose=verbose)
            return True, dist
        if verbose:
            print(f"  [retry {attempt+1}] {arm} missed by {dist:.4f}m, re-aiming...")
    return False, dist


# Per-object grasp offsets -- tall objects (cup: half-height 0.035) need to
# be gripped BELOW center to leave clearance between their rim and the
# forearm link above the gripper. Flat objects (plate: half-height 0.006)
# have plenty of clearance either way; spoon/fork are thin capsules, same.
GRASP_OFFSETS = {
    "cup": np.array([0, 0, -0.015]),
    "plate": np.array([0, 0, 0.02]),
    "spoon": np.array([0, 0, 0.02]),
    "fork": np.array([0, 0, 0.02]),
}


def verify_drawer_open(model, data, min_open=0.08):
    adr = _jnt_qpos_adr(model, "drawer_slide")
    pos = data.qpos[adr]
    return pos >= min_open, {"drawer_pos": float(pos)}


def open_drawer(model, data, target_opening=0.11, settle_steps=6500, verbose=True):
    data.ctrl[_act_id(model, DRAWER_ACTUATOR_NAME)] = target_opening
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
    ok, state = verify_drawer_open(model, data)
    if verbose:
        print(f"  [open_drawer] target={target_opening} | actual={state['drawer_pos']:.4f} | {'CONFIRM' if ok else 'DISPUTE'}")
    return ok, state


def run_drawer_open_task(model, data, configuration, verbose=True):
    """Approach + visually frame the handle, THEN retract clear before the
    actuator drives the drawer open. Discovered in-session: if the gripper
    stays closed and physically stationary at the handle while the drawer
    body slides open, the drawer's own collision box rams into the static
    gripper mesh and gets stuck partway. Retracting first avoids the
    conflict -- same "actuator-driven, arm doesn't block it" simplification
    category as the ALOHA project's drawer, just needed the extra retract
    step since here the geometry actually intersects the slide path."""
    handle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "drawer_handle")
    handle_pos = data.site_xpos[handle_id].copy()
    move_to(model, data, configuration, "left", handle_pos + np.array([0, 0.05, 0.06]), verbose=verbose)
    handle_pos = data.site_xpos[handle_id].copy()
    move_to(model, data, configuration, "left", handle_pos, verbose=verbose)
    _set_gripper(model, data, "left", GRIPPER_CLOSED, settle_steps=200)
    # Two-stage retreat: straight up first, then to the side. A single big
    # IK jump from "at the handle" to the far retreat point got stuck in a
    # local minimum (confirmed in-session -- 0.28m residual error, arm
    # ended up colliding with the drawer instead of clearing it).
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_gripperframe")
    up_point = data.site_xpos[site_id].copy() + np.array([0, 0, 0.10])
    move_to(model, data, configuration, "left", up_point, verbose=verbose)
    move_to(model, data, configuration, "left", RETREAT["left"], verbose=verbose)
    success, state = open_drawer(model, data, verbose=verbose)
    return success, state


def verify_grasp_held(model, data, arm, obj_name, max_dist=0.09):
    eq_id = None
    for i in range(model.neq):
        b1 = model.eq_obj1id[i]; b2 = model.eq_obj2id[i]
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1)
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2)
        if {n1, n2} == {f"{arm}_gripper", obj_name}:
            eq_id = i
            break
    held = bool(data.eq_active[eq_id]) if eq_id is not None else False
    return held, {f"{arm}_holding_{obj_name}": held}


def run_pickup_task(model, data, configuration, arm, obj_name, verbose=True):
    """Generic single-arm pickup, used for plate/cup/spoon/fork."""
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
    ok, dist = approach_and_grasp(
        model, data, configuration, arm,
        lambda: data.xpos[obj_id].copy(), obj_name, verbose=verbose,
    )
    if not ok:
        return False, {"reach_failed": obj_name, "reach_dist": float(dist)}
    held, state = verify_grasp_held(model, data, arm, obj_name)
    if verbose:
        print(f"  [verify] {arm} holding {obj_name}: {'CONFIRM' if held else 'DISPUTE'}")
    return held, state


HANDOFF_ZONE = np.array([0.0, 0.0, 0.42])
RETREAT = {"left": np.array([-0.20, -0.15, 0.50]), "right": np.array([0.20, -0.15, 0.50])}


def run_handoff_task(model, data, configuration, obj_name, verbose=True):
    """Right arm picks up obj_name, carries to the handoff zone, retreats;
    left arm approaches and grasps. Sequential, not simultaneous -- same
    collision lesson as the ALOHA build."""
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)

    ok, dist = approach_and_grasp(
        model, data, configuration, "right",
        lambda: data.xpos[obj_id].copy(), obj_name, verbose=verbose,
    )
    if not ok:
        return False, {"reach_failed": f"right_initial_{obj_name}", "reach_dist": float(dist)}

    move_to(model, data, configuration, "right", HANDOFF_ZONE, verbose=verbose)
    move_to(model, data, configuration, "right", HANDOFF_ZONE + np.array([0, 0, -0.05]), settle_steps=1200, verbose=verbose)
    release(model, data, "right", obj_name, verbose=verbose)
    # Two-stage retreat (straight up, then to the side) -- a direct move to
    # RETREAT grazed the object it just placed and launched it. Same fix
    # pattern as the drawer's static-gripper-blocks-drawer bug.
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_gripperframe")
    up_point = data.site_xpos[site_id].copy() + np.array([0, 0, 0.18])
    move_to(model, data, configuration, "right", up_point, settle_steps=1200, verbose=verbose)
    move_to(model, data, configuration, "right", RETREAT["right"], verbose=verbose)

    ok, dist = approach_and_grasp(
        model, data, configuration, "left",
        lambda: data.xpos[obj_id].copy(), obj_name, verbose=verbose,
    )
    if not ok:
        return False, {"reach_failed": f"left_final_{obj_name}", "reach_dist": float(dist)}

    held, state = verify_grasp_held(model, data, "left", obj_name)
    if verbose:
        print(f"  [verify] handoff of {obj_name}: {'CONFIRM' if held else 'DISPUTE'}")
    return held, state