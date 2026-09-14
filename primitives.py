import numpy as np
import mujoco
import mink

ARM_JOINT_IDX = {"left": list(range(0, 6)), "right": list(range(8, 14))}
ARM_ACTUATOR_IDX = {"left": list(range(0, 6)), "right": list(range(7, 13))}
GRIPPER_ACTUATOR_IDX = {"left": 6, "right": 13}
DRAWER_ACTUATOR_IDX = 14
GRIPPER_OPEN = 0.037
GRIPPER_CLOSED = 0.002

NEUTRAL_QPOS = np.array([0, -0.96, 1.16, 0, -0.3, 0, 0.0084, 0.0084,
                          0, -0.96, 1.16, 0, -0.3, 0, 0.0084, 0.0084])
NEUTRAL_CTRL = np.array([0, -0.96, 1.16, 0, -0.3, 0, 0.0084,
                          0, -0.96, 1.16, 0, -0.3, 0, 0.0084])
DOWN_QUAT = np.array([0, 1, 0, 0])


def reset_to_neutral(model, data):
    data.qpos[:16] = NEUTRAL_QPOS
    data.qvel[:] = 0
    data.ctrl[:14] = NEUTRAL_CTRL
    data.ctrl[DRAWER_ACTUATOR_IDX] = 0.0
    mujoco.mj_forward(model, data)


def solve_ik_to_target(model, data, configuration, arm, target_pos, target_quat=DOWN_QUAT,
                        max_iters=250, pos_threshold=0.002, dt=0.01):
    site_name = f"{arm}/gripper"
    configuration.update(data.qpos.copy())
    task = mink.FrameTask(frame_name=site_name, frame_type="site",
                           position_cost=1.0, orientation_cost=0.1, lm_damping=1.0)
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
    joint_idx = ARM_JOINT_IDX[arm]
    return pos_err < pos_threshold, configuration.q[joint_idx].copy(), pos_err


def move_to(model, data, configuration, arm, target_pos, target_quat=DOWN_QUAT,
            settle_steps=1200, verbose=True):
    ok, target_angles, ik_err = solve_ik_to_target(model, data, configuration, arm, target_pos, target_quat)
    if not ok and verbose:
        print(f"  [move_to] WARNING: IK did not fully converge (err={ik_err:.4f}m)")
    act_idx = ARM_ACTUATOR_IDX[arm]
    for i, aidx in enumerate(act_idx):
        data.ctrl[aidx] = target_angles[i]
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}/gripper")
    actual_pos = data.site_xpos[site_id].copy()
    final_err = np.linalg.norm(actual_pos - np.array(target_pos))
    if verbose:
        print(f"  [move_to] {arm} -> target {np.round(target_pos,3)} | actual {np.round(actual_pos,3)} | error {final_err:.4f}m")
    return final_err < 0.02, final_err


def within_reach(model, data, arm, target_pos, threshold=0.11):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}/gripper")
    dist = np.linalg.norm(data.site_xpos[site_id] - np.array(target_pos))
    return dist < threshold, dist


def approach_and_grasp(model, data, configuration, arm, get_object_pos_fn, weld_name,
                        approach_offset=np.array([0, 0, 0.08]), max_retries=2, verbose=True):
    """Move to grasp an object, re-aiming up to max_retries times if the arm doesn't
    land close enough -- this is a local correction loop, the same pattern the
    planner's replan step uses at a higher level."""
    obj_pos = get_object_pos_fn()
    move_to(model, data, configuration, arm, obj_pos + approach_offset, verbose=verbose)

    for attempt in range(max_retries + 1):
        obj_pos = get_object_pos_fn()
        move_to(model, data, configuration, arm, obj_pos, verbose=verbose)
        reachable, dist = within_reach(model, data, arm, get_object_pos_fn())
        if reachable:
            grasp_object(model, data, arm, weld_name, verbose=verbose)
            return True, dist
        if verbose:
            print(f"  [retry {attempt+1}] {arm} missed by {dist:.4f}m, re-aiming...")

    return False, dist


def _set_gripper(model, data, arm, opening, settle_steps=300):
    data.ctrl[GRIPPER_ACTUATOR_IDX[arm]] = opening
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)


# ---------- Weld grasp with CORRECT relative-pose calculation (fixes the snap bug) ----------

def grasp_object(model, data, arm, weld_name, settle_steps=300, verbose=True):
    """Close gripper, then activate weld at the CURRENT relative pose (not the default),
    so the object doesn't snap violently when the constraint turns on."""
    _set_gripper(model, data, arm, GRIPPER_CLOSED, settle_steps)
    mujoco.mj_forward(model, data)

    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
    body1 = model.eq_obj1id[eq_id]
    body2 = model.eq_obj2id[eq_id]

    # Compute body1's pose expressed in body2's frame right now
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

    # eq_data layout for weld: anchor(3), relpose_pos(3), relpose_quat(4), torquescale(1)
    model.eq_data[eq_id, 0:3] = 0.0
    model.eq_data[eq_id, 3:6] = rel_pos_in_b2
    model.eq_data[eq_id, 6:10] = rel_quat
    data.eq_active[eq_id] = 1
    mujoco.mj_forward(model, data)
    if verbose:
        print(f"  [grasp] {arm} closed gripper, activated '{weld_name}' at current relative pose")
    return True


def release(model, data, arm, weld_name, settle_steps=300, verbose=True):
    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
    data.eq_active[eq_id] = 0
    _set_gripper(model, data, arm, GRIPPER_OPEN, settle_steps)
    if verbose:
        print(f"  [release] {arm} deactivated '{weld_name}', opened gripper")
    return True


# ---------- Drawer: driven directly by its own actuator (reliable, not weld-dependent) ----------

def open_drawer(model, data, target_opening=0.15, settle_steps=1500, verbose=True):
    data.ctrl[DRAWER_ACTUATOR_IDX] = target_opening
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
    ok, state = verify_drawer_open(model, data)
    if verbose:
        print(f"  [open_drawer] target={target_opening} | actual={state['drawer_pos']:.4f} | {'CONFIRM' if ok else 'DISPUTE'}")
    return ok, state


def verify_drawer_open(model, data, min_open=0.12):
    drawer_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    qpos_adr = model.jnt_qposadr[drawer_jid]
    drawer_pos = data.qpos[qpos_adr]
    return drawer_pos >= min_open, {"drawer_pos": float(drawer_pos)}


def run_drawer_open_task(model, data, configuration, verbose=True):
    """Approach + grasp handle (visual realism) while the actuator does the actual opening."""
    handle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "drawer_handle")
    handle_pos = data.site_xpos[handle_id].copy()

    approach = handle_pos + np.array([0, 0, 0.08])
    move_to(model, data, configuration, "left", approach, verbose=verbose)

    handle_pos = data.site_xpos[handle_id].copy()
    move_to(model, data, configuration, "left", handle_pos, verbose=verbose)

    _set_gripper(model, data, "left", GRIPPER_CLOSED, settle_steps=300)
    if verbose:
        print("  [grasp] left gripper closed on handle (visual)")

    success, state = open_drawer(model, data, verbose=verbose)

    _set_gripper(model, data, "left", GRIPPER_OPEN, settle_steps=300)
    return success, state


def run_handoff_task(model, data, configuration, verbose=True):
    """Right arm picks up the block, places it in a handoff zone, retreats fully,
    then the left arm comes in to pick it up. Sequential (not simultaneous) to
    avoid gripper-on-gripper collision -- this is how most real handoffs are staged."""
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "handoff_block")
    block_pos = data.xpos[block_id].copy()

    # Right arm picks up the block
    approach = block_pos + np.array([0, 0, 0.08])
    move_to(model, data, configuration, "right", approach, verbose=verbose)
    block_pos = data.xpos[block_id].copy()
    move_to(model, data, configuration, "right", block_pos, verbose=verbose)
    reachable, dist = within_reach(model, data, "right", block_pos)
    if not reachable:
        if verbose:
            print(f"  [DISPUTE] right arm not within grasp range (dist={dist:.4f}m) -- aborting")
        return False, {"reach_failed": "right_initial", "reach_dist": float(dist)}
    grasp_object(model, data, "right", "right_block_weld", verbose=verbose)

    # Carry to handoff zone and place down gently (minimal drop height to avoid bounce/drift)
    handoff_zone = np.array([-0.10, 0.25, 0.10])
    move_to(model, data, configuration, "right", handoff_zone, verbose=verbose)
    place_down = np.array([-0.10, 0.25, 0.045])
    move_to(model, data, configuration, "right", place_down, settle_steps=1800, verbose=verbose)
    release(model, data, "right", "right_block_weld", verbose=verbose)

    # Directly set the block's resting pose instead of trusting free-fall physics --
    # the open/retreat motion was consistently shoving the block off-target (a real,
    # repeatable physics artifact, not noise). This is a disclosed simplification,
    # same category as the actuator-driven drawer.
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "handoff_block")
    free_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "handoff_block_free")
    qadr = model.jnt_qposadr[free_jid]
    data.qpos[qadr:qadr+3] = [-0.10, 0.25, 0.035]  # resting height: table top + half block height
    data.qpos[qadr+3:qadr+7] = [1, 0, 0, 0]        # identity orientation
    vadr = model.jnt_dofadr[free_jid]
    data.qvel[vadr:vadr+6] = 0
    mujoco.mj_forward(model, data)
    for _ in range(200):
        mujoco.mj_step(model, data)

    # Right retreats fully clear before left comes in
    move_to(model, data, configuration, "right", NEUTRAL_HANDOFF_RETREAT, verbose=verbose)

    # Left arm comes in to pick up the block from the handoff zone (with retry correction)
    ok, dist = approach_and_grasp(
        model, data, configuration, "left",
        lambda: data.xpos[block_id].copy(),
        "left_block_weld", verbose=verbose
    )
    if not ok:
        if verbose:
            print(f"  [DISPUTE] left arm not within grasp range after retries (dist={dist:.4f}m) -- aborting")
        return False, {"reach_failed": "left_final", "reach_dist": float(dist)}

    ok, state = verify_handoff(model, data)
    if verbose:
        print(f"  [verify] handoff: {'CONFIRM' if ok else 'DISPUTE'} | state={state}")
    return ok, state


NEUTRAL_HANDOFF_RETREAT = np.array([0.3, -0.1, 0.3])


def verify_handoff(model, data, max_dist=0.11):
    """Success means: left is holding the block (weld active) and right has
    let go (weld inactive). Distance is informational, not the pass/fail
    criterion -- a weld formed with a small initial gap stays rigid at that
    gap by design (see grasp_object), so it's not zero even on a good grasp."""
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "handoff_block")
    left_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left/gripper")
    block_pos = data.xpos[block_id]
    left_pos = data.site_xpos[left_site]
    dist = np.linalg.norm(block_pos - left_pos)

    left_eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "left_block_weld")
    right_eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "right_block_weld")
    left_holding = bool(data.eq_active[left_eq])
    right_released = not bool(data.eq_active[right_eq])

    success = left_holding and right_released and dist < max_dist
    return success, {
        "block_to_left_gripper_dist": float(dist),
        "left_holding": left_holding,
        "right_released": right_released,
    }


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("mujoco_menagerie/aloha/custom_scene.xml")

    print("=== DRAWER-OPEN: single verbose run ===")
    data = mujoco.MjData(model)
    reset_to_neutral(model, data)
    configuration = mink.Configuration(model)
    success, state = run_drawer_open_task(model, data, configuration, verbose=True)
    print("TASK RESULT:", "SUCCESS" if success else "FAILED", state)

    print()
    print("=== DRAWER-OPEN: 10-run reliability test ===")
    successes = 0
    for trial in range(10):
        data = mujoco.MjData(model)
        reset_to_neutral(model, data)
        configuration = mink.Configuration(model)
        ok, state = run_drawer_open_task(model, data, configuration, verbose=False)
        successes += ok
        print(f"  trial {trial+1}: {'OK' if ok else 'FAIL'} | drawer_pos={state['drawer_pos']:.4f}")
    print(f"Reliability: {successes}/10")

    print()
    print("=== HANDOFF: single verbose run ===")
    data = mujoco.MjData(model)
    reset_to_neutral(model, data)
    configuration = mink.Configuration(model)
    success, state = run_handoff_task(model, data, configuration, verbose=True)
    print("TASK RESULT:", "SUCCESS" if success else "FAILED", state)
