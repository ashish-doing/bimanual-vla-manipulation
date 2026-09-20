"""
Task primitives for the dual-SO101 dinner-table scene.

Backend-agnostic: talks to arm_backend.ArmBackend, not MuJoCo directly.
See arm_backend.py for SimBackend/RealBackend.
"""
import numpy as np
from arm_backend import GRIPPER_CLOSED

HANDOFF_ZONE = np.array([0.0, 0.0, 0.42])
RETREAT = {"left": np.array([-0.20, -0.15, 0.50]), "right": np.array([0.20, -0.15, 0.50])}
GRASP_OFFSETS = {
    "cup": np.array([0, 0, -0.015]),
    "plate": np.array([0, 0, 0.02]),
    "spoon": np.array([0, 0, 0.02]),
    "fork": np.array([0, 0, 0.02]),
}


def approach_and_grasp(backend, arm, obj_name, approach_offset=np.array([0, 0, 0.08]),
                        grasp_offset=None, max_retries=2, verbose=True):
    if grasp_offset is None:
        grasp_offset = GRASP_OFFSETS.get(obj_name, np.array([0, 0, 0.02]))
    obj_pos = backend.get_object_pos(obj_name)
    backend.move_to(arm, obj_pos + approach_offset, verbose=verbose)
    for attempt in range(max_retries + 1):
        obj_pos = backend.get_object_pos(obj_name)
        backend.move_to(arm, obj_pos + grasp_offset, verbose=verbose)
        reachable, dist = backend.within_reach(arm, backend.get_object_pos(obj_name))
        if reachable:
            backend.grasp(arm, obj_name, verbose=verbose)
            return True, dist
        if verbose:
            print(f"  [retry {attempt+1}] {arm} missed by {dist:.4f}m, re-aiming...")
    return False, dist


def run_drawer_open_task(backend, verbose=True):
    handle_pos = backend.get_site_pos("drawer_handle")
    backend.move_to("left", handle_pos + np.array([0, 0.05, 0.06]), verbose=verbose)
    handle_pos = backend.get_site_pos("drawer_handle")
    backend.move_to("left", handle_pos, verbose=verbose)
    backend.set_gripper("left", GRIPPER_CLOSED, settle_steps=200)
    up_point = backend.get_site_pos("left_gripperframe") + np.array([0, 0, 0.10])
    backend.move_to("left", up_point, verbose=verbose)
    backend.move_to("left", RETREAT["left"], verbose=verbose)
    backend.set_drawer(0.11, settle_steps=6500)
    success, state = backend.verify_drawer_open()
    if verbose:
        print(f"  [open_drawer] actual={state['drawer_pos']:.4f} | {'CONFIRM' if success else 'DISPUTE'}")
    return success, state


def run_pickup_task(backend, arm, obj_name, verbose=True):
    ok, dist = approach_and_grasp(backend, arm, obj_name, verbose=verbose)
    if not ok:
        return False, {"reach_failed": obj_name, "reach_dist": float(dist)}
    held, state = backend.verify_grasp_held(arm, obj_name)
    if verbose:
        print(f"  [verify] {arm} holding {obj_name}: {'CONFIRM' if held else 'DISPUTE'}")
    return held, state


def run_handoff_task(backend, obj_name, verbose=True):
    ok, dist = approach_and_grasp(backend, "right", obj_name, verbose=verbose)
    if not ok:
        return False, {"reach_failed": f"right_initial_{obj_name}", "reach_dist": float(dist)}

    backend.move_to("right", HANDOFF_ZONE, verbose=verbose)
    backend.move_to("right", HANDOFF_ZONE + np.array([0, 0, -0.05]), settle_steps=1200, verbose=verbose)
    backend.release("right", obj_name, verbose=verbose)

    up_point = backend.get_site_pos("right_gripperframe") + np.array([0, 0, 0.18])
    backend.move_to("right", up_point, settle_steps=1200, verbose=verbose)
    backend.move_to("right", RETREAT["right"], verbose=verbose)

    ok, dist = approach_and_grasp(backend, "left", obj_name, verbose=verbose)
    if not ok:
        return False, {"reach_failed": f"left_final_{obj_name}", "reach_dist": float(dist)}

    held, state = backend.verify_grasp_held("left", obj_name)
    if verbose:
        print(f"  [verify] handoff of {obj_name}: {'CONFIRM' if held else 'DISPUTE'}")
    return held, state