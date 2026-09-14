"""
Verify/replan loop: takes a plan (from planner.make_plan()) and executes each
step against the MuJoCo sim, using the primitive-level verify functions in
primitives.py as the pass/fail (CONFIRM/DISPUTE) signal. Each step gets up to
MAX_RETRIES total attempts before the whole plan is marked FAILED.

This mirrors the CONFIRM/DISPUTE verification loop from QUARRY: every action
is checked against real sim state after it runs, never assumed to have
worked just because it executed without throwing.
"""

import time
import mujoco
import mink

import primitives as prim

MAX_RETRIES = 3  # total attempts per step, including the first

ACTION_RUNNERS = {
    "drawer_open": prim.run_drawer_open_task,
    "handoff": prim.run_handoff_task,
}

# Spawn state for task objects, used to reset between retries so a failed
# attempt (e.g. a dropped block) doesn't leave the object stranded out of
# reach for the next try.
DRAWER_JOINT = "drawer_slide"
HANDOFF_BLOCK_JOINT = "handoff_block_free"
HANDOFF_BLOCK_SPAWN_POS = [0.15, 0.10, 0.035]
HANDOFF_BLOCK_SPAWN_QUAT = [1, 0, 0, 0]
WELD_NAMES = ("left_drawer_weld", "right_block_weld", "left_block_weld")


def _reset_task_objects(model, data):
    """Reset drawer + handoff block to spawn state (reset_to_neutral only
    touches the arms, not these -- call both when retrying a failed step)."""
    drawer_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, DRAWER_JOINT)
    data.qpos[model.jnt_qposadr[drawer_jid]] = 0.0

    block_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, HANDOFF_BLOCK_JOINT)
    qadr = model.jnt_qposadr[block_jid]
    data.qpos[qadr:qadr + 3] = HANDOFF_BLOCK_SPAWN_POS
    data.qpos[qadr + 3:qadr + 7] = HANDOFF_BLOCK_SPAWN_QUAT
    vadr = model.jnt_dofadr[block_jid]
    data.qvel[vadr:vadr + 6] = 0

    for weld in WELD_NAMES:
        eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld)
        data.eq_active[eq_id] = 0

    mujoco.mj_forward(model, data)


def run_plan(model, data, configuration, plan, on_event=None):
    """Execute a validated plan dict from planner.make_plan().

    on_event: optional callback(dict) called after every meaningful event
    (step_start, attempt_result, step_result, plan_result, unsupported) --
    this is the hook the dashboard's WebSocket handler uses to stream
    progress live. Safe to leave as None for a plain script run.
    """
    def emit(event):
        if on_event:
            on_event(event)

    if plan.get("unsupported"):
        emit({"type": "unsupported", "message": plan["unsupported"]})

    steps = plan.get("steps", [])
    if not steps:
        result = {"overall": "FAILED", "reason": "no valid steps in plan", "steps": []}
        emit({"type": "plan_result", **result})
        return result

    step_results = []
    for i, step in enumerate(steps):
        action = step["action"]
        runner = ACTION_RUNNERS[action]
        emit({"type": "step_start", "index": i, "action": action, "reason": step.get("reason", "")})

        success = False
        attempts = []
        for attempt in range(1, MAX_RETRIES + 1):
            prim.reset_to_neutral(model, data)
            if attempt > 1:
                _reset_task_objects(model, data)
            t0 = time.time()
            ok, state = runner(model, data, configuration, verbose=False)
            elapsed = time.time() - t0
            tag = "CONFIRM" if ok else "DISPUTE"
            attempts.append({"attempt": attempt, "tag": tag, "state": state, "elapsed_s": round(elapsed, 2)})
            emit({"type": "attempt_result", "index": i, "action": action,
                  "attempt": attempt, "tag": tag, "state": state})
            if ok:
                success = True
                break

        step_results.append({"action": action, "success": success, "attempts": attempts})
        emit({"type": "step_result", "index": i, "action": action, "success": success,
              "attempts_used": len(attempts)})

        if not success:
            result = {
                "overall": "FAILED",
                "reason": f"step {i} ('{action}') DISPUTEd after {MAX_RETRIES} attempts",
                "steps": step_results,
            }
            emit({"type": "plan_result", **result})
            return result

    result = {"overall": "SUCCESS", "reason": "", "steps": step_results}
    emit({"type": "plan_result", **result})
    return result


def new_sim(scene_path="mujoco_menagerie/aloha/custom_scene.xml"):
    """Fresh model/data/configuration triple, ready to run a plan."""
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    prim.reset_to_neutral(model, data)
    configuration = mink.Configuration(model)
    return model, data, configuration


if __name__ == "__main__":
    import sys
    from planner import make_plan

    cmd = " ".join(sys.argv[1:]) or "open the drawer, then hand off the block"
    print(f"Command: {cmd!r}")
    plan = make_plan(cmd)
    print("Plan:", plan, "\n")

    model, data, configuration = new_sim()
    result = run_plan(model, data, configuration, plan, on_event=lambda e: print(" event:", e))
    print("\nFinal result:", result["overall"], "-", result.get("reason", ""))
