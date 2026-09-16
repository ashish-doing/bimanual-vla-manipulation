"""
Verify/replan executor for the SO-101 dinner-table task set. Same
CONFIRM/DISPUTE + bounded-retry pattern as the original ALOHA build's
executor.py, repointed at primitives_so101.py's verified-reliable tasks.
"""
import time
import mujoco
import mink

import primitives_so101 as prim

MAX_RETRIES = 3

SCENE_PATH = "dinner_scene.xml"


def _run_step(model, data, configuration, step):
    action = step["action"]
    if action == "drawer_open":
        return prim.run_drawer_open_task(model, data, configuration, verbose=False)
    elif action == "pickup":
        return prim.run_pickup_task(model, data, configuration, step["arm"], step["object"], verbose=False)
    raise ValueError(f"unsupported action reached executor: {action}")


def run_plan(model, data, configuration, plan, on_event=None):
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
        label = action if action == "drawer_open" else f"pickup({step['object']}, {step['arm']})"
        emit({"type": "step_start", "index": i, "action": label, "reason": step.get("reason", "")})

        success = False
        attempts = []
        for attempt in range(1, MAX_RETRIES + 1):
            prim.reset_to_neutral(model, data)  # includes the object pre-settle
            t0 = time.time()
            ok, state = _run_step(model, data, configuration, step)
            elapsed = time.time() - t0
            tag = "CONFIRM" if ok else "DISPUTE"
            attempts.append({"attempt": attempt, "tag": tag, "state": state, "elapsed_s": round(elapsed, 2)})
            emit({"type": "attempt_result", "index": i, "action": label,
                  "attempt": attempt, "tag": tag, "state": state})
            if ok:
                success = True
                break

        step_results.append({"action": label, "success": success, "attempts": attempts})
        emit({"type": "step_result", "index": i, "action": label, "success": success,
              "attempts_used": len(attempts)})

        if not success:
            result = {"overall": "FAILED",
                       "reason": f"step {i} ('{label}') DISPUTEd after {MAX_RETRIES} attempts",
                       "steps": step_results}
            emit({"type": "plan_result", **result})
            return result

    result = {"overall": "SUCCESS", "reason": "", "steps": step_results}
    emit({"type": "plan_result", **result})
    return result


def new_sim(scene_path: str = SCENE_PATH):
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    prim.reset_to_neutral(model, data)
    configuration = mink.Configuration(model)
    return model, data, configuration


if __name__ == "__main__":
    import sys
    from planner_so101 import make_plan

    cmd = " ".join(sys.argv[1:]) or "open the drawer, then pick up the plate with the right arm"
    print(f"Command: {cmd!r}")
    plan = make_plan(cmd)
    print("Plan:", plan, "\n")

    model, data, configuration = new_sim()
    result = run_plan(model, data, configuration, plan, on_event=lambda e: print(" event:", e))
    print("\nFinal result:", result["overall"], "-", result.get("reason", ""))