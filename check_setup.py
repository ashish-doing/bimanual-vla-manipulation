"""
Run this before touching anything else, and again right before recording the
pitch video. Checks the things that actually break demos: missing packages,
a missing/rebuilt scene file, no API key, and a quick live sanity run of both
tasks (1 trial each, not the full 10-trial suite -- this is a fast gate, not
the reliability test).

Usage: python check_setup.py
Exit code 0 = green light. Non-zero = something needs fixing before you build
on top of it.
"""
import importlib
import os
import sys

REQUIRED_PACKAGES = [
    "mujoco", "mink", "dm_control", "fastapi", "uvicorn",
    "websockets", "groq", "numpy",
]

SCENE_PATH = "mujoco_menagerie/aloha/custom_scene.xml"

EXPECTED_COUNTS = {"nq": 24, "nbody": 23, "nu": 15, "neq": 5}


def check_packages():
    print("== Checking packages ==")
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            print(f"  OK   {pkg}")
        except ImportError:
            print(f"  MISSING  {pkg}")
            missing.append(pkg)
    return missing


def check_scene():
    print("\n== Checking scene file ==")
    if not os.path.exists(SCENE_PATH):
        print(f"  MISSING  {SCENE_PATH}")
        print("  Run: python build_scene.py")
        return False

    import mujoco
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    actual = {"nq": model.nq, "nbody": model.nbody, "nu": model.nu, "neq": model.neq}
    ok = actual == EXPECTED_COUNTS
    status = "OK  " if ok else "MISMATCH"
    print(f"  {status} counts: {actual} (expected {EXPECTED_COUNTS})")
    return ok


def check_api_key():
    print("\n== Checking planner API key ==")
    key = os.environ.get("GROQ_API_KEY")
    if key:
        print("  OK   GROQ_API_KEY is set")
        return True
    print("  MISSING  GROQ_API_KEY (copy .env.example to .env and fill it in,")
    print("           or `export GROQ_API_KEY=...`). Only needed for planner.py")
    print("           and the dashboard -- primitives.py alone doesn't need it.")
    return False


def check_live_tasks():
    print("\n== Quick live sanity run (1 trial each, not the full reliability suite) ==")
    try:
        import mujoco
        import mink
        import primitives as prim
    except Exception as e:
        print(f"  SKIPPED  (import failed: {e})")
        return False

    model = mujoco.MjModel.from_xml_path(SCENE_PATH)

    data = mujoco.MjData(model)
    prim.reset_to_neutral(model, data)
    configuration = mink.Configuration(model)
    ok, state = prim.run_drawer_open_task(model, data, configuration, verbose=False)
    print(f"  drawer_open: {'CONFIRM' if ok else 'DISPUTE'} | {state}")

    data = mujoco.MjData(model)
    prim.reset_to_neutral(model, data)
    configuration = mink.Configuration(model)
    ok2, state2 = prim.run_handoff_task(model, data, configuration, verbose=False)
    print(f"  handoff:     {'CONFIRM' if ok2 else 'DISPUTE'} | {state2}")

    return ok and ok2


def main():
    missing = check_packages()
    scene_ok = check_scene() if not missing else False
    check_api_key()  # informational only, not a hard gate
    tasks_ok = check_live_tasks() if (not missing and scene_ok) else False

    print("\n== Summary ==")
    if missing:
        print(f"FAIL: missing packages: {missing}")
        print("  pip install " + " ".join(p.split('.')[0] for p in missing))
        sys.exit(1)
    if not scene_ok:
        print("FAIL: scene not built or doesn't match the expected working geometry.")
        sys.exit(1)
    if not tasks_ok:
        print("FAIL: a live task run DISPUTEd. Something regressed -- don't build on")
        print("      top of this until primitives.py passes on its own again.")
        sys.exit(1)

    print("PASS: environment, scene, and both tasks are green.")
    sys.exit(0)


if __name__ == "__main__":
    main()
