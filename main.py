"""
CLI entry point: natural language command -> plan -> execution, printed to
the terminal. Use this to sanity-check the planner+executor before recording
the pitch video or wiring up the dashboard.

Usage:
    python main.py "open the drawer"
    python main.py "open the drawer, then hand off the block"
    python main.py                       # uses a default demo command
"""
import sys
import json

from planner import make_plan, PlannerError
from executor import new_sim, run_plan


def main():
    command = " ".join(sys.argv[1:]) or "open the drawer, then hand off the block"
    print(f"Command: {command!r}\n")

    try:
        plan = make_plan(command)
    except PlannerError as e:
        print(f"PLANNER ERROR: {e}")
        sys.exit(1)

    print("Plan:", json.dumps(plan, indent=2))
    if plan.get("unsupported"):
        print(f"\n(note: {plan['unsupported']})")
    if not plan["steps"]:
        print("\nNo runnable steps -- nothing to execute.")
        sys.exit(1)

    print("\nExecuting...\n")
    model, data, configuration = new_sim()
    result = run_plan(
        model, data, configuration, plan,
        on_event=lambda e: print(" ", e),
    )

    print(f"\n=== {result['overall']} ===")
    if result.get("reason"):
        print(result["reason"])
    sys.exit(0 if result["overall"] == "SUCCESS" else 1)


if __name__ == "__main__":
    main()
