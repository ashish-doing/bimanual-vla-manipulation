"""
LLM planner for the SO-101 dinner-table task set.

Scope: all 5 verified-reliable tasks (10/10 each, confirmed in-session
including under randomization -- see so101_scene/STATUS.md):
  - "drawer_open"  -> primitives_so101.run_drawer_open_task
  - "pickup"       -> primitives_so101.run_pickup_task(arm, object)
      objects: "plate" (right arm), "spoon" (left arm), "fork" (left arm)
  - "handoff"      -> primitives_so101.run_handoff_task(object)
      objects: "cup" (right arm picks up, hands to left arm)

Model: Groq (same choice as the original build, still current as of Sep 2026).
"""
import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-20b"

VALID_ACTIONS = {"drawer_open", "pickup", "handoff"}
VALID_PICKUP = {  # object -> allowed arm, all verified 10/10 reliable
    "plate": "right",
    "spoon": "left",
    "fork": "left",
}
VALID_HANDOFF = {"cup"}  # right arm picks up, hands to left arm -- verified 10/10

SYSTEM_PROMPT = """You are a task planner for a dual-arm SO-101 dinner-table \
robot with a SPECIFIC, LIMITED set of verified-reliable skills. You do NOT \
control the robot directly -- you only produce a JSON plan that a downstream \
execution engine will run.

Available actions (ONLY these exist -- do not invent others):
  - "drawer_open": opens the drawer using the left arm. No parameters.
  - "pickup": picks up ONE object with ONE specific arm. Only these
    object/arm pairs are verified reliable and allowed:
      - object="plate", arm="right"
      - object="spoon", arm="left"
      - object="fork", arm="left"
  - "handoff": the right arm picks up an object and hands it to the left
    arm. Only allowed for:
      - object="cup"

Any other object (mug, glass) or wrong arm for an object, or a handoff of
anything other than the cup, is NOT supported yet.

Rules:
1. Read the user's natural language command.
2. Decide which of the available actions are being requested, in order.
3. If the command asks for something not on this exact list (pouring,
   arranging, a handoff of a non-cup object, anything else), do NOT invent
   a plan for it. Set "unsupported" to a short, honest description of what
   couldn't be planned and why.
4. If ambiguous or unrecognized, return an empty "steps" list and explain in
   "unsupported".
5. Respond with ONLY valid JSON, no markdown fences, no prose, matching this
   exact schema:

{
  "steps": [
    {"action": "drawer_open", "reason": "<short reason>"},
    {"action": "pickup", "object": "plate"|"spoon"|"fork", "arm": "right"|"left", "reason": "<short reason>"},
    {"action": "handoff", "object": "cup", "reason": "<short reason>"}
  ],
  "unsupported": "<string, empty if nothing unsupported>"
}
"""


class PlannerError(Exception):
    pass


def _get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise PlannerError(
            "GROQ_API_KEY not set. Copy .env.example to .env and fill it in, "
            "or `export GROQ_API_KEY=...` before running. Get a free key at "
            "https://console.groq.com/keys"
        )
    return Groq(api_key=api_key)


def make_plan(command: str, model_name: str = DEFAULT_MODEL) -> dict:
    """Turn a natural language command into a validated plan dict.

    Returns: {"steps": [...], "unsupported": "..."}
    Never raises on a bad/unsupported command. Raises PlannerError only on
    infra problems (missing key, malformed model output).
    """
    client = _get_client()
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": command},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    raw = completion.choices[0].message.content.strip()

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as e:
        raise PlannerError(f"Planner returned non-JSON output: {raw[:200]!r}") from e

    steps = plan.get("steps", [])
    unsupported = plan.get("unsupported", "")

    clean_steps = []
    for step in steps:
        action = step.get("action")
        if action == "drawer_open":
            clean_steps.append({"action": "drawer_open", "reason": step.get("reason", "")})
        elif action == "pickup":
            obj = step.get("object")
            arm = step.get("arm")
            if obj in VALID_PICKUP and arm == VALID_PICKUP[obj]:
                clean_steps.append({"action": "pickup", "object": obj, "arm": arm,
                                     "reason": step.get("reason", "")})
            else:
                unsupported = (
                    unsupported
                    + f" [dropped unsupported pickup: object={obj!r} arm={arm!r}]"
                ).strip()
        elif action == "handoff":
            obj = step.get("object")
            if obj in VALID_HANDOFF:
                clean_steps.append({"action": "handoff", "object": obj,
                                     "reason": step.get("reason", "")})
            else:
                unsupported = (
                    unsupported + f" [dropped unsupported handoff: object={obj!r}]"
                ).strip()
        else:
            unsupported = (unsupported + f" [dropped invalid action {action!r}]").strip()

    return {"steps": clean_steps, "unsupported": unsupported}


if __name__ == "__main__":
    import sys
    cmd = " ".join(sys.argv[1:]) or "open the drawer, then pick up the plate with the right arm"
    print(f"Command: {cmd!r}\n")
    print(json.dumps(make_plan(cmd), indent=2))