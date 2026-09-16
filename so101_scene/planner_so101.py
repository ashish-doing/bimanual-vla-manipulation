"""
LLM planner for the SO-101 dinner-table task set.

Scope (locked to what's verified 10/10 reliable -- see so101_scene/STATUS.md):
  - "drawer_open"          -> primitives_so101.run_drawer_open_task
  - "pickup"               -> primitives_so101.run_pickup_task(arm, object)
      objects: "plate" (right arm only, verified), "spoon" (left arm only, verified)
  - "handoff" / "fork" pickup are NOT in the whitelist -- both still fail a
    known, diagnosed-but-unresolved launch bug (see STATUS.md). Listing them
    here would let the planner promise something the executor can't deliver.

Model: Groq (same choice as the original build, still current as of Sep 2026).
"""
import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-20b"

VALID_ACTIONS = {"drawer_open", "pickup"}
VALID_PICKUP = {  # (object, allowed arm) -- only the verified-reliable pairs
    "plate": "right",
    "spoon": "left",
}

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
    Any other object (fork, cup, mug) or wrong arm for an object is NOT
    supported yet -- do not invent a plan for it.

Rules:
1. Read the user's natural language command.
2. Decide which of the available actions are being requested, in order.
3. If the command asks for something not on this exact list (handoff, cup,
   fork, pouring, arranging, anything else), do NOT invent a plan for it.
   Set "unsupported" to a short, honest description of what couldn't be
   planned and why (e.g. "handoff is not yet reliable, see project status").
4. If ambiguous or unrecognized, return an empty "steps" list and explain in
   "unsupported".
5. Respond with ONLY valid JSON, no markdown fences, no prose, matching this
   exact schema:

{
  "steps": [
    {"action": "drawer_open", "reason": "<short reason>"},
    {"action": "pickup", "object": "plate"|"spoon", "arm": "right"|"left", "reason": "<short reason>"}
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
                    + f" [dropped unsupported pickup: object={obj!r} arm={arm!r}"
                      f" -- only plate/right and spoon/left are verified reliable]"
                ).strip()
        else:
            unsupported = (unsupported + f" [dropped invalid action {action!r}]").strip()

    return {"steps": clean_steps, "unsuspported": unsupported}


if __name__ == "__main__":
    import sys
    cmd = " ".join(sys.argv[1:]) or "open the drawer, then pick up the plate with the right arm"
    print(f"Command: {cmd!r}\n")
    print(json.dumps(make_plan(cmd), indent=2))