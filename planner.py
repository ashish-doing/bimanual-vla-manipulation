"""
LLM planner: natural language command -> JSON plan of primitive task calls.

Scope (locked): only two tasks exist in this system.
  - "drawer_open"  -> primitives.run_drawer_open_task
  - "handoff"      -> primitives.run_handoff_task

Framing for the pitch: this is "LLM planner + classical/IK execution," NOT a
trained VLA. The LLM never touches the robot -- it only translates a natural
language request into a small, whitelisted action list. Everything physical
is done by the tested primitives in primitives.py.

Model: uses Groq (OpenAI-compatible API, very fast inference, generous free
tier). Default model is `openai/gpt-oss-20b`, which has confirmed native
JSON-mode support on Groq. If it 404s or gets deprecated by the time you run
this, check https://console.groq.com/docs/models (or `GET
https://api.groq.com/openai/v1/models` with your key) and update
DEFAULT_MODEL below -- Groq's free-tier lineup gets deprecated/rotated
fairly often, don't assume this name is still live without checking.
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # reads .env automatically -- no more manual `export` before every run

VALID_ACTIONS = {"drawer_open", "handoff"}
DEFAULT_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """You are a task planner for a bimanual robot with exactly two \
available skills. You do NOT control the robot directly -- you only produce a \
JSON plan that a downstream execution engine will run.

Available actions (ONLY these two exist -- do not invent others):
  - "drawer_open": opens a drawer using the left arm.
  - "handoff": right arm picks up a block and hands it off to the left arm.

Rules:
1. Read the user's natural language command.
2. Decide which of the two actions are being requested, and in what order.
3. If the command asks for something not supported (e.g. pouring, stacking, \
   anything besides drawer_open/handoff), do NOT invent a fake action. Instead \
   set "unsupported" to a short description of what couldn't be planned.
4. If the command is ambiguous or mentions nothing recognizable, return an \
   empty "steps" list and explain in "unsupported".
5. Respond with ONLY valid JSON, no markdown fences, no prose, matching this \
   exact schema:

{
  "steps": [{"action": "drawer_open" | "handoff", "reason": "<short reason>"}],
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

    Returns: {"steps": [{"action": "...", "reason": "..."}], "unsupported": "..."}
    Never raises on a bad/unsupported command -- that comes back as
    steps=[] (or a trimmed list) plus a filled "unsupported" string. Raises
    PlannerError only on infra problems (missing key, malformed model output).
    """
    client = _get_client()
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": command},
        ],
        response_format={"type": "json_object"},  # Groq's native JSON mode -- no fence-stripping needed
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
        if action not in VALID_ACTIONS:
            unsupported = (unsupported + f" [dropped invalid action '{action}']").strip()
            continue
        clean_steps.append({"action": action, "reason": step.get("reason", "")})

    # Known composition limitation (see README "Known limitations"): if handoff
    # runs before drawer_open, the left arm is still welded to the block when
    # drawer_open tries to command that same arm. Not fixed -- flagged instead,
    # since only single-task demos were built/verified end to end.
    actions_in_order = [s["action"] for s in clean_steps]
    if "handoff" in actions_in_order and "drawer_open" in actions_in_order:
        if actions_in_order.index("handoff") < actions_in_order.index("drawer_open"):
            unsupported = (
                unsupported
                + " [warning: handoff before drawer_open leaves the left arm holding"
                  " the block during drawer_open -- untested composition, not part of"
                  " the verified 10/10 scope]"
            ).strip()

    return {"steps": clean_steps, "unsupported": unsupported}


if __name__ == "__main__":
    import sys
    cmd = " ".join(sys.argv[1:]) or "open the drawer, then hand the block to the left arm"
    print(f"Command: {cmd!r}\n")
    print(json.dumps(make_plan(cmd), indent=2))
