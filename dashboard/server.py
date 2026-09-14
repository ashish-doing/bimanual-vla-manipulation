"""
FastAPI + WebSocket dashboard for the bimanual planner/executor.

Deliberately simple: a live event log + status badges, not a live 3D render
loop. That's a scope call, not an oversight -- see README "Dashboard scope"
for why. Run from the project root (not from inside dashboard/) so the
relative scene path in executor.py still resolves:

    uvicorn dashboard.server:app --host 0.0.0.0 --port 8000
"""
import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import executor
from planner import make_plan, PlannerError

app = FastAPI(title="Bimanual VLA Planner Dashboard")

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws/run")
async def ws_run(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "malformed message"})
                continue

            command = (msg.get("command") or "").strip()
            if not command:
                await websocket.send_json({"type": "error", "message": "empty command"})
                continue

            await websocket.send_json({"type": "planning", "command": command})
            try:
                plan = await asyncio.to_thread(make_plan, command)
            except PlannerError as e:
                await websocket.send_json({"type": "error", "message": str(e)})
                continue

            await websocket.send_json({"type": "plan", "plan": plan})

            if not plan["steps"]:
                await websocket.send_json({
                    "type": "plan_result",
                    "overall": "FAILED",
                    "reason": plan.get("unsupported") or "no valid steps",
                })
                continue

            loop = asyncio.get_event_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def on_event(event, _loop=loop, _queue=queue):
                _loop.call_soon_threadsafe(_queue.put_nowait, event)

            def run_sync():
                model, data, configuration = executor.new_sim()
                return executor.run_plan(model, data, configuration, plan, on_event=on_event)

            task = asyncio.create_task(asyncio.to_thread(run_sync))

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.2)
                    await websocket.send_json(event)
                except asyncio.TimeoutError:
                    if task.done():
                        break

            # Drain anything queued in the final instant before task.done() flipped.
            while not queue.empty():
                await websocket.send_json(queue.get_nowait())

            try:
                task.result()  # re-raise if run_sync threw
            except Exception as e:
                await websocket.send_json({"type": "error", "message": f"executor crashed: {e}"})

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.server:app", host="0.0.0.0", port=8000)
