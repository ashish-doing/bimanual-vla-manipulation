<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Syne&weight=800&size=24&duration=3000&pause=1000&color=0071C5&center=true&vCenter=true&width=1100&lines=Bimanual+VLA+Manipulation+%E2%80%94+Setting+Up+a+Dinner+Table;Dual+SO-101+arms+%C2%B7+MuJoCo+%C2%B7+Camera+observations+%C2%B7+OpenVINO;LLM+planner+%2B+classical+IK%2C+not+a+trained+VLA+%E2%80%94+said+plainly;Intel+Physical+AI+Online+Challenge+%C2%B7+AI+Infra+Summit+Hackathon" alt="Bimanual VLA Manipulation" />

</div>

<p align="center">
  <a href="./ARCHITECTURE.md"><img src="https://img.shields.io/badge/📐%20ARCHITECTURE-DEEP%20DIVE-8a3ffc?style=for-the-badge" /></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-0071C5?style=for-the-badge" /></a>
  <a href="./so101_scene/STATUS.md"><img src="https://img.shields.io/badge/📍%20HONEST%20STATUS-READ%20FIRST-F5A623?style=for-the-badge" /></a>
  <a href="https://github.com/TheRobotStudio/SO-ARM100"><img src="https://img.shields.io/badge/Robot-SO--101%20(Apache--2.0)-1a1a1a?style=for-the-badge" /></a>
</p>

<p align="center">
  <strong>Intel Physical AI Online Challenge — "Bimanual VLA Manipulation with Multi-Modal Reasoning"</strong><br/>
  Challenge option: <strong>Setting Up a Dinner Table</strong> · AI Infra Summit Hackathon (lablab.ai + Kisaco Research)
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Simulator-MuJoCo-1a1a1a?style=flat-square" />
  <img src="https://img.shields.io/badge/Robot-Dual%20SO--101-1a1a1a?style=flat-square" />
  <img src="https://img.shields.io/badge/Planner-Groq%20(LLM)-1a1a1a?style=flat-square" />
  <img src="https://img.shields.io/badge/Deploy-Docker%20%2B%20Render-1a1a1a?style=flat-square" />
  <img src="https://img.shields.io/badge/Edge-OpenVINO-0071C5?style=flat-square" />
</p>

---

## Read this first

This repo is mid-pivot. It started as a different (ALOHA-rig, drawer +
handoff) build for a different scope, then was corrected against the actual
official brief once it was read carefully — the real challenge requires dual
**SO-101** arms, **camera-based** reasoning, a **dinner-table** task, and
**OpenVINO on Intel Core Ultra**, none of which the original build touched.
This README describes the corrected, in-progress build. **The single most
important file in this repo right now is [`so101_scene/STATUS.md`](./so101_scene/STATUS.md)**
— it states exactly which task primitives are verified reliable, which have
a known unresolved regression, and what hasn't been started. Read it before
trusting any claim below.

## What this is

An end-to-end pipeline for two simulated SO-101 arms setting a dinner table
in MuJoCo: open a drawer, retrieve cutlery, pick up a plate and cup, hand
objects between arms, guided by natural-language commands and (in progress)
real camera-based scene observation, with inference optimized via OpenVINO
for Intel Core Ultra hardware.

**Framing, stated plainly for the same reason the original project stated it
plainly**: the planning layer is an LLM (Groq) producing a small, whitelisted
JSON action plan — not a trained VLA policy. Training a real SmolVLA/ACT/
Pi0.5 policy needs GPU-hours and demonstration data this project doesn't
have access to on the timeline available. What's built instead, and
disclosed as such rather than dressed up: classical IK execution + a
CONFIRM/DISPUTE verify/replan loop + (planned) a small vision model trained
on synthetic renders from the sim's own ground truth, genuinely converted to
OpenVINO IR and genuinely run for inference — real and small, not fake and
impressive-sounding.

## Status against the actual judging rubric

| # | Criterion | Points | Current state |
|---|---|---|---|
| 1 | End-to-End Task Completion & Bimanual Manipulation | 30 | Dual-SO-101 scene built, physically verified. **3 of 5 tasks are 10/10 reliable** (drawer-open, plate pickup, spoon pickup — both arms exercised) and Groq planner + verify/replan executor drive them end-to-end. Fork pickup and cup handoff (the bimanual hand-off itself) still fail a diagnosed-but-unresolved launch bug — see STATUS.md |
| 2 | VLA / Multi-Modal Reasoning | 20 | Real, working: `table_cam` renders, a small classifier (trained on synthetic sim renders, not hand-labeled) reads it and correctly detects drawer open/closed from the actual image — verified against ground truth (before/after a real drawer-open: p=0.013 → p=0.977, matches physics state). Scope is intentionally small (one binary fact from vision, not a general scene parser) — see "Framing" above |
| 3 | Robustness & Generalization (10 seeds) | 15 | **10/10 across all 3 reliable tasks**, under randomized object placement (±2cm), mass (±30%), and friction (±30%) per seed — see `so101_scene/randomization_harness.py` and its output `randomization_results.json` |
| 4 | OpenVINO & Intel Core Ultra Optimization | 20 | Real pipeline built and run: sklearn model → hand-built standard-ops ONNX graph (skl2onnx's default output wasn't OpenVINO-compatible) → OpenVINO IR → real inference, benchmarked at ~0.03ms latency / ~29k inferences/sec **on the dev machine's CPU** (13th Gen i7-13650HX — confirmed no Core Ultra, no NPU). Intel AI PC Cloud access for real Core Ultra numbers is the remaining piece — see `so101_scene/benchmark_openvino.py` |
| 5 | Technical Quality & Reproducibility | 10 | Clean, tested scene-build pipeline; every bug found this session is documented with its fix, not hidden; real reproducible scripts for data generation, training, conversion, benchmarking, and randomized evaluation |
| 6 | Innovation & Technical Demonstration | 5 | CONFIRM/DISPUTE verify/replan lineage carried over from prior projects' mission-verification and observability patterns, applied to a new domain; vision-grounded verification (checking the physics state AND a real trained classifier's read of the camera image) is a small but genuine multi-modal check, not just a label |

This table will be updated as work continues — treat any version of this
README as a snapshot, not a permanent scorecard.

## Architecture

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full diagram, component
reference, and data flow. Summary:

```
NL command + camera frame (table_cam)
   -> Understand (Groq LLM planner, whitelisted to 3 verified-reliable tasks)
   -> Plan (whitelisted JSON action list)
   -> Act (mink IK-driven primitives on dual SO-101, weld-based grasping)
   -> Verify (real MuJoCo state check — CONFIRM/DISPUTE; drawer-state also
      cross-checked by a real OpenVINO-run vision classifier reading the
      camera frame, not just the ground-truth qpos)
   -> Replan (bounded retries)
-> Robustness harness: 10 randomized seeds (placement/mass/friction),
   10/10 across all 3 reliable tasks -- so101_scene/randomization_harness.py
-> Optimize: small classifier trained on synthetic sim renders, converted to
   OpenVINO IR, benchmarked (~0.03ms / ~29k inf/sec on the dev CPU; Core
   Ultra numbers pending Intel AI PC Cloud access)
-> Dashboard: FastAPI + WebSocket live event log, Intel-themed frontend,
   wired to this pipeline (so101_scene/dashboard_server.py)
```

## Repo layout

```
bimanual-hackathon/
├── so101_scene/                  <- the actual current build (read STATUS.md)
│   ├── STATUS.md                  honest verified/unverified breakdown
│   ├── build_dinner_scene.py      builds the dual-SO101 dinner-table scene
│   ├── dinner_scene.xml           compiled scene (regenerate, don't hand-edit)
│   ├── primitives_so101.py        task primitives (move_to/grasp/verify/etc.)
│   ├── planner_so101.py           Groq planner, whitelisted to the 3 reliable tasks
│   ├── executor_so101.py          verify/replan loop for those tasks
│   ├── generate_vision_data.py    synthetic training data from sim ground truth
│   ├── train_vision_model.py      trains classifier, converts to OpenVINO IR
│   ├── perception.py              runs the OpenVINO model on a live camera frame
│   ├── benchmark_openvino.py      Intel inference benchmark (deliverable #3)
│   ├── randomization_harness.py   10-seed robustness eval (deliverable #2)
│   ├── dashboard_server.py        FastAPI/WebSocket server for this pipeline
│   ├── vision_model.onnx, vision_model_ir/   trained + converted model artifacts
│   ├── so101_new_calib.xml + assets/   vendored SO-101 model (Apache-2.0)
│   └── SO-ARM100_LICENSE
├── primitives.py, build_scene.py  <- earlier ALOHA-rig build (superseded scope,
│                                      kept for reference, not the submission target)
├── planner.py, executor.py, main.py   <- ALOHA-era Groq planner + verify/replan loop
├── dashboard/                     <- ALOHA-era FastAPI + WebSocket dashboard,
│                                      plus static/index_intel_redesign.html
│                                      (the redesigned frontend, served by
│                                      so101_scene/dashboard_server.py)
├── Dockerfile, render.yaml        <- deployment (headless MuJoCo + OpenVINO)
├── ARCHITECTURE.md
├── LICENSE
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/ashish-doing/bimanual-vla-manipulation
cd bimanual-vla-manipulation
conda activate bimanual   # or your equivalent env
pip install -r requirements.txt

cd so101_scene
python build_dinner_scene.py
# prints: nq=41 nbody=20 nu=13 neq=8  -- confirms the scene loaded correctly

# Vision + OpenVINO pipeline (only needs running once -- artifacts are committed):
python generate_vision_data.py   # renders 400 synthetic labeled frames
python train_vision_model.py     # trains + converts to OpenVINO IR
python benchmark_openvino.py     # real latency/throughput numbers on THIS machine

# Robustness evaluation (10 seeds):
python randomization_harness.py

# Full pipeline test (needs GROQ_API_KEY in .env):
python executor_so101.py "open the drawer, then pick up the plate with the right arm"

# Dashboard:
uvicorn dashboard_server:app --host 0.0.0.0 --port 8001
# open http://localhost:8001
```

Then see [`so101_scene/STATUS.md`](./so101_scene/STATUS.md) for the exact
next diagnostic to run on fork/cup before assuming they're reliable.

## Known limitations — disclosed honestly, not buried

- **No trained VLA/imitation-learning policy.** The brief lists SmolVLA/
  Pi0.5/ACT as candidate approaches; none is used. This is a stated,
  deliberate scope decision given available time and compute, not an
  oversight.
- **No Intel Core Ultra hardware owned.** Local machine is a 13th Gen Intel
  Core i7-13650HX (Raptor Lake, no NPU). OpenVINO benchmark numbers so far
  are real but CPU-only, on non-Core-Ultra hardware. Plan is free remote
  access via Intel's AI PC Cloud; if that access doesn't arrive in time,
  this limitation stays stated plainly in the submission, not hidden.
- **2 of 5 task primitives (fork pickup, cup handoff) are not yet
  reliable.** Both fail a diagnosed launch bug during transport/retreat —
  see STATUS.md. The demo and evaluation scope is honestly narrowed to the
  3 tasks that are 10/10 reliable (drawer-open, plate pickup, spoon
  pickup) rather than including a flaky bimanual hand-off.
- **The vision component is intentionally small.** One binary classifier
  (drawer open/closed) trained on synthetic renders, not a general scene
  parser. Scoped to what's genuinely achievable and genuinely run, not
  claimed at a scale that wasn't actually built.
- **`planner.py`/`executor.py` (repo root) are the superseded ALOHA-era
  versions** — the current pipeline is `so101_scene/planner_so101.py` and
  `so101_scene/executor_so101.py`.

## Lineage

This project's verify/replan discipline — treat every action as unproven
until checked against real state, log CONFIRM/DISPUTE, retry with a bound —
carries over from two earlier projects by the same author: a mission
verification pattern from an autonomous-survey UGV project, and
decision-visibility instincts from an agent-observability project. Same
instincts, applied to a new domain, not the same codebase.

## Author

**Ashish Kumar** — B.Tech ECE, IIIT Guwahati (Batch 2024–2028)

[![GitHub](https://img.shields.io/badge/GitHub-ashish--doing-181717?style=flat-square&logo=github)](https://github.com/ashish-doing)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-ashish--kumar-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/ashish-kumar-014aaa3b9)

## License

MIT — see [`LICENSE`](./LICENSE). The vendored SO-101 model files retain
their original Apache-2.0 license — see
[`so101_scene/SO-ARM100_LICENSE`](./so101_scene/SO-ARM100_LICENSE).