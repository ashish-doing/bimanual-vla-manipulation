# SO-101 Dinner-Table Pivot — Status (this session)

Written at the point this session's work was handed off. Read this before
trusting anything in this folder — it says exactly what's proven and what
isn't, same discipline as the rest of this repo's documentation.

## What's real and verified (actually run, not just written)

- **Scene compiles correctly**: `nq=41, nbody=20, nu=13, neq=8`. Two
  independent SO-101 arms (vendored from `TheRobotStudio/SO-ARM100`,
  Apache-2.0) composed via MuJoCo's `MjSpec.attach_body` with `left_`/`right_`
  prefixes, facing each other across a table, plus a drawer, spoon, fork,
  plate, cup, and a fixed overhead `table_cam`.
- **Neutral pose is collision-free** — searched and confirmed 0 cross-arm
  contacts at `NEUTRAL_ARM = [0, -1.5, 1.8, 0.6, 0]` for both arms.
- **Four real physics bugs found and fixed this session**:
  1. Cutlery spawn position overlapped the drawer's own collision box,
     mechanically blocking it from opening (drawer got stuck at ~half its
     target travel). Fixed by moving spoon/fork clear of the drawer's swept
     footprint.
  2. The gripper stayed closed and physically stationary at the drawer
     handle while the drawer body slid open underneath/through it — the
     drawer's collision box rammed into the static gripper mesh and got
     stuck partway. Fixed by retracting the arm clear before actuating the
     drawer open (two-stage retreat: straight up, then to the side — a
     single big IK jump got stuck in a local minimum).
  3. Arm base separation (0.76m total) put the handoff zone ~0.44m from
     each base, well beyond SO-101's demonstrated ~0.27-0.30m reach (a much
     shorter arm than ALOHA's). Fixed by tightening to 0.48m total
     separation and repositioning all objects to fit.
  4. Moving the gripper frame exactly to a **tall** object's center (the
     cup) drove the gripper mesh into the object's bulk, generating a
     violent contact-force launch on settle. A thin object (plate) didn't
     show this. Fixed with a small upward grasp offset for the final
     approach point.
- **Each task has independently hit 100% reliability at some point in this
  session**: drawer-open, plate pickup (right arm), cup handoff
  (right→left), fork pickup (left arm), spoon pickup (left arm) — but see
  "Known unresolved issue" below, because they haven't all been *simultaneously*
  green in the same run.

## Known unresolved issue — read this before continuing

The last change (pre-settling free objects for 400 steps right after
`reset_to_neutral`, to fix spoon/fork chasing a still-falling target) fixed
spoon/fork but **broke cup_handoff and drawer_open**, which had both been
5/5 immediately before that change. This means there's a real remaining
interaction between how the objects settle and how drawer/cup specifically
behave — most likely:
- **Drawer**: was landing at `drawer_pos=0.0885` against a `min_open=0.09`
  threshold before the settle change — i.e. right at the edge, not
  fundamentally broken. Try loosening the threshold to `0.08`, or increase
  `open_drawer`'s `settle_steps` further, or increase the drawer actuator's
  `gainprm` (currently 500) for a stronger/faster pull, before assuming a
  new bug.
- **Cup**: likely tips or rolls slightly differently now that it gets 400
  extra steps to settle before the handoff sequence starts, changing its
  resting orientation/position enough to break the grasp-offset fix from
  bug #4 above. Check the cup's actual position/orientation right before
  `run_handoff_task` starts under the new settle, the same way the spoon's
  drift was diagnosed earlier in this session (print `data.xpos`/`data.xquat`
  for the object right after `reset_to_neutral` returns).

**Do not assume all 5 tasks are simultaneously reliable until you've run
all 5 in the same test pass and seen all 5 pass together.** Every number in
this file is from an individual test of that one task in isolation.

## What's genuinely not started yet

- Vision perception (camera → scene-state) — `table_cam` exists and
  renders, nothing reads it yet.
- OpenVINO conversion/inference — zero lines written.
- Domain randomization harness (10 seeds, varying placement/friction/
  lighting) — zero lines written.
- Intel Core Ultra benchmark script.
- Dashboard update to show the camera feed.
- README/ARCHITECTURE/LICENSE rewrite for this pivot.
- Everything above is still worth ~55+ of the 100 rubric points untouched.

## Files in this folder

- `so101_new_calib.xml` + `assets/` — vendored SO-101 arm model
  (`TheRobotStudio/SO-ARM100`, Apache-2.0 — see `SO-ARM100_LICENSE`).
- `build_dinner_scene.py` — builds `dinner_scene.xml` from scratch using
  MuJoCo's `MjSpec` API. Run `python build_dinner_scene.py` to regenerate.
- `dinner_scene.xml` — the compiled/exported scene (regenerable, not hand-
  edited — treat the `.py` builder as the source of truth).
- `primitives_so101.py` — task primitives ported from the ALOHA version:
  `reset_to_neutral`, `move_to`, `grasp_object`, `release`,
  `approach_and_grasp`, `run_drawer_open_task`, `run_pickup_task`,
  `run_handoff_task`, plus verify functions. This is the file with the
  unresolved cross-task issue described above.

## Immediate next steps, in order

1. Run all 5 tasks (drawer, plate, cup handoff, spoon, fork) in one test
   pass, back to back, and see which combination of settle-timing /
   threshold changes gets all 5 green together — not each in isolation.
2. Once primitives are genuinely 5/5-simultaneously reliable, port the
   `executor.py` verify/replan loop and `planner.py` (Groq) from the
   original ALOHA build to call these new task functions instead —
   the loop logic itself doesn't need to change, just which functions it
   calls.
3. Vision + OpenVINO (highest untouched point value — 40 pts combined
   across two rubric criteria): render `table_cam`, build a small trained
   classifier (not a fabricated "VLA policy" claim) on synthetic renders,
   convert to OpenVINO IR, run inference, feed into the planner.
4. Randomization harness across 10 seeds.
5. Dashboard, README, ARCHITECTURE.md, LICENSE, git push to the same
   `bimanual-vla-manipulation` repo.
