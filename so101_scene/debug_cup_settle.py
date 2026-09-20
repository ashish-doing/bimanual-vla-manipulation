import numpy as np
import mujoco
import mink
from arm_backend import SimBackend
import primitives_so101 as prim
from randomization_harness import randomize_scene, SCENE_PATH

rng = np.random.default_rng(1)  # seed 1: a FAILING seed this time
model = mujoco.MjModel.from_xml_path(SCENE_PATH)
randomize_scene(model, mujoco.MjData(model), rng)
data = mujoco.MjData(model)
configuration = mink.Configuration(model)
backend = SimBackend(model, data, configuration)
backend.reset_to_neutral()

ok, dist = prim.approach_and_grasp(backend, "right", "cup", verbose=False)
print("right grasp ok:", ok, dist)
backend.move_to("right", prim.HANDOFF_ZONE, verbose=False)
backend.move_to("right", prim.HANDOFF_ZONE + np.array([0, 0, -0.05]), settle_steps=1200, verbose=False)
backend.release("right", "cup", verbose=False)
up_point = backend.get_site_pos("right_gripperframe") + np.array([0, 0, 0.18])
backend.move_to("right", up_point, settle_steps=1200, verbose=False)
backend.move_to("right", prim.RETREAT["right"], verbose=False)

print("Long trace on a FAILING seed (checking: does it ever stop, or roll forever):")
prev_pos = backend.get_object_pos("cup")
for i in range(100):  # 100 * 100 = 10000 steps -- well past the 3000 cap
    for _ in range(100):
        mujoco.mj_step(model, data)
    pos = backend.get_object_pos("cup")
    delta = np.linalg.norm(pos - prev_pos)
    print(f"  step {(i+1)*100:5d}  delta={delta:.5f}  pos={np.round(pos,4)}")
    prev_pos = pos
