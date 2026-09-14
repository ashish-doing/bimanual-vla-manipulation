"""
Run this once, after cloning mujoco_menagerie, to build custom_scene.xml.
Usage: python build_scene.py   (run from your project root, where
mujoco_menagerie/ already exists as a subfolder)
"""
import shutil
import os

SRC = "mujoco_menagerie/aloha/scene.xml"
DST = "mujoco_menagerie/aloha/custom_scene.xml"

if not os.path.exists(SRC):
    raise FileNotFoundError(
        f"{SRC} not found. Clone it first:\n"
        "  git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git"
    )

shutil.copy(SRC, DST)
with open(DST) as f:
    content = f.read()

# 1. Task objects: drawer (slide joint) + handoff block (free joint)
objects_block = """
    <!-- ===== Custom task objects added for hackathon project ===== -->
    <body name="drawer" pos="-0.28 0.20 0.03">
      <joint name="drawer_slide" type="slide" axis="0 -1 0" range="0 0.15" damping="4"/>
      <geom name="drawer_geom" type="box" size="0.06 0.05 0.03" rgba="0.6 0.35 0.15 1" mass="0.2" friction="1 0.5 0.5"/>
      <site name="drawer_handle" pos="0 -0.05 0" size="0.008" rgba="1 0 0 1"/>
    </body>

    <body name="handoff_block" pos="0.15 0.10 0.035">
      <joint name="handoff_block_free" type="free"/>
      <geom name="handoff_block_geom" type="box" size="0.02 0.02 0.02" rgba="0.1 0.6 0.9 1" mass="0.05" friction="1 0.5 0.5"/>
    </body>
    <!-- ===== End custom task objects ===== -->
  </worldbody>"""
content = content.replace("  </worldbody>", objects_block)

# 2. Weld constraints for kinematic grasping (must be added BEFORE the actuator
#    block below, since that block locates itself by searching for "<equality>")
eq_block = """
  <equality>
    <weld name="left_drawer_weld" body1="left/gripper_base" body2="drawer" active="false" solref="0.01 1"/>
    <weld name="right_block_weld" body1="right/gripper_base" body2="handoff_block" active="false" solref="0.01 1"/>
    <weld name="left_block_weld" body1="left/gripper_base" body2="handoff_block" active="false" solref="0.01 1"/>
  </equality>
</mujoco>"""
content = content.replace("</mujoco>", eq_block)

# 3. Drawer actuator (drives the drawer open/closed directly -- see README for why)
actuator_block = """  <actuator>
    <position name="drawer_actuator" joint="drawer_slide" kp="500" ctrlrange="0 0.15"/>
  </actuator>

  <equality>"""
content = content.replace("  <equality>", actuator_block, 1)

with open(DST, "w") as f:
    f.write(content)

print(f"Built {DST}")

import sys
import mujoco

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/aloha/custom_scene.xml")
data = mujoco.MjData(model)

print("Model loaded OK")
print("nq (dof):", model.nq)
print("nbody:", model.nbody)
print("nu (actuators):", model.nu)

# `--headless-build` (used by the Dockerfile, and any CI/build step with no
# display) skips the interactive viewer below and just confirms the model
# loads. Without this flag it launches the passive viewer as before, for
# local sanity-checking on your dev machine.
if "--headless-build" not in sys.argv:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
