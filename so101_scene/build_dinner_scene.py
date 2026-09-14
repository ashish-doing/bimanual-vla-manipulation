"""
Builds the dual-SO101 dinner-table scene for the Intel Physical AI challenge
("Bimanual VLA Manipulation with Multi-Modal Reasoning" / "Setting Up a
Dinner Table").

Uses MuJoCo's MjSpec API to compose two independent SO-101 arm instances
(vendored from TheRobotStudio/SO-ARM100, Apache-2.0) into one scene, facing
each other across a table, plus a drawer (spoon + fork), a plate, and a cup
-- all free bodies -- and a fixed overhead camera for the vision pipeline.

Run: python build_dinner_scene.py
Produces: dinner_scene.xml (also loadable directly via mujoco.MjModel)
"""
import mujoco
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_XML = os.path.join(HERE, "so101_new_calib.xml")

# ---- Table geometry ----
# Arm base separation was originally 0.76m (0.38 each side) -- discovered
# in-session to put the handoff zone ~0.44m from each base, well beyond
# SO-101's demonstrated ~0.27-0.30m reach (a much shorter arm than ALOHA's).
# Tightened so a shared handoff point sits within both arms' reach.
TABLE_HALF_X, TABLE_HALF_Y, TABLE_HEIGHT = 0.32, 0.26, 0.35
ARM_BASE_X = 0.24

# ---- Object placements (world frame, z relative to table surface = 0) ----
# Drawer footprint (half-sizes 0.07 x, 0.06 y) spans roughly x=[-0.20,-0.06],
# y=[-0.02,0.22] once open -- cutlery must sit clear of that box or it
# physically blocks the drawer sliding open (confirmed by a real
# fork<->drawer contact found in-session, same bug category as the ALOHA
# project's table-clipping).
DRAWER_POS = [-0.13, 0.16, 0.02]
SPOON_POS = [-0.24, 0.02, 0.05]
FORK_POS = [-0.24, -0.06, 0.05]
PLATE_POS = [0.08, -0.08, 0.02]
CUP_POS = [0.15, 0.06, 0.02]
HANDOFF_ZONE = [0.0, 0.0, 0.10]


def build_spec() -> mujoco.MjSpec:
    main = mujoco.MjSpec()
    main.compiler.degree = False
    main.meshdir = os.path.join(HERE, "assets")

    # ---- Two independent SO-101 arm instances ----
    arm_l = mujoco.MjSpec.from_file(ARM_XML)
    arm_r = mujoco.MjSpec.from_file(ARM_XML)

    frame_l = main.worldbody.add_frame(pos=[-ARM_BASE_X, 0, TABLE_HEIGHT])
    frame_l.attach_body(arm_l.body("base"), "left_", "")

    # Right arm rotated 180 deg about Z so both arms face the table center
    frame_r = main.worldbody.add_frame(
        pos=[ARM_BASE_X, 0, TABLE_HEIGHT], quat=[0, 0, 0, 1]
    )
    frame_r.attach_body(arm_r.body("base"), "right_", "")

    # ---- Table ----
    main.worldbody.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, TABLE_HEIGHT - 0.02],
        size=[TABLE_HALF_X, TABLE_HALF_Y, 0.02],
        rgba=[0.55, 0.38, 0.22, 1],
    )
    main.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0, 0, 0],
        size=[0, 0, 0.05],
        rgba=[0.25, 0.25, 0.28, 1],
    )
    main.worldbody.add_light(
        pos=[0, 0, 1.2], dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )

    # ---- Overhead scene camera (the "raw camera observation" input) ----
    main.worldbody.add_camera(
        name="table_cam",
        pos=[0, -0.55, TABLE_HEIGHT + 0.55],
        xyaxes=[1, 0, 0, 0, 0.75, 0.66],
    )

    z = TABLE_HEIGHT

    # ---- Drawer (actuator-driven slide, same disclosed pattern as before) ----
    drawer = main.worldbody.add_body(
        name="drawer", pos=[DRAWER_POS[0], DRAWER_POS[1], z + DRAWER_POS[2]]
    )
    drawer.add_joint(
        name="drawer_slide", type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, -1, 0], range=[0, 0.12], damping=4,
    )
    drawer.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.07, 0.06, 0.03],
        rgba=[0.5, 0.32, 0.15, 1], mass=0.25,
    )
    # Handle target sits 4cm proud of the drawer face (not flush with it) --
    # a flush target puts the gripper mesh in physical collision with the
    # drawer box during approach, which derails position-controlled settling
    # even though the IK solve itself converges. Confirmed by contact check
    # in-session (same category of bug as the ALOHA project's table-clipping).
    drawer.add_site(name="drawer_handle", pos=[0, -0.10, 0], size=[0.008])

    # ---- Spoon & fork (free bodies, sit near the drawer front) ----
    spoon = main.worldbody.add_body(
        name="spoon", pos=[SPOON_POS[0], SPOON_POS[1], z + SPOON_POS[2]]
    )
    spoon.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
    spoon.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.006, 0.04],
        rgba=[0.75, 0.75, 0.78, 1], mass=0.02, friction=[1, 0.5, 0.5],
    )

    fork = main.worldbody.add_body(
        name="fork", pos=[FORK_POS[0], FORK_POS[1], z + FORK_POS[2]]
    )
    fork.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
    fork.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.006, 0.04],
        rgba=[0.8, 0.8, 0.82, 1], mass=0.02, friction=[1, 0.5, 0.5],
    )

    # ---- Plate (free body) ----
    plate = main.worldbody.add_body(
        name="plate", pos=[PLATE_POS[0], PLATE_POS[1], z + PLATE_POS[2]]
    )
    plate.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
    plate.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.06, 0.006],
        rgba=[0.9, 0.9, 0.85, 1], mass=0.08, friction=[1, 0.5, 0.5],
    )

    # ---- Cup (free body) ----
    cup = main.worldbody.add_body(
        name="cup", pos=[CUP_POS[0], CUP_POS[1], z + CUP_POS[2]]
    )
    cup.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
    cup.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.025, 0.035],
        rgba=[0.2, 0.55, 0.85, 1], mass=0.04, friction=[1, 0.5, 0.5],
    )

    # ---- Drawer actuator ----
    gainprm = [500.0] + [0.0] * 9
    biasprm = [0.0, -500.0] + [0.0] * 8
    main.add_actuator(
        name="drawer_actuator", target="drawer_slide",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=gainprm,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=biasprm,
        ctrlrange=[0, 0.12],
    )

    # ---- Weld constraints for kinematic grasping (one per object per arm) ----
    for obj in ("spoon", "fork", "plate", "cup"):
        for side in ("left", "right"):
            main.add_equality(
                type=mujoco.mjtEq.mjEQ_WELD,
                name1=f"{side}_gripper", name2=obj,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                active=False, solref=[0.01, 1],
            )

    return main


def build_and_save(out_path: str = None) -> mujoco.MjModel:
    spec = build_spec()
    model = spec.compile()
    if out_path:
        with open(out_path, "w") as f:
            f.write(spec.to_xml())
    return model


if __name__ == "__main__":
    out = os.path.join(HERE, "dinner_scene.xml")
    model = build_and_save(out)
    print(f"Built {out}")
    print(f"nq={model.nq} nbody={model.nbody} nu={model.nu} neq={model.neq}")
    print("bodies:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)])
