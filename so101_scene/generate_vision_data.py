"""
Generates labeled training data for the drawer-state vision classifier using
the sim's own ground truth -- no manual labeling needed, since the simulator
knows the true drawer position exactly.

This is deliberately small in scope: a real, small, genuinely-trained model
answering one real question ("is the drawer open?") from the camera image,
not a fabricated claim of a large VLA policy. See README.md's "Framing"
section for why this scope was chosen.

Usage: MUJOCO_GL=osmesa python generate_vision_data.py
Produces: vision_data.npz (images + labels)
"""
import os

# See perception.py for why this is conditional, not a hard default --
# forcing osmesa on a desktop with a real display + NVIDIA driver crashes.
if not os.environ.get("DISPLAY") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import mujoco

import primitives_so101 as prim

IMG_H, IMG_W = 48, 64  # downsampled -- keeps the classifier small and fast
N_SAMPLES = 400
SCENE_PATH = os.path.join(os.path.dirname(__file__), "dinner_scene.xml")


def render_downsampled(renderer, data, camera="table_cam"):
    renderer.update_scene(data, camera=camera)
    img = renderer.render()  # (H, W, 3) uint8 at the renderer's native size
    # Simple box-downsample via striding (no extra deps like PIL/cv2 needed)
    h, w = img.shape[:2]
    ys = np.linspace(0, h - 1, IMG_H).astype(int)
    xs = np.linspace(0, w - 1, IMG_W).astype(int)
    small = img[np.ix_(ys, xs)]
    return small


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    renderer = mujoco.Renderer(model, height=240, width=320)

    drawer_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    drawer_adr = model.jnt_qposadr[drawer_jid]

    rng = np.random.default_rng(0)
    images = []
    labels = []

    for i in range(N_SAMPLES):
        data = mujoco.MjData(model)
        prim.reset_to_neutral(model, data)

        # Randomize: drawer open/closed (the label), lighting via a random
        # ambient jitter is skipped here (no light body to jitter cheaply) --
        # instead randomize arm neutral pose slightly and drawer opening
        # amount so the classifier can't just memorize one exact rendering.
        is_open = rng.random() < 0.5
        opening = rng.uniform(0.075, 0.12) if is_open else rng.uniform(0.0, 0.015)
        data.qpos[drawer_adr] = opening
        # small random jitter on both arms' neutral pose for visual variety
        jitter = rng.uniform(-0.08, 0.08, size=5)
        for side in ("left", "right"):
            for jname, base, dj in zip(prim.ARM_JOINT_NAMES[side], prim.NEUTRAL_ARM, jitter):
                jadr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                data.qpos[model.jnt_qposadr[jadr]] = base + dj
        mujoco.mj_forward(model, data)

        img = render_downsampled(renderer, data)
        images.append(img)
        labels.append(1 if is_open else 0)

        if (i + 1) % 50 == 0:
            print(f"  generated {i+1}/{N_SAMPLES}")

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int64)
    out_path = os.path.join(os.path.dirname(__file__), "vision_data.npz")
    np.savez_compressed(out_path, images=images, labels=labels)
    print(f"Saved {out_path}: images {images.shape}, labels {labels.shape}, "
          f"positive rate {labels.mean():.2f}")


if __name__ == "__main__":
    main()