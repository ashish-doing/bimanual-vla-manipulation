"""
Perception module: renders the table_cam frame and runs it through the
OpenVINO-converted drawer-state classifier, producing a real camera-grounded
fact ("is the drawer open?") that the executor can check against -- this is
the actual "reasons over camera observations" piece, scoped honestly small
(see README.md "Framing").
"""
import os

# Only force the headless OSMesa backend when there's no real display
# (Docker, CI, this project's own sandbox testing) -- forcing it on a
# desktop with a real X server and an NVIDIA driver causes a hard OpenGL
# crash (confirmed: 'NoneType' object has no attribute 'glGetError'),
# because OSMesa's software GL conflicts with the system's hardware GL.
# On a real desktop, MuJoCo's natural default backend (GLFW) already works
# fine without any of this -- confirmed by generate_vision_data.py, whose
# osmesa line was an accidental no-op (set after mujoco was already
# imported) and which still rendered correctly.
if not os.environ.get("DISPLAY") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import mujoco
import openvino as ov

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "vision_model_ir", "vision_model.xml")
IMG_H, IMG_W = 48, 64

_core = None
_compiled = None
_renderer_cache = {}


def _get_compiled_model(device="CPU"):
    global _core, _compiled
    if _compiled is None:
        _core = ov.Core()
        model = _core.read_model(MODEL_PATH)
        _compiled = _core.compile_model(model, device)
    return _compiled


def _downsample(img):
    h, w = img.shape[:2]
    ys = np.linspace(0, h - 1, IMG_H).astype(int)
    xs = np.linspace(0, w - 1, IMG_W).astype(int)
    return img[np.ix_(ys, xs)]


def observe_drawer_state(model, data, device="CPU"):
    """Render table_cam, run it through the OpenVINO classifier, return
    (is_open: bool, probability: float). This is real inference on a real
    rendered frame -- not the ground-truth qpos value primitives.py's own
    verify function uses. The two are expected to usually agree; when they
    don't, that's the vision model being genuinely wrong, not a scripted
    number."""
    key = id(model)
    if key not in _renderer_cache:
        _renderer_cache[key] = mujoco.Renderer(model, height=240, width=320)
    renderer = _renderer_cache[key]

    renderer.update_scene(data, camera="table_cam")
    img = renderer.render()
    small = _downsample(img)
    x = small.reshape(1, -1).astype(np.float32) / 255.0

    compiled = _get_compiled_model(device)
    infer_request = compiled.create_infer_request()
    result = infer_request.infer({0: x})
    prob_open = float(list(result.values())[0].flatten()[0])
    return prob_open > 0.5, prob_open


if __name__ == "__main__":
    import primitives_so101 as prim

    model = mujoco.MjModel.from_xml_path(os.path.join(HERE, "dinner_scene.xml"))
    data = mujoco.MjData(model)
    prim.reset_to_neutral(model, data)

    is_open, prob = observe_drawer_state(model, data)
    print(f"Before opening: vision says drawer open={is_open} (p={prob:.3f})")
    prim.run_drawer_open_task(model, data, mink_config := __import__("mink").Configuration(model), verbose=False)
    is_open2, prob2 = observe_drawer_state(model, data)
    print(f"After opening:  vision says drawer open={is_open2} (p={prob2:.3f})")
    ground_truth = prim.verify_drawer_open(model, data)[0]
    print(f"Ground truth (physics state) says drawer open={ground_truth}")
    print(f"Vision matches ground truth: {is_open2 == ground_truth}")