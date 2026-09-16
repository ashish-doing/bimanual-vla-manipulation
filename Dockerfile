# MuJoCo needs real system libs even for headless/offscreen use (GL + OSMesa).
# A plain buildpack deploy (no apt access) will fail on import mujoco with a
# GLFW/GL error -- that's the #1 reason "it works locally, 500s on Render"
# happens with this stack. Docker deploy sidesteps that by installing them
# explicitly below. Test this image locally before trusting it on deploy day:
#   docker build -t bimanual . && docker run -p 8000:8000 --env-file .env bimanual

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglfw3 \
    libosmesa6-dev \
    libglew-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Headless rendering backend -- must be set before mujoco is imported.
ENV MUJOCO_GL=osmesa

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Scene must exist in the image -- build_scene.py needs mujoco_menagerie
# cloned first. Do that at image build time so the container doesn't need
# network access (or a live git clone) at startup.
RUN git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git \
    && python build_scene.py --headless-build

# SO-101 dinner-table scene (so101_scene/) -- assets are already vendored in
# the repo (TheRobotStudio/SO-ARM100, Apache-2.0), so no extra clone needed.
# Also generates the synthetic vision training data and trains + converts
# the OpenVINO drawer-state classifier at build time, so the image is fully
# self-contained at startup (no training happening on first request).
RUN cd so101_scene \
    && python build_dinner_scene.py \
    && python generate_vision_data.py \
    && python train_vision_model.py

EXPOSE 8000
EXPOSE 8001

# Default: the original ALOHA-era dashboard (proven deployed and working).
# To run the newer SO-101 dashboard instead, override the command (it must
# run from so101_scene/ since its imports are bare module names, not a
# package path):
#   docker run -p 8001:8001 --env-file .env bimanual \
#     sh -c "cd so101_scene && uvicorn dashboard_server:app --host 0.0.0.0 --port 8001"
CMD ["uvicorn", "dashboard.server:app", "--host", "0.0.0.0", "--port", "8000"]