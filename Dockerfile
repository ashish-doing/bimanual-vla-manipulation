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

EXPOSE 8000

CMD ["uvicorn", "dashboard.server:app", "--host", "0.0.0.0", "--port", "8000"]
