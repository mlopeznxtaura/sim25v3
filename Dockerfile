FROM nvidia/cuda:12.3.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MUJOCO_GL=egl

RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl wget \
    libgl1-mesa-glx libglu1-mesa libglfw3 \
    libxrandr2 libxinerama-dev libxi-dev libxcursor-dev \
    ffmpeg colmap \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

COPY . .

ENV WANDB_API_KEY=""
ENV MLFLOW_TRACKING_URI="./mlruns"

ENTRYPOINT ["python3", "main.py"]
CMD ["--algo", "ppo", "--timesteps", "1000000"]
