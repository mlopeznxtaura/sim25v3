#!/bin/bash
set -e

echo "[physics-native-ai-trainer] Setting up environment..."

pip install --upgrade pip
pip install -r requirements.txt

# Download MuJoCo assets
python -c "import mujoco; print('MuJoCo', mujoco.__version__, 'ready')"

# Init W&B
if [ -z "$WANDB_API_KEY" ]; then
  echo "Warning: WANDB_API_KEY not set. Run: export WANDB_API_KEY=<your-key>"
fi

echo "Setup complete."
