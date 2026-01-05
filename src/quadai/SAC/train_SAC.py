"""
Train an SAC agent using sb3 on the droneEnv environment
"""

import os

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import wandb
from wandb.integration.sb3 import WandbCallback

from env_noisy_SAC import droneEnv

run = wandb.init(
    project="quadai",
    sync_tensorboard=True,
    monitor_gym=False,
)

# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = droneEnv(
    render_every_frame=False,
    mouse_target=False,
    wind_enabled=True,          # attiva vento
    wind_dir_min_deg=0.0,       # direzione media tra 0° e 360°
    wind_dir_max_deg=360.0,
    wind_speed_min=0.0,         # vento minimo
    wind_speed_max=0.04,        # vento massimo (prova 0.02–0.05)
    wind_update_every=30,       # ogni quanto cambia lentamente il vento
    wind_dir_rw_std_deg=2,      # quanto oscilla la direzione a ogni update
    wind_speed_rw_std=0.003,    # quanto oscilla l’intensità a ogni update
    sensor_noise_enabled=True,  # attiva rumore sensori
    # opzionale: se non metti questo, usa i default nel costruttore
    # sensor_noise_std=[0.01, 0.02, 0.01, 0.02, 0.01, 0.01, 0.02],
)
env = Monitor(env, log_dir)

# Create SAC agent
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=100000, save_path=log_dir, name_prefix="rl_model_v2"
)

# Train the agent
model.learn(
    total_timesteps=10000000,
    callback=[
        checkpoint_callback,
        WandbCallback(
            gradient_save_freq=100000,
            model_save_path=f"models/{run.id}",
            model_save_freq=100000,
            verbose=2,
        ),
    ],
)
