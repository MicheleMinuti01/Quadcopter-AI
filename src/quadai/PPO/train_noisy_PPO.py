import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from quadai.utils.paths import get_raw_logs_dir, get_models_dir, get_checkpoints_dir
from env_noisy_PPO import droneEnv

def train():
    ALGO = "PPO"
    VERSION = "v2_noise"
    TIMESTEPS = 4000000
    
    log_dir = get_raw_logs_dir(ALGO)            
    checkpoint_dir = get_checkpoints_dir(ALGO)
    models_dir = get_models_dir()           

    print(f"--- TRAINING PPO {VERSION} (NOISY) ---")

    env = droneEnv(
        render_every_frame=False,
        mouse_target=False,
        wind_enabled=True,
        wind_speed_max=0.04,
        sensor_noise_enabled=True
    )
    env = Monitor(env, os.path.join(log_dir, "monitor_noisy.csv"))

    # CORREZIONE: Rimosso tb_log_name da __init__
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,   
        learning_rate=0.0007,
        clip_range=0.2, 
        n_epochs=10
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path=checkpoint_dir,
        name_prefix=f"{ALGO.lower()}_checkpoint_{VERSION}"
    )

    # CORREZIONE: Aggiunto tb_log_name qui
    model.learn(
        total_timesteps=TIMESTEPS, 
        callback=checkpoint_callback,
        tb_log_name="PPO_NOISY"
    )
    
    filename = f"{ALGO.lower()}_model_{VERSION}_{TIMESTEPS}_steps"
    final_path = os.path.join(models_dir, filename)
    model.save(final_path)
    print(f"Modello Noisy salvato in: {final_path}")

if __name__ == "__main__":
    train()