import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from quadai.utils.paths import get_raw_logs_dir, get_models_dir, get_checkpoints_dir
from env_noisy_A2C import droneEnv

def train():
    # --- IMPOSTAZIONI ---
    ALGO = "A2C"
    VERSION = "v2_noise"
    TIMESTEPS = 5000000
    
    # --- PERCORSI ---
    log_dir = get_raw_logs_dir(ALGO)            
    checkpoint_dir = get_checkpoints_dir(ALGO)
    models_dir = get_models_dir()           

    print(f"--- TRAINING A2C {VERSION} (NOISY) ---")

    # --- AMBIENTE (NOISY) ---
    env = droneEnv(
        render_every_frame=False,
        mouse_target=False,
        wind_enabled=True,
        wind_speed_max=0.04,
        sensor_noise_enabled=True
    )
    env = Monitor(env, os.path.join(log_dir, "monitor_noisy.csv"))

    # --- MODELLO A2C ---
    model = A2C(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=0.0007,
        gamma=0.99,
        ent_coef=0.01, # Un po' di entropia aiuta col vento
        n_steps=20
    )

    # --- CALLBACK ---
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path=checkpoint_dir,
        name_prefix=f"{ALGO.lower()}_checkpoint_{VERSION}"
    )

    # --- TRAINING ---
    # Fondamentale: tb_log_name qui dentro!
    model.learn(
        total_timesteps=TIMESTEPS, 
        callback=checkpoint_callback,
        tb_log_name="A2C_NOISY"
    )
    
    filename = f"{ALGO.lower()}_model_{VERSION}_{TIMESTEPS}_steps"
    final_path = os.path.join(models_dir, filename)
    
    model.save(final_path)
    print(f"Modello Noisy salvato in: {final_path}")

if __name__ == "__main__":
    train()