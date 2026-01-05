import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from quadai.utils.paths import get_raw_logs_dir, get_models_dir, get_checkpoints_dir
from env_noisy_SAC import droneEnv

def train():
    # --- IMPOSTAZIONI ---
    ALGO = "SAC"
    VERSION = "v1_noise"
    TIMESTEPS = 3300000
    
    # --- PERCORSI ---
    log_dir = get_raw_logs_dir(ALGO)            
    checkpoint_dir = get_checkpoints_dir(ALGO)
    models_dir = get_models_dir()           

    print(f"--- TRAINING SAC {VERSION} (NOISY) ---")

    # --- AMBIENTE (NOISY) ---
    env = droneEnv(
        render_every_frame=False,
        mouse_target=False,
        wind_enabled=True,
        wind_speed_max=0.04,
        sensor_noise_enabled=True
    )
    # Monitor specifico
    env = Monitor(env, log_dir)

    # --- MODELLO SAC ---
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=0.0003,
        ent_coef='auto',
        batch_size=256
    )

    # --- CALLBACK ---
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=checkpoint_dir,
        name_prefix=f"{ALGO.lower()}_checkpoint_{VERSION}"
    )

    # --- TRAINING ---
    # CORREZIONE: tb_log_name va messo QUI!
    model.learn(
        total_timesteps=TIMESTEPS, 
        callback=checkpoint_callback,
        tb_log_name="SAC_NOISY" 
    )
    
    filename = f"{ALGO.lower()}_model_{VERSION}_{TIMESTEPS}_steps"
    final_path = os.path.join(models_dir, filename)
    
    model.save(final_path)
    print(f"Modello Noisy salvato in: {final_path}")

if __name__ == "__main__":
    train()