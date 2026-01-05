import sys
import os

# Path setup per importare i moduli quadai
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# Importiamo le utility per i percorsi
from quadai.utils.paths import get_raw_logs_dir, get_models_dir, get_checkpoints_dir
from env_SAC import droneEnv

def train():
    # --- IMPOSTAZIONI ---
    ALGO = "SAC"
    VERSION = "v1" 
    # SAC è sample-efficient, spesso bastano meno step di PPO, ma teniamo 1M per sicurezza
    TIMESTEPS = 1000000 
    
    # --- PERCORSI AUTOMATICI ---
    log_dir = get_raw_logs_dir(ALGO)            # .../SAC/logs_sac
    checkpoint_dir = get_checkpoints_dir(ALGO)  # .../SAC/models_checkpoint
    models_dir = get_models_dir()               # .../models

    print(f"--- TRAINING SAC {VERSION} ---")
    print(f"Logs: {log_dir}")

    # --- AMBIENTE ---
    env = droneEnv(render_every_frame=False, mouse_target=False)
    env = Monitor(env, os.path.join(log_dir, "monitor.csv"))

    # --- MODELLO SAC ---
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=0.0003,
        buffer_size=50000,
        batch_size=256,
        ent_coef='auto', # SAC ottimizza l'entropia automaticamente
        tau=0.005
    )

    # --- CALLBACK ---
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=checkpoint_dir,
        name_prefix=f"{ALGO.lower()}_checkpoint_{VERSION}"
    )

    # --- TRAINING ---
    model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)
    
    # --- SALVATAGGIO ---
    filename = f"{ALGO.lower()}_model_{VERSION}_{TIMESTEPS}_steps"
    final_path = os.path.join(models_dir, filename)
    
    model.save(final_path)
    print(f"Finito! Modello salvato in: {final_path}")

if __name__ == "__main__":
    train()