import sys
import os
# Setup path per importare moduli quadai
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# Importiamo le funzioni di percorso corrette
from quadai.utils.paths import get_raw_logs_dir, get_models_dir, get_checkpoints_dir
from env_PPO import droneEnv 

def train():
    # --- IMPOSTAZIONI ---
    ALGO = "PPO"
    VERSION = "v1" 
    TIMESTEPS = 4000000
    
    # --- PERCORSI AUTOMATICI ---
    # Log: src/quadai/PPO/logs_ppo
    log_dir = get_raw_logs_dir(ALGO) 
    # Checkpoint: src/quadai/PPO/models_checkpoint
    checkpoint_dir = get_checkpoints_dir(ALGO)
    # Modello Finale: src/quadai/models
    models_dir = get_models_dir()           

    print(f"--- TRAINING PPO {VERSION} ---")
    print(f"Tensorboard Log: {log_dir}")
    print(f"Checkpoints:     {checkpoint_dir}")

    # --- AMBIENTE ---
    env = droneEnv(render_every_frame=False, mouse_target=False)
    # Monitor salva monitor.csv dentro logs_ppo
    env = Monitor(env, os.path.join(log_dir, "monitor.csv"))

    # --- MODELLO (Default Architecture) ---
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir, 
        learning_rate=0.0007,
        clip_range=0.2, 
        n_epochs=10
    )

    # --- CALLBACK ---
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path=checkpoint_dir,
        name_prefix=f"{ALGO.lower()}_checkpoint_{VERSION}"
    )

    # --- ESECUZIONE ---
    model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)
    
    # --- SALVATAGGIO FINALE ---
    filename = f"{ALGO.lower()}_model_{VERSION}_{TIMESTEPS}_steps"
    final_path = os.path.join(models_dir, filename)
    
    model.save(final_path)
    print(f"Finito! Modello salvato in: {final_path}")

if __name__ == "__main__":
    train()