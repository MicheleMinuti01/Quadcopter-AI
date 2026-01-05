import sys
import os

# Aggiungi path per trovare quadai
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from quadai.utils.paths import get_raw_logs_dir, get_models_dir # type: ignore

from env_A2C import droneEnv 

def train():
    # --- IMPOSTAZIONI ---
    TIMESTEPS = 5000000  
    VERSION = "v2"
    ALGO = "A2C"       
    
# --- 1. CONFIGURAZIONE PERCORSI ---
 
    log_dir = get_raw_logs_dir(ALGO)           # src/quadai/A2C/logs_a2c
    models_dir = get_models_dir()          # src/quadai/models
    
    # Checkpoint lo mettiamo dentro i log o in una cartella dedicata gestita da paths
    checkpoint_dir = os.path.join(os.path.dirname(log_dir), "models_checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Log salvati in: {log_dir}")
    print(f"Modelli salvati in: {models_dir}")

# --- 2. AMBIENTE ---
    env = droneEnv(render_every_frame=False, mouse_target=False)
    env = Monitor(env, os.path.join(log_dir, "monitor.csv"))

    # --- 3. MODELLO ---
    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, 
            learning_rate=0.0007, gamma=0.99, ent_coef=0.0, n_steps=20)

    # --- 4. CALLBACK & TRAIN ---
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path=checkpoint_dir,
        name_prefix=f"{ALGO.lower()}_checkpoint_{VERSION}"
    )

    # --- 5. ADDESTRAMENTO ---
    print(f"Inizio addestramento A2C ({VERSION}) per {TIMESTEPS} step...")
    model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)
    
    # --- 6. SALVATAGGIO ---
    final_path = os.path.join(models_dir, f"{ALGO.lower()}_model_{VERSION}_{TIMESTEPS}_steps")
    model.save(final_path)
    print(f"Finito! Modello salvato in:")
    print(final_path)

if __name__ == "__main__":
    train()