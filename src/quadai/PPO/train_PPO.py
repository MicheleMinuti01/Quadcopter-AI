import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from quadai.utils.paths import get_raw_logs_dir, get_models_dir # type: ignore
from env_PPO import droneEnv 

def train():
    # --- IMPOSTAZIONI ---
    TIMESTEPS = 4000000
    VERSION = "v1"
    ALGO = "PPO"
    
    # --- 2. CONFIGURAZIONE PERCORSI ---
    log_dir = get_raw_logs_dir(ALGO)            
    models_dir = get_models_dir()           
    
    # I checkpoint li mettiamo in una sottocartella dei log per pulizia
    checkpoint_dir = os.path.join(os.path.dirname(log_dir), "models_checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Log PPO salvati in: {log_dir}")
    print(f"Modelli salvati in: {models_dir}")

    # --- 3. AMBIENTE ---
    env = droneEnv(render_every_frame=False, mouse_target=False)
    env = Monitor(env, os.path.join(log_dir, "monitor.csv"))

    # --- 4. MODELLO PPO ---
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, learning_rate=0.0007,clip_range=0.2, n_epochs=10)

    # --- 5. CALLBACK ---
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path=checkpoint_dir,
        name_prefix=f"{ALGO.lower()}_checkpoint"
    )

    # --- 6. TRAINING ---
    print(f"Inizio addestramento PPO ({VERSION})...")
    model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)
    
    # --- 7. SALVATAGGIO FINALE ---
    filename = f"{ALGO.lower()}_model_{VERSION}_{TIMESTEPS}_steps"
    final_path = os.path.join(models_dir, filename)
    
    model.save(final_path)
    print(f"Finito! Modello salvato in: {final_path}")

if __name__ == "__main__":
    train()