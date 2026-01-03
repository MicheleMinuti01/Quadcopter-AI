import os
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from env_A2C import droneEnv 

def train():
    # --- IMPOSTAZIONI ---
    TIMESTEPS = 100000  
    VERSION = "v0"       
    
    # --- 1. CONFIGURAZIONE PERCORSI ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Definiamo i percorsi relativi
    log_dir = os.path.join(script_dir, "logs_a2c/")
    checkpoint_dir = os.path.join(script_dir, "models_checkpoint/")
    final_model_dir = os.path.join(script_dir, "../models/") 

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)

    # --- 2. AMBIENTE ---
    env = droneEnv(render_every_frame=False, mouse_target=False)
    env = Monitor(env, log_dir)

    # --- 3. MODELLO ---
    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    # --- 4. CALLBACK ---
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path=checkpoint_dir,
        name_prefix="a2c_checkpoint"
    )

    # --- 5. ADDESTRAMENTO ---
    print(f"Inizio addestramento A2C ({VERSION}) per {TIMESTEPS} step...")
    model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)
    
    # --- 6. SALVATAGGIO FINALE ---
    filename = f"a2c_model_{VERSION}_{TIMESTEPS}_steps"
    final_path = os.path.join(final_model_dir, filename)
    
    model.save(final_path)
    print(f"Finito! Modello salvato in:")
    print(final_path)

if __name__ == "__main__":
    train()