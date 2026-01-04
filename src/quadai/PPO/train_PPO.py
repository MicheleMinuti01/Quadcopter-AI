import os
from stable_baselines3 import PPO  # <--- Cambio importante
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from env_PPO import droneEnv 

def train():
    # --- IMPOSTAZIONI ---
    TIMESTEPS = 100000 
    VERSION = "v0"
    
    # --- 1. CONFIGURAZIONE PERCORSI ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Percorsi specifici per PPO
    log_dir = os.path.join(script_dir, "logs_ppo/")
    checkpoint_dir = os.path.join(script_dir, "models_checkpoint/")
    final_model_dir = os.path.join(script_dir, "../models/") 

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)

    # --- 2. AMBIENTE ---
    env = droneEnv(render_every_frame=False, mouse_target=False)
    env = Monitor(env, log_dir)

    # --- 3. MODELLO PPO ---
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, clip_range=0.2)

    # --- 4. CALLBACK ---
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path=checkpoint_dir,
        name_prefix="ppo_checkpoint"
    )

    # --- 5. ADDESTRAMENTO ---
    print(f"Inizio addestramento PPO ({VERSION}) per {TIMESTEPS} step...")
    model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)
    
    # --- 6. SALVATAGGIO FINALE ---
    filename = f"ppo_model_{VERSION}_{TIMESTEPS}_steps"
    final_path = os.path.join(final_model_dir, filename)
    
    model.save(final_path)
    print(f"Finito! Modello PPO salvato in:")
    print(final_path)

if __name__ == "__main__":
    train()