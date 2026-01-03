import os
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from env_A2C import droneEnv 

def train():
    # Configurazione percorsi
    log_dir = "tmp_A2C/"
    os.makedirs(log_dir, exist_ok=True)

    # Inizializza ambiente
    env = droneEnv(render_every_frame=False, mouse_target=False)
    env = Monitor(env, log_dir)

    # Crea agente A2C
    # Usiamo MlpPolicy per osservazioni numeriche (vettore di stato) 
    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    # Callback per salvataggi intermedi
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, save_path=log_dir, name_prefix="a2c_model_v1"
    )

    # Addestramento del modello 
    model.learn(total_timesteps=10000, callback=checkpoint_callback)
    
    # Salvataggio finale
    model.save("a2c_model_v0_10000_steps")

if __name__ == "__main__":
    train()