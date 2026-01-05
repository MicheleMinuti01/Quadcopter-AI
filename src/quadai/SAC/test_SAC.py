import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from stable_baselines3 import SAC
from env_SAC import droneEnv
from quadai.utils.paths import get_models_dir 

def test():
    # Nome del file generato dal training
    FILENAME = "sac_model_v1_1000000_steps.zip"
    
    models_dir = get_models_dir()
    model_path = os.path.join(models_dir, FILENAME)

    if not os.path.exists(model_path):
        print(f"\nERRORE: Modello non trovato in: {model_path}")
        return

    print(f"Caricamento: {FILENAME}")

    # Ambiente
    env = droneEnv(render_every_frame=True, mouse_target=False)
    
    # Caricamento Agente
    model = SAC.load(model_path, env=env)

    # Valutazione
    for i in range(5):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            # Il render è gestito internamente dall'env se render_every_frame=True, 
            # ma lo chiamiamo qui per sicurezza se impostato a False nel costruttore
            env.render("yes")
            
        print(f"Episodio {i+1}: Reward = {episode_reward:.2f}")

if __name__ == "__main__":
    test()