import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor 

from quadai.utils.paths import get_models_dir
from env_noisy_A2C import droneEnv

def test_stats():
    N_EPISODES = 1000
    VERSION = "v2_noise"
    TIMESTEPS = 5000000
    FILENAME = f"a2c_model_{VERSION}_{TIMESTEPS}_steps.zip"
    
    models_dir = get_models_dir()
    model_path = os.path.join(models_dir, FILENAME)

    if not os.path.exists(model_path):
        print(f"ERRORE: Modello non trovato: {model_path}")
        return

    print(f"--- TEST A2C NOISY ({N_EPISODES} ep) ---")
    
    env = droneEnv(
        render_every_frame=False,
        mouse_target=False,
        wind_enabled=True,
        sensor_noise_enabled=True
    )
    env = Monitor(env)
    
    model = A2C.load(model_path, env=env)
    
    print("Valutazione...")
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=N_EPISODES, deterministic=True, render=False
    )

    print("\n" + "="*40)
    print(f"RISULTATI TEST A2C NOISY")
    print("="*40)
    print(f"Reward Medio:      {mean_reward:.2f}")
    print(f"Deviazione Std:    {std_reward:.2f}")
    print("="*40)

if __name__ == "__main__":
    test_stats()