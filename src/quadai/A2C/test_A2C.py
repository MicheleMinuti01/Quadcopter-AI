import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from stable_baselines3 import A2C
from env_A2C import droneEnv
from quadai.utils.paths import get_models_dir

def test():
    # Cerca il modello V1 standard
    FILENAME = "a2c_model_v1_5000000_steps.zip"
    
    models_dir = get_models_dir()
    model_path = os.path.join(models_dir, FILENAME)

    if not os.path.exists(model_path):
        print(f"\nERRORE: Modello non trovato: {model_path}")
        return

    print(f"Caricamento: {FILENAME}")
    env = droneEnv(True, False) # Render=True
    model = A2C.load(model_path, env=env)

    for i in range(5):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            env.render("yes")
        print(f"Episodio {i+1}: Reward {episode_reward:.2f}")

if __name__ == "__main__":
    test()