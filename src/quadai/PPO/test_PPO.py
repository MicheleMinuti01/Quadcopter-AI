import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from stable_baselines3 import PPO
from env_PPO import droneEnv

from quadai.utils.paths import get_models_dir # type: ignore

def test():
    
    FILENAME = "ppo_model_v1_4000000_steps.zip"
    
    models_dir = get_models_dir()
    model_path = os.path.join(models_dir, FILENAME)

    if not os.path.exists(model_path):
        print(f"\nERRORE: Non trovo il modello finale!")
        print(f"Ho cercato in: {model_path}")
        return

    print(f"Caricamento modello finale da: {model_path}")

    # Create and wrap the environment
    env = droneEnv(True, False)
    
    # Load the trained agent
    model = PPO.load(model_path, env=env)

    # Evaluate the agent
    for i in range(10):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        print("Episode reward", episode_reward)
        env.render("yes")

if __name__ == "__main__":
    test()