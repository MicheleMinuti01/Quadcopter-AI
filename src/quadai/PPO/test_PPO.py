import os
import time
from stable_baselines3 import PPO
from env_PPO import droneEnv

def test():
    # Nome del file PPO che avrai creato col train
    FILENAME = "ppo_model_v0_100000_steps.zip"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.normpath(os.path.join(script_dir, "..", "models", FILENAME))

    if not os.path.exists(model_path):
        print(f"Non trovo il modello in: {model_path}")
        return

    env = droneEnv(render_every_frame=True, mouse_target=False)
    
    # Caricamento PPO
    model = PPO.load(model_path, env=env)

    for ep in range(5):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            score += reward
            time.sleep(0.01)
        print(f"Score episodio: {score:.2f}")

if __name__ == "__main__":
    test()