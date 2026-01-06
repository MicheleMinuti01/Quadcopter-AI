import sys
import os
import numpy as np
from math import sqrt

# Setup dei percorsi
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from quadai.utils.paths import get_models_dir

# ------ selezione dell'environment NOMINALE (senza vento/rumore)
from env_SAC import droneEnv
env_name = "env_SAC"


def test_stats():
    # --- CONFIGURAZIONE ---
    N_EPISODES = 500  # usa 100 per debug veloce, 500-1000 per numeri “seri”
    VERSION = "v1_noise"
    TIMESTEPS = 3300000
    FILENAME = f"sac_model_{VERSION}_{TIMESTEPS}_steps.zip"

    # Percorso modello
    models_dir = get_models_dir()
    model_path = os.path.join(models_dir, FILENAME)

    if not os.path.exists(model_path):
        print(f"ERRORE: Non trovo il modello in {model_path}")
        return

    print(f"--- INIZIO TEST STATISTICO SAC (Su {N_EPISODES} episodi) ---")
    print(f"Caricamento modello: {FILENAME}")

    # --- 1. Ambiente di test ---
    env = droneEnv(
        render_every_frame=False,
        mouse_target=False,
    )
    env = Monitor(env)

    # --- 2. Carichiamo il modello SAC ---
    model = SAC.load(model_path, env=env)

    # --- 3. Reward medio con evaluate_policy ---
    print("Calcolo reward medio con evaluate_policy...")
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=N_EPISODES,
        deterministic=True,
        render=False,
        return_episode_rewards=False,
    )

    # --- 4. Run manuale per palloncini e crash ---
    print("Calcolo palloncini medi e % crash...")

    episode_rewards = []
    balloons_per_ep = []
    crashes = 0

    for ep in range(N_EPISODES):
        obs = env.reset()
        done = False
        ep_reward = 0.0

        inner_env = env.env  # droneEnv interno
        balloons = 0
        crashed = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward

            # distanza drone-target come nell'env
            x = inner_env.x
            y = inner_env.y
            xt = inner_env.xt
            yt = inner_env.yt
            dist = sqrt((x - xt) ** 2 + (y - yt) ** 2)

            # se reward dello step è molto positivo (~+100) => raccolto palloncino
            if reward > 50:
                balloons += 1

            # se l'episodio finisce e la distanza è > 1000, consideriamo crash
            if done and dist > 1000:
                crashed = True

        episode_rewards.append(ep_reward)
        balloons_per_ep.append(balloons)
        if crashed:
            crashes += 1

    episode_rewards = np.array(episode_rewards, dtype=np.float32)
    balloons_per_ep = np.array(balloons_per_ep, dtype=np.float32)
    crash_rate = crashes / N_EPISODES * 100.0

    # --- 5. Risultati ---
    print("\n" + "=" * 50)
    print(f"RISULTATI TEST SAC ({N_EPISODES} EPISODI)")
    print(f"Modello: {FILENAME}")
    print(f"Environment: {env_name}")
    print("=" * 50)
    print(f"Reward Medio: {mean_reward:.2f}")
    print(f"Reward Std: {std_reward:.2f}")
    print(f"Palloncini medi/episodio: {balloons_per_ep.mean():.2f}")
    print(f"% Crash: {crash_rate:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    test_stats()
