import sys
import os
import numpy as np

# Setup dei percorsi
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from quadai.utils.paths import get_models_dir
from env_noisy_SAC import droneEnv
from stable_baselines3.common.monitor import Monitor 

def test_stats():
    # --- CONFIGURAZIONE ---
    N_EPISODES = 1000  # Numero di partite da testare
    VERSION = "v1_noise"
    TIMESTEPS = 4000000
    FILENAME = f"sac_model_{VERSION}_{TIMESTEPS}_steps.zip"
    
    # Percorso modello
    models_dir = get_models_dir()
    model_path = os.path.join(models_dir, FILENAME)

    if not os.path.exists(model_path):
        print(f"ERRORE: Non trovo il modello in {model_path}")
        return

    print(f"--- INIZIO TEST STATISTICO (Su {N_EPISODES} episodi) ---")
    print(f"Caricamento modello: {FILENAME}")

    # --- 1. Creiamo l'ambiente di test ---
    env = droneEnv(
    render_every_frame=False,
    mouse_target=False,
    wind_enabled=True,          # attiva vento
    wind_dir_min_deg=0.0,       # direzione media tra 0° e 360°
    wind_dir_max_deg=360.0,
    wind_speed_min=0.0,         # vento minimo
    wind_speed_max=0.04,        # vento massimo (prova 0.02–0.05)
    wind_update_every=30,       # ogni quanto cambia lentamente il vento
    wind_dir_rw_std_deg=2,      # quanto oscilla la direzione a ogni update
    wind_speed_rw_std=0.003,    # quanto oscilla l’intensità a ogni update
    sensor_noise_enabled=True,  # attiva rumore sensori
    # opzionale: se non metti questo, usa i default nel costruttore
    # sensor_noise_std=[0.01, 0.02, 0.01, 0.02, 0.01, 0.01, 0.02],
    )
    env = Monitor(env)
    
    # --- 2. Carichiamo il modello ---
    model = SAC.load(model_path, env=env)

    # --- 3. Valutazione ---
    print("Esecuzione in corso... (potrebbe volerci un minuto)")
    
    # evaluate_policy fa girare il modello per N episodi
    # deterministic=True significa che il modello usa l'azione migliore possibile (senza esplorazione casuale)
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=N_EPISODES, 
        deterministic=True, 
        render=False,
        return_episode_rewards=False
    )

    # --- 4. Risultati ---
    print("\n" + "="*40)
    print(f"RISULTATI TEST PPO NOISY ({N_EPISODES} EPISODI)")
    print("="*40)
    print(f"Reward Medio:      {mean_reward:.2f}")
    print(f"Deviazione Std:    {std_reward:.2f}")
    print("-" * 40)
    
    # Interpretazione rapida
    if mean_reward > 500:
        print("Giudizio: ECCELLENTE (Vola benissimo col vento)")
    elif mean_reward > 0:
        print("Giudizio: BUONO (Sopravvive ma raccoglie poco)")
    else:
        print("Giudizio: INSUFFICIENTE (Cade spesso)")
    print("="*40)

if __name__ == "__main__":
    test_stats()