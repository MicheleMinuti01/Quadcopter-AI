import sys
import os

# --- PATH SETUP ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env_PPO import droneEnv 

# --- IMPORT CORRETTO ---
# Usiamo il nome esatto che hai nel tuo paths.py
from quadai.utils.paths import get_raw_tune_dir 

def run_tuning():
    
    # --- IMPOSTAZIONI TUNING PPO ---
    TEST_TIMESTEPS = 500000 
    ALGO = "PPO"

    # GRID SEARCH BASATA SUI TUOI RISULTATI (Totale 12 test)
    learning_rates = [0.0007, 0.001, 0.0015] 
    clip_ranges = [0.2, 0.3]         
    n_epochs_list = [10, 15]

    # --- GESTIONE CARTELLE ---
    # 1. Otteniamo il percorso target (quadai/PPO/tune_result)
    results_out_dir = get_raw_tune_dir(ALGO)
    
    # 2. IMPORTANTE: Creiamo la cartella se non esiste
    # (Il tuo paths.py non lo faceva automaticamente per questa funzione)
    os.makedirs(results_out_dir, exist_ok=True)
    
    # 3. Definiamo il file risultati completo
    results_file = os.path.join(results_out_dir, f"tuning_results_{ALGO}_FULL.txt")
    
    # Cartella temporanea per i log
    tmp_log_base = os.path.join(results_out_dir, "tmp_tuning")
    os.makedirs(tmp_log_base, exist_ok=True)
    
    # Intestazione file
    with open(results_file, "w") as f:
        f.write("LR, CLIP_RANGE, N_EPOCHS, MEAN_REWARD, MAX_REWARD\n")
    
    print(f"Inizio Tuning PPO.")
    print(f"Risultati salvati in: {results_file}")

    best_reward = -float('inf')
    best_config = None
    counter = 0

    for lr in learning_rates:
        for clip in clip_ranges:
            for epochs in n_epochs_list:
                counter += 1
                config_name = f"lr{lr}_clip{clip}_ep{epochs}"
                
                # Cartella temporanea specifica
                current_log_dir = os.path.join(tmp_log_base, config_name)
                os.makedirs(current_log_dir, exist_ok=True)
                
                print(f"\n[{counter}/12] Testando: {config_name} ...")
                
                env = droneEnv(render_every_frame=False, mouse_target=False)
                env = Monitor(env, current_log_dir)
                
                model = PPO(
                    "MlpPolicy", 
                    env, 
                    verbose=0, 
                    learning_rate=lr, 
                    clip_range=clip,
                    n_epochs=epochs
                )
                
                model.learn(total_timesteps=TEST_TIMESTEPS)
                
                try:
                    df = pd.read_csv(os.path.join(current_log_dir, "monitor.csv"), skiprows=1)
                    mean_r = df['r'].tail(100).mean()
                    max_r = df['r'].max()
                except:
                    mean_r, max_r = -1000, -1000
                
                print(f"--> Risultato: Media {mean_r:.2f} | Max {max_r:.2f}")
                
                with open(results_file, "a") as f:
                    f.write(f"{lr}, {clip}, {epochs}, {mean_r}, {max_r}\n")
                
                if mean_r > best_reward:
                    best_reward = mean_r
                    best_config = (lr, clip, epochs)

    print("\n" + "="*50)
    print("TUNING PPO COMPLETO!")
    print(f"Miglior Configurazione: LR={best_config[0]}, Clip={best_config[1]}, Epochs={best_config[2]}")
    print(f"Reward: {best_reward:.2f}")
    print(f"File salvato in: {results_file}")

if __name__ == "__main__":
    run_tuning()