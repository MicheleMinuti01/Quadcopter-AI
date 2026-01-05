import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from env_A2C import droneEnv 

from quadai.utils.paths import get_tuning_dir # type: ignore

def run_tuning():
    # --- IMPOSTAZIONI PER UN TUNING PROFONDO ---
    TEST_TIMESTEPS = 500000 
    ALGO = "A2C"
    
    # Parametri da combinare (Grid Search)
    learning_rates = [0.0005, 0.0007, 0.001]
    gammas = [0.99, 0.995]
    ent_coefs = [0.0, 0.01]
    n_steps_list = [5, 20]

    # --- GESTIONE CARTELLE ---
    # Usiamo la cartella risultati centralizzata per il file .txt
    results_out_dir = get_tuning_dir(ALGO) 
    results_file = os.path.join(results_out_dir, f"tuning_results_{ALGO}_FULL.txt")
    
    # Usiamo una cartella locale allo script per i log temporanei (monitor.csv)
    tmp_log_base = os.path.join(results_out_dir, "tmp_tuning")
    os.makedirs(tmp_log_base, exist_ok=True)

    # Scriviamo l'intestazione (sovrascrive se esiste)
    with open(results_file, "w") as f:
        f.write("LR, GAMMA, ENT_COEF, N_STEPS, MEAN_REWARD, MAX_REWARD\n")
    
    print(f"Inizio Tuning Approfondito {ALGO}.")
    print(f"I risultati finali saranno in: {results_file}")
    print(f"Test per configurazione: {TEST_TIMESTEPS} steps")
    print(f"Totale combinazioni: {len(learning_rates)*len(gammas)*len(ent_coefs)*len(n_steps_list)}")
    print("-" * 50)

    best_reward = -float('inf')
    best_config = None
    counter = 0

    # --- IL QUADRUPLO LOOP ---
    for lr in learning_rates:
        for gamma in gammas:
            for ent_coef in ent_coefs:
                for n_steps in n_steps_list:
                    counter += 1
                    config_name = f"lr{lr}_g{gamma}_ent{ent_coef}_n{n_steps}"
                    
                    # Cartella log specifica per questa combinazione
                    current_log_dir = os.path.join(tmp_log_base, config_name)
                    os.makedirs(current_log_dir, exist_ok=True)
                    
                    print(f"\n[{counter}/24] Testando: {config_name} ...")
                    
                    # Creazione ambiente
                    env = droneEnv(render_every_frame=False, mouse_target=False)
                    env = Monitor(env, current_log_dir)
                    
                    # Creazione modello
                    model = A2C(
                        "MlpPolicy", 
                        env, 
                        verbose=0, # Fondamentale: non vogliamo migliaia di log qui
                        learning_rate=lr, 
                        gamma=gamma,
                        ent_coef=ent_coef,
                        n_steps=n_steps
                    )
                    
                    # Training
                    model.learn(total_timesteps=TEST_TIMESTEPS)
                    
                    # Analisi risultati dal monitor.csv
                    try:
                        df = pd.read_csv(os.path.join(current_log_dir, "monitor.csv"), skiprows=1)
                        mean_r = df['r'].tail(100).mean()
                        max_r = df['r'].max()
                    except Exception as e:
                        print(f"Errore lettura log per {config_name}: {e}")
                        mean_r, max_r = -1000, -1000
                    
                    print(f"--> Risultato: Media {mean_r:.2f} | Max {max_r:.2f}")
                    
                    # Salvataggio su file TXT
                    with open(results_file, "a") as f:
                        f.write(f"{lr}, {gamma}, {ent_coef}, {n_steps}, {mean_r}, {max_r}\n")
                    
                    # Aggiornamento record
                    if mean_r > best_reward:
                        best_reward = mean_r
                        best_config = (lr, gamma, ent_coef, n_steps)

    print("\n" + "="*50)
    print("TUNING COMPLETO!")
    print(f"Miglior Configurazione trovata:")
    print(f"LR: {best_config[0]} | Gamma: {best_config[1]} | Entropia: {best_config[2]} | N_Steps: {best_config[3]}")
    print(f"Reward Medio Massimo: {best_reward:.2f}")
    print(f"Dati salvati in: {results_file}")
    print("="*50)

if __name__ == "__main__":
    run_tuning()