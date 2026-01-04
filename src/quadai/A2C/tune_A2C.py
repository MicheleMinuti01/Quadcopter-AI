import os
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from env_A2C import droneEnv 

def run_tuning():
    # --- IMPOSTAZIONI PER UN TUNING PROFONDO ---
    TEST_TIMESTEPS = 500000 

    # 2. I parametri da combinare (Grid Search)
    # Totale combinazioni: 3 * 2 * 2 * 2 = 24 test
    learning_rates = [0.0005, 0.0007, 0.001]
    gammas = [0.99, 0.995]
    ent_coefs = [0.0, 0.01]
    n_steps_list = [5, 20]

    # --- GESTIONE CARTELLE (MODIFICATA) ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Creiamo la cartella principale "tune_result" se non esiste
    main_tune_dir = os.path.join(script_dir, "tune_result")
    os.makedirs(main_tune_dir, exist_ok=True)

    # Il file dei risultati va dentro "tune_result"
    results_file = os.path.join(main_tune_dir, "tuning_results_A2C_FULL.txt")
    
    # Scriviamo l'intestazione
    with open(results_file, "w") as f:
        f.write("LR, GAMMA, ENT_COEF, N_STEPS, MEAN_REWARD, MAX_REWARD\n")
    
    print(f"Inizio Tuning Approfondito.")
    print(f"I risultati saranno salvati in: {main_tune_dir}")
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
                    print(f"\n[{counter}/24] Testando: {config_name} ...")
                    
                    # Creazione cartella per questa configurazione
                    log_dir = os.path.join(main_tune_dir, "tmp_tuning", config_name)
                    os.makedirs(log_dir, exist_ok=True)
                    
                    # Creazione ambiente
                    env = droneEnv(render_every_frame=False, mouse_target=False)
                    env = Monitor(env, log_dir)
                    
                    # Creazione modello con TUTTI i parametri
                    model = A2C(
                        "MlpPolicy", 
                        env, 
                        verbose=0, 
                        learning_rate=lr, 
                        gamma=gamma,
                        ent_coef=ent_coef,
                        n_steps=n_steps
                    )
                    
                    # Training
                    model.learn(total_timesteps=TEST_TIMESTEPS)
                    
                    # Analisi risultati
                    try:
                        df = pd.read_csv(os.path.join(log_dir, "monitor.csv"), skiprows=1)
                        # Prendiamo la media degli ultimi 100 episodi
                        mean_r = df['r'].tail(100).mean()
                        max_r = df['r'].max()
                    except:
                        mean_r, max_r = -1000, -1000
                    
                    print(f"--> Risultato: Media {mean_r:.2f} | Max {max_r:.2f}")
                    
                    # Salvataggio su file
                    with open(results_file, "a") as f:
                        f.write(f"{lr}, {gamma}, {ent_coef}, {n_steps}, {mean_r}, {max_r}\n")
                    
                    # Aggiornamento record
                    if mean_r > best_reward:
                        best_reward = mean_r
                        best_config = (lr, gamma, ent_coef, n_steps)

    print("\n" + "="*50)
    print("TUNING COMPLETO!")
    print(f"Miglior Configurazione salvata in {results_file}")
    print(f"LR: {best_config[0]}")
    print(f"Gamma: {best_config[1]}")
    print(f"Entropia: {best_config[2]}")
    print(f"N_Steps: {best_config[3]}")
    print(f"Reward Medio: {best_reward}")
    print("="*50)

if __name__ == "__main__":
    run_tuning()