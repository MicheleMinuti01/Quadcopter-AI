import os
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from env_A2C import droneEnv 

def run_tuning():
    # --- IMPOSTAZIONI OTTIMIZZATE PER ~30 MINUTI ---
    # Testiamo solo i parametri più impattanti
    learning_rates = [0.0001, 0.0007, 0.001] 
    gammas = [0.99, 0.95]
    TEST_TIMESTEPS = 200000 # Circa 5 minuti a test (a 680 FPS)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(script_dir, "tuning_results_A2C.txt")
    
    with open(results_file, "w") as f:
        f.write("LR, GAMMA, MEAN_REWARD, MAX_REWARD\n")
    
    print(f"Inizio Tuning Rapido (6 combinazioni, ~30 min totali)")

    best_reward = -float('inf')
    best_config = None

    for lr in learning_rates:
        for gamma in gammas:
            print(f"\n>>> TEST: LR={lr}, Gamma={gamma}")
            
            # Cartella log specifica per questo test
            log_dir = os.path.join(script_dir, f"tmp_tuning/lr{lr}_g{gamma}/")
            os.makedirs(log_dir, exist_ok=True)
            
            env = droneEnv(render_every_frame=False, mouse_target=False)
            env = Monitor(env, log_dir)
            
            # Creazione modello A2C
            model = A2C("MlpPolicy", env, verbose=0, learning_rate=lr, gamma=gamma)
            
            # Training breve
            model.learn(total_timesteps=TEST_TIMESTEPS)
            
            # Analisi risultati dal monitor.csv
            try:
                df = pd.read_csv(os.path.join(log_dir, "monitor.csv"), skiprows=1)
                # Media degli ultimi 50 episodi per vedere la stabilità finale
                mean_r = df['r'].tail(50).mean()
                max_r = df['r'].max()
            except:
                mean_r, max_r = -1000, -1000
            
            print(f"Risultato: Reward Medio {mean_r:.2f} | Max {max_r:.2f}")
            
            with open(results_file, "a") as f:
                f.write(f"{lr}, {gamma}, {mean_r}, {max_r}\n")
            
            if mean_r > best_reward:
                best_reward = mean_r
                best_config = (lr, gamma)

    print("\n" + "="*30)
    print("TUNING FINITO!")
    print(f"Usa questi nel train_A2C.py: LR={best_config[0]}, Gamma={best_config[1]}")
    print("="*30)

if __name__ == "__main__":
    run_tuning()