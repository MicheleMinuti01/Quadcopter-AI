import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env_PPO import droneEnv 

def run_tuning():
    # --- IMPOSTAZIONI TUNING PPO ---
    TEST_TIMESTEPS = 500000 

    # GRID SEARCH SPECIFICA PER PPO
    learning_rates = [0.0003, 0.001] # 0.0003 è il default standard di PPO
    clip_ranges = [0.1, 0.2, 0.3]    # Il parametro magico del PPO
    n_epochs_list = [3, 10]          # Quante volte "ripassa" la lezione

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Cartella risultati dentro PPO/tune_result
    main_tune_dir = os.path.join(script_dir, "tune_result")
    os.makedirs(main_tune_dir, exist_ok=True)

    results_file = os.path.join(main_tune_dir, "tuning_results_PPO.txt")
    
    with open(results_file, "w") as f:
        f.write("LR, CLIP_RANGE, N_EPOCHS, MEAN_REWARD, MAX_REWARD\n")
    
    print(f"Inizio Tuning PPO.")
    print(f"Totale combinazioni: {len(learning_rates)*len(clip_ranges)*len(n_epochs_list)}")

    best_reward = -float('inf')
    best_config = None
    counter = 0

    for lr in learning_rates:
        for clip in clip_ranges:
            for epochs in n_epochs_list:
                counter += 1
                config_name = f"lr{lr}_clip{clip}_ep{epochs}"
                print(f"\n[{counter}] Testando: {config_name} ...")
                
                log_dir = os.path.join(main_tune_dir, "tmp_tuning", config_name)
                os.makedirs(log_dir, exist_ok=True)
                
                env = droneEnv(render_every_frame=False, mouse_target=False)
                env = Monitor(env, log_dir)
                
                # Modello PPO con parametri dinamici
                model = PPO(
                    "MlpPolicy", 
                    env, 
                    verbose=0, 
                    learning_rate=lr, 
                    clip_range=clip,
                    n_epochs=epochs
                )
                
                model.learn(total_timesteps=TEST_TIMESTEPS)
                
                # Analisi
                try:
                    df = pd.read_csv(os.path.join(log_dir, "monitor.csv"), skiprows=1)
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
    print(f"Reward: {best_reward}")

if __name__ == "__main__":
    run_tuning()