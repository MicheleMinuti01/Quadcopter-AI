import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from env_noisy_A2C import droneEnv 
from quadai.utils.paths import get_raw_tune_dir 

def run_tuning():
    ALGO = "A2C"
    TEST_TIMESTEPS = 500000 
    
    # Grid ridotta per fare prima (i parametri migliori di solito sono simili)
    learning_rates = [0.0005, 0.0007]
    ent_coefs = [0.0, 0.01] # Entropia importante col vento
    n_steps_list = [20]

    # --- GESTIONE CARTELLE ---
    results_dir = get_raw_tune_dir(ALGO)
    results_file = os.path.join(results_dir, f"tuning_results_{ALGO}_NOISY.txt")
    tmp_log_base = os.path.join(results_dir, "tmp_tuning")
    os.makedirs(tmp_log_base, exist_ok=True)
    
    with open(results_file, "w") as f:
        f.write("LR, ENT_COEF, N_STEPS, MEAN_REWARD, MAX_REWARD\n")
    
    print(f"Inizio Tuning A2C NOISY. File: {results_file}")

    best_reward = -float('inf')
    best_config = None
    counter = 0

    for lr in learning_rates:
        for ent in ent_coefs:
            for n_steps in n_steps_list:
                counter += 1
                config_name = f"noisy_lr{lr}_ent{ent}_n{n_steps}"
                current_log_dir = os.path.join(tmp_log_base, config_name)
                os.makedirs(current_log_dir, exist_ok=True)
                
                print(f"\n[{counter}] Test Noisy: {config_name}")
                
                env = droneEnv(False, False, wind_enabled=True, sensor_noise_enabled=True)
                env = Monitor(env, current_log_dir)
                
                model = A2C(
                    "MlpPolicy", 
                    env, 
                    verbose=0, 
                    learning_rate=lr, 
                    ent_coef=ent,
                    n_steps=n_steps
                )
                
                model.learn(total_timesteps=TEST_TIMESTEPS)
                
                try:
                    df = pd.read_csv(os.path.join(current_log_dir, "monitor.csv"), skiprows=1)
                    mean_r = df['r'].tail(100).mean()
                    max_r = df['r'].max()
                except:
                    mean_r, max_r = -1000, -1000
                
                print(f"--> Media: {mean_r:.2f}")
                
                with open(results_file, "a") as f:
                    f.write(f"{lr}, {ent}, {n_steps}, {mean_r}, {max_r}\n")
                
                if mean_r > best_reward:
                    best_reward = mean_r
                    best_config = (lr, ent, n_steps)

    print("\n" + "="*50)
    print(f"Best Config Noisy: LR={best_config[0]}, Ent={best_config[1]}")

if __name__ == "__main__":
    run_tuning()