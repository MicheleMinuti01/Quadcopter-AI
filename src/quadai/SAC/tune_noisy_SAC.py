import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from env_noisy_SAC import droneEnv 
from quadai.utils.paths import get_raw_tune_dir 

def run_tuning():
    ALGO = "SAC"
    TEST_TIMESTEPS = 100000 

    # Parametri SAC
    learning_rates = [0.0001, 0.0003, 0.001]
    taus = [0.005, 0.01] 
    batch_sizes = [256] 

    # Gestione Cartelle
    results_dir = get_raw_tune_dir(ALGO)
    results_file = os.path.join(results_dir, f"tuning_results_{ALGO}_NOISY.txt")
    tmp_log_base = os.path.join(results_dir, "tmp_tuning")
    os.makedirs(tmp_log_base, exist_ok=True)
    
    with open(results_file, "w") as f:
        f.write("LR, TAU, BATCH_SIZE, MEAN_REWARD, MAX_REWARD\n")
    
    print(f"Inizio Tuning SAC NOISY.")

    best_reward = -float('inf')
    best_config = None
    counter = 0

    for lr in learning_rates:
        for tau in taus:
            for bs in batch_sizes:
                counter += 1
                config_name = f"noisy_lr{lr}_tau{tau}_bs{bs}"
                current_log_dir = os.path.join(tmp_log_base, config_name)
                os.makedirs(current_log_dir, exist_ok=True)
                
                print(f"\n[{counter}] Test Noisy: {config_name}")
                
                # AMBIENTE NOISY
                env = droneEnv(
                    render_every_frame=False, 
                    mouse_target=False,
                    wind_enabled=True,
                    sensor_noise_enabled=True
                )
                env = Monitor(env, current_log_dir)
                
                model = SAC(
                    "MlpPolicy", 
                    env, 
                    verbose=0, 
                    learning_rate=lr, 
                    tau=tau,
                    batch_size=bs,
                    ent_coef='auto'
                )
                
                model.learn(total_timesteps=TEST_TIMESTEPS)
                
                try:
                    df = pd.read_csv(os.path.join(current_log_dir, "monitor.csv"), skiprows=1)
                    mean_r = df['r'].tail(50).mean()
                    max_r = df['r'].max()
                except:
                    mean_r, max_r = -1000, -1000
                
                print(f"--> Media: {mean_r:.2f}")
                
                with open(results_file, "a") as f:
                    f.write(f"{lr}, {tau}, {bs}, {mean_r}, {max_r}\n")
                
                if mean_r > best_reward:
                    best_reward = mean_r
                    best_config = (lr, tau, bs)

    print("\n" + "="*50)
    print(f"Miglior Config Noisy: LR={best_config[0]}, Tau={best_config[1]}")

if __name__ == "__main__":
    run_tuning()