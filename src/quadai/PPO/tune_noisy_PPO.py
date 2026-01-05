import sys
import os
import pandas as pd

# --- PATH SETUP ---
# Aggiungiamo la root del progetto al path per importare i moduli comuni
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# NOTA: Importiamo l'ambiente NOISY
from env_noisy_PPO import droneEnv 

# Importiamo il gestore dei percorsi
from quadai.utils.paths import get_tune_dir 

def run_tuning():
    
    # --- IMPOSTAZIONI TUNING PPO (NOISY) ---
    TEST_TIMESTEPS = 400000  # Un po' più lunghi perché col rumore serve più tempo per capire se impara
    ALGO = "PPO"

    # GRID SEARCH PARAMETERS
    # Testiamo parametri standard.
    # Nota: Col vento spesso serve un Learning Rate leggermente più basso per stabilità.
    learning_rates = [0.0003, 0.0007, 0.001] 
    clip_ranges = [0.2, 0.3]         
    n_epochs_list = [10, 15]

    # --- GESTIONE CARTELLE ---
    # 1. Otteniamo la cartella target: .../quadai/PPO/tune_result
    results_out_dir = get_tune_dir(ALGO)
    
    # 2. Definiamo il file risultati specifico per il NOISY
    results_file = os.path.join(results_out_dir, f"tuning_results_{ALGO}_NOISY.txt")
    
    # 3. Cartella temporanea per i log (monitor.csv)
    tmp_log_base = os.path.join(results_out_dir, "tmp_tuning")
    os.makedirs(tmp_log_base, exist_ok=True)
    
    # Intestazione file
    with open(results_file, "w") as f:
        f.write("LR, CLIP_RANGE, N_EPOCHS, MEAN_REWARD, MAX_REWARD\n")
    
    print(f"--- INIZIO TUNING PPO (NOISY ENVIRONMENT) ---")
    print(f"I risultati verranno salvati in: {results_file}")

    best_reward = -float('inf')
    best_config = None
    counter = 0
    total_tests = len(learning_rates) * len(clip_ranges) * len(n_epochs_list)

    for lr in learning_rates:
        for clip in clip_ranges:
            for epochs in n_epochs_list:
                counter += 1
                config_name = f"noisy_lr{lr}_clip{clip}_ep{epochs}"
                
                # Sottocartella specifica per questo test
                current_log_dir = os.path.join(tmp_log_base, config_name)
                os.makedirs(current_log_dir, exist_ok=True)
                
                print(f"\n[{counter}/{total_tests}] Testando Configurazione: {config_name}")
                
                # --- ISTANZIAMENTO AMBIENTE CON VENTO ---
                env = droneEnv(
                    render_every_frame=False, 
                    mouse_target=False,
                    wind_enabled=True,         # VENTO ATTIVO
                    wind_speed_max=0.04,       # Forza vento standard
                    sensor_noise_enabled=True  # RUMORE SENSORI ATTIVO
                )
                
                # Monitor fondamentale per leggere i reward
                env = Monitor(env, current_log_dir)
                
                # Creazione Modello (Architettura Default come richiesto)
                model = PPO(
                    "MlpPolicy", 
                    env, 
                    verbose=0, 
                    learning_rate=lr, 
                    clip_range=clip,
                    n_epochs=epochs
                )
                
                # Training
                model.learn(total_timesteps=TEST_TIMESTEPS)
                
                # Analisi Risultati (Leggiamo il CSV generato)
                try:
                    df = pd.read_csv(os.path.join(current_log_dir, "monitor.csv"), skiprows=1)
                    # Prendiamo la media degli ultimi 100 episodi per avere un dato stabile
                    mean_r = df['r'].tail(100).mean()
                    max_r = df['r'].max()
                except Exception as e:
                    print(f"Errore lettura log: {e}")
                    mean_r, max_r = -1000, -1000
                
                print(f"--> Risultato Noisy: Media {mean_r:.2f} | Max {max_r:.2f}")
                
                # Scrittura su file
                with open(results_file, "a") as f:
                    f.write(f"{lr}, {clip}, {epochs}, {mean_r}, {max_r}\n")
                
                # Aggiornamento Best
                if mean_r > best_reward:
                    best_reward = mean_r
                    best_config = (lr, clip, epochs)

    print("\n" + "="*50)
    print("TUNING PPO NOISY COMPLETATO!")
    print(f"Miglior Configurazione col Vento: LR={best_config[0]}, Clip={best_config[1]}, Epochs={best_config[2]}")
    print(f"Reward Medio Migliore: {best_reward:.2f}")
    print(f"Report completo: {results_file}")

if __name__ == "__main__":
    run_tuning()