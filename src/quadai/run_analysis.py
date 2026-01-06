import os
import sys

# Hack per assicurarsi di trovare i moduli se lanciato da terminale
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.paths import get_raw_tune_dir, get_raw_logs_dir
from utils.plot_tuning import plot_tuning_comparison
# Importiamo SOLO la funzione paper style
from utils.plot_tensorboard import plot_paper_style 

def run_all_plots():
    print("--- INIZIO ANALISI E GENERAZIONE GRAFICI ---")

    # --- CONFIGURAZIONE ---
    run_versions = {
        # "A2C": "3",        # Cerca A2C_3
        # "PPO": "1",        # Cerca PPO_1
        # "PPO_NOISY": "1",   # Cerca PPO_NOISY_1
        "SAC_NOISY": "1"   # Cerca SAC_noisy_2
    }

    for key, version in run_versions.items():
        print(f"\n" + "="*50)
        print(f"Elaborazione: {key} (v{version})")
        print("="*50)

        # 1. SETUP VARIABILI
        if "_NOISY" in key:
            algo_base = key.replace("_NOISY", "")
            is_noisy = True
        else:
            algo_base = key
            is_noisy = False
            
        log_base_dir = get_raw_logs_dir(algo_base)
        tune_base_dir = get_raw_tune_dir(algo_base)

        # 2. GRAFICI TUNING
        if is_noisy:
            tune_file = f"tuning_results_{algo_base}_NOISY.txt"
        else:
            tune_file = f"tuning_results_{algo_base}_FULL.txt"
            
        tune_path = os.path.join(tune_base_dir, tune_file)
        plot_tuning_comparison(tune_path, algo_base)

        # 3. GRAFICI TENSORBOARD (SOLO PAPER STYLE)
        target_folder = f"{key}_{version}"
        tb_path = os.path.join(log_base_dir, target_folder)

        if os.path.exists(tb_path):
            try:
                # Genera SOLO il grafico 2x2 completo
                plot_paper_style(tb_path, algo_base)
            except Exception as e:
                print(f"   ERRORE TensorBoard: {e}")
        else:
            print(f"   [!] Cartella TensorBoard non trovata: {target_folder}")

    print("\n--- TUTTO FATTO. CONTROLLA LA CARTELLA RESULTS ---")

if __name__ == "__main__":
    run_all_plots()