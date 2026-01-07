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
    # key = nome cartella di TensorBoard (sotto logs_ALGO)
    # version = suffisso opzionale (es. "1" -> _1). Per curriculum lo lasciamo vuoto.
    run_versions = {
        # Esempi vecchi:
        # "A2C": "3",
        # "PPO": "1",
        # "PPO_NOISY": "1",
        # "SAC_NOISY": "1",

        #"PPO_CURR_PHASE0": "1",

         "A2C_CURR_PHASE0": "2",
         "A2C_CURR_PHASE1": "2",
         "A2C_CURR_PHASE2": "2",
        # "A2C_CURR_PHASE0": "1",
        # "A2C_CURR_PHASE1": "1",
        # "A2C_CURR_PHASE2": "1",
        # "SAC_CURR_PHASE0": "1",
        # "SAC_CURR_PHASE1": "1",
        # "SAC_CURR_PHASE2": "1",
    }

    for key, version in run_versions.items():
        print("\n" + "=" * 50)
        print(f"Elaborazione: {key} (v{version if version else 'no-suffix'})")
        print("=" * 50)

        # 1. SETUP VARIABILI BASE
        # per i run *_NOISY: algo_base = "PPO", "SAC", ecc.
        if "_NOISY" in key:
            algo_base = key.replace("_NOISY", "")
            is_noisy = True
        else:
            # per curriculum e run normali: algo_base è tutto prima di eventuali _CURR...
            # es. "PPO_CURR_PHASE0" -> "PPO"
            algo_base = key.split("_")[0]
            is_noisy = False

        log_base_dir = get_raw_logs_dir(algo_base)
        tune_base_dir = get_raw_tune_dir(algo_base)

        # 2. GRAFICI TUNING
        # I run curriculum in genere non hanno tuning, quindi li saltiamo
        if "CURR" in key:
            print("   [i] Nessun tuning per run curriculum, salto plot_tuning.")
        else:
            if is_noisy:
                tune_file = f"tuning_results_{algo_base}_NOISY.txt"
            else:
                tune_file = f"tuning_results_{algo_base}_FULL.txt"

            tune_path = os.path.join(tune_base_dir, tune_file)
            if os.path.exists(tune_path):
                try:
                    plot_tuning_comparison(tune_path, algo_base)
                except Exception as e:
                    print(f"   ERRORE plot_tuning: {e}")
            else:
                print(f"   [TUNING] File non trovato: {tune_path}")

        # 3. GRAFICI TENSORBOARD (SOLO PAPER STYLE)
        # per curriculum NON mettiamo suffisso, le cartelle sono esattamente:
        # logs_PPO/PPO_CURR_PHASE0, logs_PPO/PPO_CURR_PHASE1, ecc.
        if version:
            target_folder = f"{key}_{version}"
        else:
            target_folder = key

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
