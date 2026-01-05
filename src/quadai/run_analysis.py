import os
from utils.paths import get_raw_tune_dir, get_raw_logs_dir
from utils.plot_tuning import plot_tuning_comparison
from utils.plot_training import plot_training_csv
from utils.plot_tensorboard import plot_tb_simple, plot_paper_style

def run_all_plots():
    print("--- INIZIO GENERAZIONE GRAFICI (Ordinata) ---")

    # --- CONFIGURAZIONE VERSIONI ---
    run_versions = {
        "A2C": "3",  # Cerca la cartella A2C_3
        "PPO": "1"   # Cerca la cartella PPO_1
    }

    # Iteriamo su chiave (algo) e valore (version)
    for algo, version in run_versions.items():
        print(f"\n>>> Elaborazione {algo} (Run {algo}_{version})...")
        
        # 1. Recuperiamo i percorsi
        tune_path = os.path.join(get_raw_tune_dir(algo), f"tuning_results_{algo}_FULL.txt")
        log_base = get_raw_logs_dir(algo) # es. logs_a2c
        
        # COSTRUZIONE DINAMICA DEL PERCORSO TENSORBOARD
        tb_log_path = os.path.join(log_base, f"{algo}_{version}") 

        # Controllo di sicurezza
        if not os.path.exists(tb_log_path):
            print(f"ATTENZIONE: La cartella {tb_log_path} non esiste! Salto i grafici TensorBoard.")
            continue
        
        # 2. Lanciamo i grafici
        # Grafico Tuning (TXT)
        plot_tuning_comparison(tune_path, algo)
        
        # Grafico Training (CSV)
        plot_training_csv(log_base, algo)
        
        # Grafici TensorBoard (se la cartella esiste)
        if os.path.exists(tb_log_path):
            plot_tb_simple(tb_log_path, algo)
            plot_paper_style(tb_log_path, algo)

    print("\n--- OPERAZIONE COMPLETATA ---")

if __name__ == "__main__":
    run_all_plots()