import os

def get_project_root():
    """
    Ritorna la root del pacchetto sorgente: .../src/quadai/
    Basato sulla posizione di questo file: .../src/quadai/utils/paths.py
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return directory

# --- CARTELLE CONDIVISE ---
def get_models_dir():
    """Dove vanno i modelli finali: src/quadai/models"""
    return ensure_dir(os.path.join(get_project_root(), "models"))

def get_assets_dir():
    """Cartella assets: src/quadai/assets"""
    return os.path.join(get_project_root(), "assets")

def get_results_dir(algo_name):
    """Cartella per grafici/analisi finali: src/quadai/results/algo_results"""
    path = os.path.join(get_project_root(), "results", f"{algo_name}_results")
    return ensure_dir(path)

# --- CARTELLE SPECIFICHE PER ALGORITMO (es. PPO) ---

def get_algo_dir(algo_name):
    """Ritorna la cartella dell'algoritmo: src/quadai/PPO"""
    return os.path.join(get_project_root(), algo_name)

def get_raw_logs_dir(algo_name):
    """
    Dove salvare i log Tensorboard.
    Es: src/quadai/PPO/logs_ppo
    """
    base = get_algo_dir(algo_name)
    return ensure_dir(os.path.join(base, f"logs_{algo_name.lower()}"))

def get_checkpoints_dir(algo_name):
    """
    Dove salvare i checkpoint intermedi.
    Es: src/quadai/PPO/models_checkpoint
    """
    base = get_algo_dir(algo_name)
    return ensure_dir(os.path.join(base, "models_checkpoint"))

# *** RINOMINATA PER COERENZA CON GLI ALTRI SCRIPT ***
def get_raw_tune_dir(algo_name):
    """
    Dove salvare i risultati del tuning.
    Es: src/quadai/PPO/tune_result
    """
    base = get_algo_dir(algo_name)
    return ensure_dir(os.path.join(base, "tune_result"))