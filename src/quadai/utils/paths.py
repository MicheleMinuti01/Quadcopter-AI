import os

def get_project_root():
    """Ritorna il percorso di src/quadai/"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return directory

def get_results_dir(algo_name):
    """Dove salviamo i GRAFICI finiti (Immagini/PDF)"""
    path = os.path.join(get_project_root(), "results", f"{algo_name}_results")
    return ensure_dir(path)

def get_raw_tune_dir(algo_name):
    """Dove si trovano i file .txt del tuning (es. quadai/A2C/tune_result)"""
    return os.path.join(get_project_root(), algo_name, "tune_result")

def get_raw_logs_dir(algo_name):
    """Dove si trovano i file monitor.csv e Tensorboard (es. quadai/A2C/logs_a2c)"""
    return os.path.join(get_project_root(), algo_name, f"logs_{algo_name.lower()}")

def get_models_dir():
    """Cartella dei modelli .zip"""
    return ensure_dir(os.path.join(get_project_root(), "models"))

def get_assets_dir():
    """Cartella: src/quadai/assets"""
    return os.path.join(get_project_root(), "assets")