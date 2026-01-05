import os
import pandas as pd
import matplotlib.pyplot as plt
from .paths import get_results_dir

def plot_training_csv(log_dir, algo_name, title="Training Progress"):
    """Genera grafico training da monitor.csv."""
    output_dir = get_results_dir(algo_name)
    save_path = os.path.join(output_dir, "training_plot_csv.png")

    file_path = os.path.join(log_dir, "monitor.csv")
    
    if not os.path.exists(file_path):
        # Silenzioso perché potremmo usare tensorboard invece
        return

    try:
        df = pd.read_csv(file_path, skiprows=1)
    except:
        df = pd.read_csv(file_path)

    x = df['l'].cumsum()
    y = df['r']
    
    window = 1000
    if len(y) < window: window = max(1, len(y) // 10)
    y_smooth = y.rolling(window=window).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, alpha=0.2, color='gray', label='Raw Reward')
    plt.plot(x, y_smooth, color='blue', linewidth=2, label=f'Media ({window} ep)')

    plt.title(f"{title} ({algo_name})", fontsize=16)
    plt.xlabel("Step Totali", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300)
    print(f"[CSV] Grafico salvato: {save_path}")
    plt.close()