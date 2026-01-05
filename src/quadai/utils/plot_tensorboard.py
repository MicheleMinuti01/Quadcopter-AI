import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from .paths import get_results_dir

def find_tb_file(log_dir):
    """Trova ricorsivamente il file events.out."""
    if not os.path.exists(log_dir):
        return None
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "events.out.tfevents" in file:
                return os.path.join(root, file)
    return None

def plot_paper_style(log_dir, algo_name):
    """
    Genera un'immagine con 4 grafici (Reward, Length, Entropy, Value Loss).
    Salva in: results/ALGO_results/NOME_RUN_paper_plots.png
    """
    output_dir = get_results_dir(algo_name)
    
    # 1. Ricaviamo il nome della run (es. PPO_NOISY_1)
    run_name = os.path.basename(os.path.normpath(log_dir))
    
    # 2. Definiamo il nome file univoco
    save_path = os.path.join(output_dir, f"{run_name}_paper_plots.png")

    # 3. Caricamento dati TensorBoard
    tb_file = find_tb_file(log_dir)
    if not tb_file:
        print(f"[TB] Nessun file events trovato in {log_dir}")
        return

    print(f"[TB] Elaborazione grafici per: {run_name}...")

    try:
        event_acc = EventAccumulator(tb_file)
        event_acc.Reload()
        tags = event_acc.Tags()['scalars']
    except Exception as e:
        print(f"[TB] Errore caricamento dati: {e}")
        return

    # Metriche da cercare
    metrics = {
        "Mean Reward": "rollout/ep_rew_mean",
        "Mean Episode Length": "rollout/ep_len_mean",
        "Entropy Loss": "train/entropy_loss",
        "Value Loss": "train/value_loss"
    }

    # Setup Grafico 2x2
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Analisi Performance: {run_name}", fontsize=20, weight='bold')
    axes_flat = axes.flatten()

    found_any = False
    for i, (title, tag) in enumerate(metrics.items()):
        ax = axes_flat[i]
        
        if tag in tags:
            found_any = True
            events = event_acc.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            # Colori diversi per metrica
            color = 'tab:blue'
            if "Loss" in title: color = 'tab:red'     # Rosso per le Loss
            if "Length" in title: color = 'tab:green' # Verde per la durata
            
            ax.plot(steps, values, color=color, linewidth=2)
            ax.set_title(title, fontsize=14, weight='bold')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Linea dello zero per il reward
            if "Reward" in title:
                ax.axhline(y=0, color='black', alpha=0.5)
        else:
            # Se manca la metrica (es. Entropy nei DQN)
            ax.text(0.5, 0.5, "N/A", ha='center', fontsize=12, color='gray')
            ax.set_title(title, fontsize=14, color='gray')
            ax.set_axis_off()

    if not found_any:
        print(f"[TB] Nessuna metrica trovata per {run_name}. Salto.")
        plt.close()
        return

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    print(f"[TB] Grafico Paper salvato: {os.path.basename(save_path)}")
    plt.close()