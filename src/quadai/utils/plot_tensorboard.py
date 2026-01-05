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

def plot_tb_simple(log_dir, algo_name, title="TensorBoard Reward"):
    """Grafico semplice solo Reward."""
    output_dir = get_results_dir(algo_name)
    save_path = os.path.join(output_dir, "training_plot_TB.png")
    
    tb_file = find_tb_file(log_dir)
    if not tb_file:
        print(f"[TB] Nessun file events trovato in {log_dir}")
        return

    print(f"[TB] Leggendo: {os.path.basename(tb_file)}...")
    event_acc = EventAccumulator(tb_file)
    event_acc.Reload()
    
    tag = 'rollout/ep_rew_mean'
    if tag not in event_acc.Tags()['scalars']:
        return

    events = event_acc.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    plt.figure(figsize=(12, 6))
    plt.plot(steps, values, color='darkorange', linewidth=2, label='Mean Reward (TB)')
    plt.title(f"{title} ({algo_name})", fontsize=16)
    plt.xlabel("Step Totali", fontsize=12)
    plt.ylabel("Reward Medio", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300)
    print(f"[TB] Grafico semplice salvato: {save_path}")
    plt.close()

def plot_paper_style(log_dir, algo_name):
    """Grafico complesso 2x2 stile Paper."""
    output_dir = get_results_dir(algo_name)
    save_path = os.path.join(output_dir, f"{algo_name}_paper_plots.png")

    tb_file = find_tb_file(log_dir)
    if not tb_file:
        return

    event_acc = EventAccumulator(tb_file)
    event_acc.Reload()
    tags = event_acc.Tags()['scalars']

    metrics = {
        "Mean Reward": "rollout/ep_rew_mean",
        "Mean Episode Length": "rollout/ep_len_mean",
        "Entropy Loss": "train/entropy_loss",
        "Value Loss": "train/value_loss"
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Analisi Performance: {algo_name}", fontsize=20, weight='bold')
    axes_flat = axes.flatten()

    for i, (title, tag) in enumerate(metrics.items()):
        ax = axes_flat[i]
        if tag in tags:
            events = event_acc.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            color = 'tab:blue'
            if "Loss" in title: color = 'tab:red'
            if "Length" in title: color = 'tab:green'
            
            ax.plot(steps, values, color=color, linewidth=2)
            ax.set_title(title, fontsize=14, weight='bold')
            ax.grid(True, linestyle='--', alpha=0.7)
            if "Reward" in title:
                ax.axhline(y=0, color='black', alpha=0.5)
        else:
            ax.text(0.5, 0.5, "N/A", ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    print(f"[TB] Grafico Paper salvato: {save_path}")
    plt.close()