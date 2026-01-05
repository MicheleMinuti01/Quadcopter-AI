import os
import pandas as pd
import matplotlib.pyplot as plt
from .paths import get_results_dir

def plot_tuning_comparison(txt_file, algo_name):
    """Genera il grafico a barre del tuning."""
    output_dir = get_results_dir(algo_name)
    save_path = os.path.join(output_dir, "tuning_comparison.png")

    if not os.path.exists(txt_file):
        print(f"[TUNING] File non trovato: {txt_file}")
        return

    try:
        df = pd.read_csv(txt_file)
        # Pulisce gli spazi nei nomi delle colonne
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"[TUNING] Errore lettura CSV: {e}")
        return

    if "MEAN_REWARD" not in df.columns:
        print("[TUNING] Colonna MEAN_REWARD mancante.")
        return

    df = df.sort_values(by="MEAN_REWARD", ascending=True)

    labels = []
    for index, row in df.iterrows():
        label = f"LR={row.get('LR', '?')}"
        if 'GAMMA' in df.columns: label += f"\nG={row['GAMMA']}"
        if 'ENT_COEF' in df.columns: label += f"\nEnt={row['ENT_COEF']}"
        if 'N_STEPS' in df.columns: label += f" N={row['N_STEPS']}"
        if 'CLIP_RANGE' in df.columns: label += f"\nClip={row['CLIP_RANGE']}"
        if 'N_EPOCHS' in df.columns: label += f" Ep={row['N_EPOCHS']}"
        labels.append(label)

    values = df['MEAN_REWARD']

    plt.figure(figsize=(14, max(6, len(values)*0.6)))
    bars = plt.barh(labels, values, color='skyblue')
    if len(bars) > 0:
        bars[-1].set_color('green')

    plt.title(f"Risultati Tuning {algo_name}", fontsize=16)
    plt.xlabel("Reward Medio", fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(values):
        plt.text(v, i, f" {v:.2f}", va='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[TUNING] Grafico salvato: {save_path}")
    plt.close()