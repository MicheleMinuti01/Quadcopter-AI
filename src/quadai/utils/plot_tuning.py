import os
import pandas as pd
import matplotlib.pyplot as plt
from .paths import get_results_dir

def plot_tuning_comparison(txt_file, algo_name):
    """
    Genera il grafico a barre del tuning (Stile Utente).
    Salva il file png nella cartella results dell'algoritmo con nome dinamico.
    """
    # 1. Recupera la directory corretta per i risultati (es. results/PPO_results)
    output_dir = get_results_dir(algo_name)
    
    # 2. Genera un nome file dinamico basato sul file di testo in input
    # Da "tuning_results_PPO_NOISY.txt" -> "tuning_PPO_NOISY.png"
    base_name = os.path.basename(txt_file)
    image_name = base_name.replace(".txt", ".png").replace("results_", "")
    save_path = os.path.join(output_dir, image_name)

    if not os.path.exists(txt_file):
        print(f"[TUNING] File non trovato: {txt_file}")
        return

    try:
        # skipinitialspace=True è utile se nel txt ci sono spazi dopo la virgola (es: 0.01, 0.2)
        df = pd.read_csv(txt_file, skipinitialspace=True)
        # Pulisce eventuali spazi residui nei nomi delle colonne
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"[TUNING] Errore lettura CSV: {e}")
        return

    if "MEAN_REWARD" not in df.columns:
        print(f"[TUNING] Colonna MEAN_REWARD mancante nel file {base_name}.")
        return

    # Ordina per reward
    df = df.sort_values(by="MEAN_REWARD", ascending=True)

    # Costruzione etichette dinamica (come nel tuo codice originale)
    labels = []
    for index, row in df.iterrows():
        # Usa .get() per evitare crash se la colonna non esiste
        label = f"LR={row.get('LR', '?')}"
        if 'GAMMA' in df.columns: label += f"\nG={row['GAMMA']}"
        if 'ENT_COEF' in df.columns: label += f"\nEnt={row['ENT_COEF']}"
        if 'N_STEPS' in df.columns: label += f" N={row['N_STEPS']}"
        if 'CLIP_RANGE' in df.columns: label += f"\nClip={row['CLIP_RANGE']}"
        if 'N_EPOCHS' in df.columns: label += f" Ep={row['N_EPOCHS']}"
        # Aggiungo supporto per NET_ARCH se presente (per il noisy)
        if 'NET_ARCH' in df.columns: label += f"\nArch={row['NET_ARCH']}"
        
        labels.append(label)

    values = df['MEAN_REWARD']

    # Plotting
    plt.figure(figsize=(14, max(6, len(values)*0.6)))
    bars = plt.barh(labels, values, color='skyblue')
    
    # Colora di verde la barra migliore
    if len(bars) > 0:
        bars[-1].set_color('green')

    # Titolo dinamico (aggiunge NOISY se presente nel nome file)
    suffix = "(NOISY)" if "NOISY" in base_name else "(Standard)"
    plt.title(f"Risultati Tuning {algo_name} {suffix}", fontsize=16)
    
    plt.xlabel("Reward Medio", fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    
    # Aggiunta valori sulle barre
    for i, v in enumerate(values):
        plt.text(v, i, f" {v:.2f}", va='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[TUNING] Grafico salvato: {save_path}")
    plt.close()