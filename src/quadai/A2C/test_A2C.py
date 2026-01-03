import os
import time
from stable_baselines3 import A2C
from env_A2C import droneEnv

def test():
    # --- 1. CONFIGURAZIONE ---
    # Il nome del file che hai salvato nella cartella \models
    FILENAME = "a2c_model_v0_100000_steps.zip"

    # Otteniamo la cartella dove si trova questo script (A2C/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Costruiamo il percorso per salire di un livello e entrare in /models
    model_path = os.path.normpath(os.path.join(script_dir, "..", "models", FILENAME))

    if not os.path.exists(model_path):
        print(f"\nERRORE: Non trovo il modello finale!")
        print(f"Ho cercato in: {model_path}")
        return

    print(f"Caricamento modello finale da: {model_path}")

    # --- 2. AMBIENTE CON GRAFICA ---
    env = droneEnv(render_every_frame=True, mouse_target=False)

    # --- 3. CARICAMENTO ---
    model = A2C.load(model_path, env=env)

    # --- 4. VOLO DI PROVA ---
    for ep in range(5):
        obs = env.reset()
        done = False
        score = 0
        
        print(f"\n--- Test Episodio {ep + 1} ---")
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            score += reward
            time.sleep(0.01) # Un piccolo delay per non farlo andare a velocità luce

        print(f"Punteggio finale: {score:.2f}")

if __name__ == "__main__":
    test()