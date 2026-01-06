"""
Training PPO con curriculum learning sull'env noisy.

Fase 0: vento molto leggero, niente rumore sensori (quasi nominale)
Fase 1: vento medio, rumore sensori moderato
Fase 2: vento “full” come nel tuo env_noisy_PPO

Il modello PPO è sempre lo stesso: si continua a fare learn
su env via via più difficili.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from quadai.utils.paths import (
    get_raw_logs_dir,
    get_models_dir,
    get_checkpoints_dir,
)
from env_noisy_PPO import droneEnv


ALGO = "PPO"
VERSION = "v1_curriculum"  # nuova etichetta per distinguerlo da v2_noise
TOTAL_TIMESTEPS = 4_000_000  # totale complessivo
# esempio: 3 fasi da ~1.33M ciascuna
PHASE_TIMESTEPS = TOTAL_TIMESTEPS // 3


def make_env(level: int):
    """
    Crea un environment con una certa difficoltà (curriculum level).

    level 0: vento leggero, niente rumore sensori
    level 1: vento medio, rumore moderato
    level 2: vento come nel tuo noisy full (max 0.04, rumore default)
    """
    if level == 0:
        # Fase facile: quasi nominale
        env = droneEnv(
            render_every_frame=False,
            mouse_target=False,
            wind_enabled=True,
            wind_dir_min_deg=0.0,
            wind_dir_max_deg=360.0,
            wind_speed_min=0.0,
            wind_speed_max=0.01,      # vento molto leggero
            wind_update_every=30,
            wind_dir_rw_std_deg=2.0,
            wind_speed_rw_std=0.002,   # random walk molto lieve
            sensor_noise_enabled=False,  # niente rumore sensori
        )
    elif level == 1:
        # Fase intermedia
        env = droneEnv(
            render_every_frame=False,
            mouse_target=False,
            wind_enabled=True,
            wind_dir_min_deg=0.0,
            wind_dir_max_deg=360.0,
            wind_speed_min=0.0,
            wind_speed_max=0.02,      # vento medio
            wind_update_every=30,
            wind_dir_rw_std_deg=2.0,
            wind_speed_rw_std=0.003,
            sensor_noise_enabled=True,
            sensor_noise_std=[        # rumore un po’ più basso del default
                0.005,
                0.01,
                0.005,
                0.01,
                0.005,
                0.005,
                0.01,
            ],
        )
    else:
        # Fase difficile: il tuo noisy “full”
        env = droneEnv(
            render_every_frame=False,
            mouse_target=False,
            wind_enabled=True,
            wind_dir_min_deg=0.0,
            wind_dir_max_deg=360.0,
            wind_speed_min=0.0,
            wind_speed_max=0.04,      # vento forte
            wind_update_every=30,
            wind_dir_rw_std_deg=2.0,
            wind_speed_rw_std=0.003,
            sensor_noise_enabled=True,  # rumore sensori default
            # se non passi sensor_noise_std usa quelli definiti nell'env
        )

    # Monitor per loggare reward per episodio
    # (nota: DummyVecEnv richiede env costruttore senza argomenti, quindi
    # usiamo lambda)
    return env


def train_curriculum():
    # --- setup path/logs ---
    log_dir = get_raw_logs_dir(ALGO)
    checkpoint_dir = get_checkpoints_dir(ALGO)
    models_dir = get_models_dir()

    print(f"--- TRAINING {ALGO} {VERSION} (CURRICULUM) ---")

    # ---------------- FASE 0: facile ----------------
    print("\n[FASE 0] Env facile (vento leggero, senza rumore)...")
    env0 = DummyVecEnv([lambda: Monitor(make_env(level=0),
                                        os.path.join(log_dir, "monitor_curr_phase0.csv"))])

    model = PPO(
        "MlpPolicy",
        env0,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=0.0007,
        clip_range=0.2,
        n_epochs=10,
    )

    checkpoint_callback0 = CheckpointCallback(
        save_freq=100_000,
        save_path=checkpoint_dir,
        name_prefix=f"{ALGO.lower()}_curriculum_phase0",
    )

    model.learn(
        total_timesteps=PHASE_TIMESTEPS,
        callback=checkpoint_callback0,
        tb_log_name="PPO_CURR_PHASE0",
    )

    # ---------------- FASE 1: media ----------------
    print("\n[FASE 1] Env medio (vento medio, rumore moderato)...")
    env1 = DummyVecEnv([lambda: Monitor(make_env(level=1),
                                        os.path.join(log_dir, "monitor_curr_phase1.csv"))])

    model.set_env(env1)

    checkpoint_callback1 = CheckpointCallback(
        save_freq=100_000,
        save_path=checkpoint_dir,
        name_prefix=f"{ALGO.lower()}_curriculum_phase1",
    )

    model.learn(
        total_timesteps=PHASE_TIMESTEPS,
        callback=checkpoint_callback1,
        tb_log_name="PPO_CURR_PHASE1",
    )

    # ---------------- FASE 2: difficile ----------------
    print("\n[FASE 2] Env difficile (vento forte, rumore pieno)...")
    env2 = DummyVecEnv([lambda: Monitor(make_env(level=2),
                                        os.path.join(log_dir, "monitor_curr_phase2.csv"))])

    model.set_env(env2)

    checkpoint_callback2 = CheckpointCallback(
        save_freq=100_000,
        save_path=checkpoint_dir,
        name_prefix=f"{ALGO.lower()}_curriculum_phase2",
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS - 2 * PHASE_TIMESTEPS,
        callback=checkpoint_callback2,
        tb_log_name="PPO_CURR_PHASE2",
    )

    # --- Salvataggio modello finale ---
    filename = f"{ALGO.lower()}_model_{VERSION}_{TOTAL_TIMESTEPS}_steps"
    final_path = os.path.join(models_dir, filename)
    model.save(final_path)
    print(f"\nModello PPO Curriculum salvato in: {final_path}")


if __name__ == "__main__":
    train_curriculum()
