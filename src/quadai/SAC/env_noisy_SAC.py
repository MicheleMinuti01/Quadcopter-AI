"""
2D Quadcopter AI by Alexandre Sajus

Gym environment for RL (SAC, DQN, ecc.)
Goal: reach randomly positioned targets.

Environment esteso con:
- vento con domain randomization per episodio
- vento che varia lentamente nel tempo
- rumore sui sensori
"""

import os
from math import sin, cos, pi, sqrt
from random import randrange, uniform
from typing import Optional

import numpy as np
import gym
from gym import spaces

import pygame
from pygame.locals import *

# path della cartella dove si trova questo file (SAC/)
BASE_PATH = os.path.dirname(__file__)


class droneEnv(gym.Env):
    def __init__(
        self,
        render_every_frame: bool,
        mouse_target: bool,
        # --- parametri vento / meteo ---
        wind_enabled: bool = True,
        wind_dir_min_deg: float = 0.0,
        wind_dir_max_deg: float = 360.0,
        wind_speed_min: float = 0.0,
        wind_speed_max: float = 0.04,
        wind_update_every: int = 30,
        wind_dir_rw_std_deg: float = 2.0,
        wind_speed_rw_std: float = 0.003,
        # --- rumore sensori ---
        sensor_noise_enabled: bool = True,
        sensor_noise_std: Optional[np.ndarray] = None,
    ):
        super(droneEnv, self).__init__()

        self.render_every_frame = render_every_frame
        # Makes the target follow the mouse
        self.mouse_target = mouse_target

        # Initialize Pygame, load sprites
        pygame.init()
        self.screen = pygame.display.set_mode((800, 800))
        self.FramePerSec = pygame.time.Clock()

        # load sprite drone e target usando path relativo a src/quadai/
        self.player = pygame.image.load(
            os.path.join(BASE_PATH, "..", "assets", "sprites", "drone_old.png")
        )
        self.player.convert()

        self.target = pygame.image.load(
            os.path.join(BASE_PATH, "..", "assets", "sprites", "target_old.png")
        )
        self.target.convert()

        pygame.font.init()
        self.myfont = pygame.font.SysFont("Comic Sans MS", 20)

        # Physics constants
        self.FPS = 60
        self.gravity = 0.08
        self.thruster_amplitude = 0.04
        self.diff_amplitude = 0.003
        self.thruster_mean = 0.04
        self.mass = 1
        self.arm = 25

        # Initialize variables (stato fisico drone)
        (self.a, self.ad, self.add) = (0.0, 0.0, 0.0)
        (self.x, self.xd, self.xdd) = (400.0, 0.0, 0.0)
        (self.y, self.yd, self.ydd) = (400.0, 0.0, 0.0)
        self.xt = randrange(200, 600)
        self.yt = randrange(200, 600)

        # Initialize game variables
        self.target_counter = 0
        self.reward = 0.0
        self.time = 0.0
        self.time_limit = 20.0
        if self.mouse_target is True:
            self.time_limit = 1000.0

        # 2 action: thrust amplitude e thrust difference in [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # 7 osservazioni: come nel codice originale
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        # --- PARAMETRI VENTO / METEO ---
        self.wind_enabled = wind_enabled
        self.wind_dir_min_deg = wind_dir_min_deg
        self.wind_dir_max_deg = wind_dir_max_deg
        self.wind_speed_min = wind_speed_min
        self.wind_speed_max = wind_speed_max
        self.wind_update_every = wind_update_every
        self.wind_dir_rw_std_deg = wind_dir_rw_std_deg
        self.wind_speed_rw_std = wind_speed_rw_std

        # stato interno del vento (per episodio)
        self.wind_dir = 0.0      # rad
        self.wind_speed = 0.0    # "intensità" del vento (unità ~ accelerazione)
        self.wind_ax = 0.0       # componente x del vento
        self.wind_ay = 0.0       # componente y del vento
        self.wind_step_counter = 0

        # --- RUMORE SENSORI ---
        self.sensor_noise_enabled = sensor_noise_enabled
        if sensor_noise_std is None:
            # valori esempio: puoi adattarli
            self.sensor_noise_std = np.array(
                [0.01, 0.02, 0.01, 0.02, 0.01, 0.01, 0.02],
                dtype=np.float32,
            )
        else:
            self.sensor_noise_std = np.array(sensor_noise_std, dtype=np.float32)

    # ------------------------------------------------------------------
    # METEO / VENTO
    # ------------------------------------------------------------------
    def _sample_episode_wind(self):
        """
        Domain randomization del vento all'inizio di ogni episodio.
        Sceglie direzione media e intensità media casuali.
        """
        if not self.wind_enabled:
            self.wind_dir = 0.0
            self.wind_speed = 0.0
            self.wind_ax = 0.0
            self.wind_ay = 0.0
            return

        # direzione iniziale del vento, uniforme tra min e max (gradi)
        dir_deg = uniform(self.wind_dir_min_deg, self.wind_dir_max_deg)
        self.wind_dir = np.deg2rad(dir_deg)

        # intensità iniziale del vento tra min e max
        self.wind_speed = uniform(self.wind_speed_min, self.wind_speed_max)

        # calcolo componenti iniziali
        self.wind_ax = self.wind_speed * np.cos(self.wind_dir)
        self.wind_ay = self.wind_speed * np.sin(self.wind_dir)

        # reset contatore interno
        self.wind_step_counter = 0

    def _update_wind(self):
        """
        Aggiorna lentamente il vento dentro l'episodio.
        Random walk sia sulla direzione sia sull'intensità.
        """
        if not self.wind_enabled:
            return

        self.wind_step_counter += 1

        # aggiorna solo ogni N step interni
        if self.wind_step_counter % self.wind_update_every != 0:
            return

        # random walk sulla direzione (in radianti)
        dir_delta = np.deg2rad(np.random.normal(0.0, self.wind_dir_rw_std_deg))
        self.wind_dir += dir_delta

        # normalizza la direzione tra 0 e 2*pi per evitare overflow
        self.wind_dir = (self.wind_dir + 2 * pi) % (2 * pi)

        # random walk sulla velocità, tenendola nel range desiderato
        self.wind_speed += np.random.normal(0.0, self.wind_speed_rw_std)
        self.wind_speed = max(self.wind_speed_min, self.wind_speed)
        self.wind_speed = min(self.wind_speed_max, self.wind_speed)

        # aggiorna componenti x,y
        self.wind_ax = self.wind_speed * np.cos(self.wind_dir)
        self.wind_ay = self.wind_speed * np.sin(self.wind_dir)

    # ------------------------------------------------------------------
    # GYM API
    # ------------------------------------------------------------------
    def reset(self):
        # Reset stato fisico
        (self.a, self.ad, self.add) = (0.0, 0.0, 0.0)
        (self.x, self.xd, self.xdd) = (400.0, 0.0, 0.0)
        (self.y, self.yd, self.ydd) = (400.0, 0.0, 0.0)
        self.xt = randrange(200, 600)
        self.yt = randrange(200, 600)

        self.target_counter = 0
        self.reward = 0.0
        self.time = 0.0

        # Domain randomization del vento: nuovo "meteo" per episodio
        self._sample_episode_wind()

        return self.get_obs()

    def get_obs(self) -> np.ndarray:
        """
        Calcola le osservazioni del drone (con rumore opzionale).
        """
        angle_to_up = self.a / 180 * pi
        velocity = sqrt(self.xd**2 + self.yd**2)
        angle_velocity = self.ad
        distance_to_target = (
            sqrt((self.xt - self.x) ** 2 + (self.yt - self.y) ** 2) / 500
        )
        angle_to_target = np.arctan2(self.yt - self.y, self.xt - self.x)
        # Angle between the to_target vector and the velocity vector
        angle_target_and_velocity = np.arctan2(
            self.yt - self.y, self.xt - self.x
        ) - np.arctan2(self.yd, self.xd)
        distance_to_target = (
            sqrt((self.xt - self.x) ** 2 + (self.yt - self.y) ** 2) / 500
        )

        obs = np.array(
            [
                angle_to_up,
                velocity,
                angle_velocity,
                distance_to_target,
                angle_to_target,
                angle_target_and_velocity,
                distance_to_target,
            ],
            dtype=np.float32,
        )

        if self.sensor_noise_enabled:
            noise = np.random.normal(0.0, self.sensor_noise_std).astype(np.float32)
            obs = obs + noise

        return obs

    def step(self, action):
        # Game loop
        self.reward = 0.0
        (action0, action1) = (float(action[0]), float(action[1]))

        # Act every 5 frames (come nel codice originale)
        for _ in range(5):
            self.time += 1 / 60

            if self.mouse_target is True:
                self.xt, self.yt = pygame.mouse.get_pos()

            # Aggiorna vento (random walk lento)
            self._update_wind()

            # Initialize accelerations
            self.xdd = 0.0
            self.ydd = self.gravity
            self.add = 0.0
            thruster_left = self.thruster_mean
            thruster_right = self.thruster_mean

            thruster_left += action0 * self.thruster_amplitude
            thruster_right += action0 * self.thruster_amplitude
            thruster_left += action1 * self.diff_amplitude
            thruster_right -= action1 * self.diff_amplitude

            # Calculating accelerations with Newton's laws of motions
            self.xdd += (
                -(thruster_left + thruster_right) * sin(self.a * pi / 180) / self.mass
            )
            self.ydd += (
                -(thruster_left + thruster_right) * cos(self.a * pi / 180) / self.mass
            )
            self.add += self.arm * (thruster_right - thruster_left) / self.mass

            # aggiungi contributo del vento come accelerazione extra
            if self.wind_enabled:
                self.xdd += self.wind_ax
                self.ydd += self.wind_ay

            # integrazione (velocità)
            self.xd += self.xdd
            self.yd += self.ydd
            self.ad += self.add

            # integrazione (posizione)
            self.x += self.xd
            self.y += self.yd
            self.a += self.ad

            dist = sqrt((self.x - self.xt) ** 2 + (self.y - self.yt) ** 2)

            # Reward per step sopravvissuto
            self.reward += 1 / 60
            # Penalty in base alla distanza dal target
            self.reward -= dist / (100 * 60)

            if dist < 50:
                # Reward se vicino al target
                self.xt = randrange(200, 600)
                self.yt = randrange(200, 600)
                self.reward += 100
                self.target_counter += 1

            # If out of time
            if self.time > self.time_limit:
                done = True
                break

            # If too far from target (crash)
            elif dist > 1000:
                self.reward -= 1000
                done = True
                break

            else:
                done = False

            if self.render_every_frame is True:
                self.render("yes")

        info = {}

        return (
            self.get_obs(),
            self.reward,
            done,
            info,
        )

    def render(self, mode):
        # Pygame rendering
        pygame.event.get()
        self.screen.fill(0)
        self.screen.blit(
            self.target,
            (
                self.xt - int(self.target.get_width() / 2),
                self.yt - int(self.target.get_height() / 2),
            ),
        )
        player_copy = pygame.transform.rotate(self.player, self.a)
        self.screen.blit(
            player_copy,
            (
                self.x - int(player_copy.get_width() / 2),
                self.y - int(player_copy.get_height() / 2),
            ),
        )

        textsurface = self.myfont.render(
            "Collected: " + str(self.target_counter), False, (255, 255, 255)
        )
        self.screen.blit(textsurface, (20, 20))
        textsurface3 = self.myfont.render(
            "Time: " + str(int(self.time)), False, (255, 255, 255)
        )
        self.screen.blit(textsurface3, (20, 50))

        pygame.display.update()
        self.FramePerSec.tick(self.FPS)

    def close(self):
        pass
