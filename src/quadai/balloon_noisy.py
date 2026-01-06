"""
2D Quadcopter AI by Alexandre Sajus

This is the main game where you can compete with AI agents.
Collect as many balloons within the time limit.

Versione estesa con:
- vento globale con domain randomization per partita
- vento che varia lentamente nel tempo
- freccia che visualizza direzione/intensità del vento
- rumore sui sensori (osservazioni disturbate come nell'env robusto)
"""

import os
from random import randrange
from math import sin, cos, pi, sqrt

import numpy as np
import pygame
from pygame.locals import *
from quadai.player import HumanPlayer, PIDPlayer, SACPlayer, A2CPlayer, PPOPlayer, PPO_noisy_Player, PPO_curriculum_Player


def correct_path(current_path):
    """
    This function is used to get the correct path to the assets folder
    """
    return os.path.join(os.path.dirname(__file__), current_path)


# -----------------------------
# Parametri e stato del vento
# -----------------------------
WIND_ENABLED = True
WIND_DIR_MIN_DEG = 0.0
WIND_DIR_MAX_DEG = 360.0
WIND_SPEED_MIN = 0.01
WIND_SPEED_MAX = 0.05
WIND_UPDATE_EVERY = 30
WIND_DIR_RW_STD_DEG = 2.0
WIND_SPEED_RW_STD = 0.003

# stato globale del vento
wind_dir = 0.0       # rad
wind_speed = 0.0     # intensità (unità ~ accelerazione)
wind_ax = 0.0        # componente x
wind_ay = 0.0        # componente y
wind_step_counter = 0

# -----------------------------
# Rumore sensori (come nell'env)
# -----------------------------
SENSOR_NOISE_ENABLED = True
SENSOR_NOISE_STD = np.array(
    [0.01, 0.02, 0.01, 0.02, 0.01, 0.01, 0.02],
    dtype=np.float32,
)


def sample_episode_wind():
    """
    Domain randomization del vento all'inizio della partita.
    """
    global wind_dir, wind_speed, wind_ax, wind_ay, wind_step_counter

    if not WIND_ENABLED:
        wind_dir = 0.0
        wind_speed = 0.0
        wind_ax = 0.0
        wind_ay = 0.0
        return

    # direzione iniziale uniforme tra min e max (gradi)
    dir_deg = np.random.uniform(WIND_DIR_MIN_DEG, WIND_DIR_MAX_DEG)
    wind_dir = np.deg2rad(dir_deg)

    # intensità iniziale tra min e max
    wind_speed = np.random.uniform(WIND_SPEED_MIN, WIND_SPEED_MAX)

    # componenti iniziali
    wind_ax = wind_speed * np.cos(wind_dir)
    wind_ay = wind_speed * np.sin(wind_dir)

    wind_step_counter = 0


def update_wind():
    """
    Aggiorna lentamente il vento durante la partita.
    Random walk su direzione e intensità.
    """
    global wind_dir, wind_speed, wind_ax, wind_ay, wind_step_counter

    if not WIND_ENABLED:
        return

    wind_step_counter += 1
    if wind_step_counter % WIND_UPDATE_EVERY != 0:
        return

    # random walk sulla direzione
    dir_delta = np.deg2rad(np.random.normal(0.0, WIND_DIR_RW_STD_DEG))
    wind_dir = (wind_dir + dir_delta + 2 * pi) % (2 * pi)

    # random walk sulla velocità
    wind_speed += np.random.normal(0.0, WIND_SPEED_RW_STD)
    wind_speed = max(WIND_SPEED_MIN, min(WIND_SPEED_MAX, wind_speed))

    # aggiorna componenti x,y
    wind_ax = wind_speed * np.cos(wind_dir)
    wind_ay = wind_speed * np.sin(wind_dir)


def add_sensor_noise(obs: np.ndarray) -> np.ndarray:
    """
    Aggiunge rumore gaussiano alle osservazioni, come in droneEnv.get_obs().
    """
    if not SENSOR_NOISE_ENABLED:
        return obs
    noise = np.random.normal(0.0, SENSOR_NOISE_STD).astype(np.float32)
    return obs + noise


def balloon_noisy():
    """
    Runs the balloon game.
    """
    # Game constants
    FPS = 60
    WIDTH = 800
    HEIGHT = 800

    # Physics constants
    gravity = 0.08
    # Propeller force for UP and DOWN
    thruster_amplitude = 0.04
    # Propeller force for LEFT and RIGHT rotations
    diff_amplitude = 0.003
    # By default, thruster will apply angle force of thruster_mean
    thruster_mean = 0.04
    mass = 1
    # Length from center of mass to propeller
    arm = 25

    # Initialize Pygame, load sprites
    FramePerSec = pygame.time.Clock()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # Loading player and target sprites
    player_width = 80
    player_animation_speed = 0.3
    player_animation = []
    for i in range(1, 5):
        image = pygame.image.load(
            correct_path(
                os.path.join(
                    "assets/balloon-flat-asset-pack/png/objects/drone-sprites/drone-"
                    + str(i)
                    + ".png"
                )
            )
        )
        image.convert()
        player_animation.append(
            pygame.transform.scale(image, (player_width, int(player_width * 0.30)))
        )

    target_width = 30
    target_animation_speed = 0.1
    target_animation = []
    for i in range(1, 8):
        image = pygame.image.load(
            correct_path(
                os.path.join(
                    "assets/balloon-flat-asset-pack/png/balloon-sprites/red-plain/red-plain-"
                    + str(i)
                    + ".png"
                )
            )
        )
        image.convert()
        target_animation.append(
            pygame.transform.scale(image, (target_width, int(target_width * 1.73)))
        )

    # Loading background sprites
    cloud1 = pygame.image.load(
        correct_path(
            os.path.join(
                "assets/balloon-flat-asset-pack/png/background-elements/cloud-1.png"
            )
        )
    )
    cloud2 = pygame.image.load(
        correct_path(
            os.path.join(
                "assets/balloon-flat-asset-pack/png/background-elements/cloud-2.png"
            )
        )
    )
    sun = pygame.image.load(
        correct_path(
            os.path.join(
                "assets/balloon-flat-asset-pack/png/background-elements/sun.png"
            )
        )
    )
    cloud1.set_alpha(124)
    (x_cloud1, y_cloud1, speed_cloud1) = (150, 200, 0.3)
    cloud2.set_alpha(124)
    (x_cloud2, y_cloud2, speed_cloud2) = (400, 500, -0.2)
    sun.set_alpha(124)

    # Loading fonts
    pygame.font.init()
    name_font = pygame.font.Font(correct_path("assets/fonts/Roboto-Bold.ttf"), 20)
    name_hud_font = pygame.font.Font(correct_path("assets/fonts/Roboto-Bold.ttf"), 15)
    time_font = pygame.font.Font(correct_path("assets/fonts/Roboto-Bold.ttf"), 30)
    score_font = pygame.font.Font(correct_path("assets/fonts/Roboto-Regular.ttf"), 20)
    respawn_timer_font = pygame.font.Font(
        correct_path("assets/fonts/Roboto-Bold.ttf"), 90
    )
    respawning_font = pygame.font.Font(
        correct_path("assets/fonts/Roboto-Regular.ttf"), 15
    )

    # Function to display info about a player
    def display_info(position):
        name_text = name_font.render(player.name, True, (255, 255, 255))
        screen.blit(name_text, (position, 20))
        target_text = score_font.render(
            "Score : " + str(player.target_counter), True, (255, 255, 255)
        )
        screen.blit(target_text, (position, 45))
        if player.dead is True:
            respawning_text = respawning_font.render(
                "Respawning...", True, (255, 255, 255)
            )
            screen.blit(respawning_text, (position, 70))

    # Initialize game variables
    time = 0
    step = 0
    time_limit = 100
    respawn_timer_max = 3







    players = [
        #HumanPlayer(),
        PIDPlayer(),
        SACPlayer(),
        #A2CPlayer(),
        PPOPlayer(),
        #PPO_noisy_Player(),
        PPO_curriculum_Player()
    ]





    # Generate 100 targets
    targets = []
    for i in range(100):
        targets.append((randrange(200, 600), randrange(200, 600)))

    # Nuovo meteo per questa partita
    sample_episode_wind()

    # Game loop
    while True:
        pygame.event.get()

        # Display background
        screen.fill((131, 176, 181))

        x_cloud1 += speed_cloud1
        if x_cloud1 > WIDTH:
            x_cloud1 = -cloud1.get_width()
        screen.blit(cloud1, (x_cloud1, y_cloud1))

        x_cloud2 += speed_cloud2
        if x_cloud2 < -cloud2.get_width():
            x_cloud2 = WIDTH
        screen.blit(cloud2, (x_cloud2, y_cloud2))

        screen.blit(sun, (630, -100))

        time += 1 / 60
        step += 1

        # aggiorna lentamente il vento
        update_wind()

        # Visualizzazione del vento con freccia + intensità in km/h
        if WIND_ENABLED:
            # centro della freccia
            cx, cy = 100, 650

            # scala per la lunghezza della freccia (più corta)
            base_scale = 2000
            scale = base_scale

            # punto finale del corpo della freccia
            end_x = cx + int(wind_ax * scale)
            end_y = cy + int(wind_ay * scale)

            # disegna il corpo della freccia
            pygame.draw.line(screen, (255, 255, 255), (cx, cy), (end_x, end_y), 3)

            # disegna la punta della freccia
            dx = end_x - cx
            dy = end_y - cy
            length = sqrt(dx * dx + dy * dy)
            if length > 0:
                udx = dx / length
                udy = dy / length

                head_len = 15
                head_angle = pi / 6  # 30°

                sin_a = sin(head_angle)
                cos_a = cos(head_angle)

                # lato 1
                hx1 = end_x - head_len * (udx * cos_a - udy * sin_a)
                hy1 = end_y - head_len * (udx * sin_a + udy * cos_a)

                # lato 2
                hx2 = end_x - head_len * (udx * cos_a + udy * sin_a)
                hy2 = end_y - head_len * (-udx * sin_a + udy * cos_a)

                pygame.draw.line(
                    screen, (255, 255, 255), (end_x, end_y), (hx1, hy1), 3
                )
                pygame.draw.line(
                    screen, (255, 255, 255), (end_x, end_y), (hx2, hy2), 3
                )

            # intensità del vento in km/h (assumendo wind_speed in m/s)
            wind_speed_kmh = abs(wind_speed) * 3.6 * 10
            intensity_text = score_font.render(
                f"Wind: {wind_speed_kmh:.2f} km/h", True, (255, 255, 255)
            )
            # più staccata dalla freccia (spostata verso il basso)
            screen.blit(
                intensity_text,
                (
                    cx - intensity_text.get_width() // 2,
                    cy + 100,
                ),
            )

        # For each player
        for player_index, player in enumerate(players):
            if player.dead is False:
                # Initialize accelerations
                player.x_acceleration = 0
                player.y_acceleration = gravity
                player.angular_acceleration = 0

                # Calculate propeller force in function of input
                if player.name == "DQN" or player.name == "PID":
                    thruster_left, thruster_right = player.act(
                        [
                            targets[player.target_counter][0] - player.x_position,
                            player.x_speed,
                            targets[player.target_counter][1] - player.y_position,
                            player.y_speed,
                            player.angle,
                            player.angular_speed,
                        ]
                    )






                elif player.name == "A2C":
                    xt = targets[player.target_counter][0]
                    yt = targets[player.target_counter][1]

                    angle_to_up = player.angle / 180 * pi
                    velocity = sqrt(player.x_speed**2 + player.y_speed**2)
                    angle_velocity = player.angular_speed
                    dist_val = sqrt(
                        (xt - player.x_position) ** 2 + (yt - player.y_position) ** 2
                    )
                    distance_to_target = dist_val / 500
                    angle_to_target = np.arctan2(
                        yt - player.y_position, xt - player.x_position
                    )
                    angle_target_and_velocity = np.arctan2(
                        yt - player.y_position, xt - player.x_position
                    ) - np.arctan2(player.y_speed, player.x_speed)

                    obs_array = np.array(
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

                    # rumore sensori
                    obs_array = add_sensor_noise(obs_array)

                    thruster_left, thruster_right = player.act(obs_array)







                elif player.name == "PPO":
                    xt = targets[player.target_counter][0]
                    yt = targets[player.target_counter][1]

                    angle_to_up = player.angle / 180 * pi
                    velocity = sqrt(player.x_speed**2 + player.y_speed**2)
                    angle_velocity = player.angular_speed
                    dist_val = sqrt(
                        (xt - player.x_position) ** 2 + (yt - player.y_position) ** 2
                    )
                    distance_to_target = dist_val / 500
                    angle_to_target = np.arctan2(
                        yt - player.y_position, xt - player.x_position
                    )
                    angle_target_and_velocity = np.arctan2(
                        yt - player.y_position, xt - player.x_position
                    ) - np.arctan2(player.y_speed, player.x_speed)

                    obs_array = np.array(
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

                    # rumore sensori (puoi disabilitare se vuoi PPO “pulito”)
                    # obs_array = add_sensor_noise(obs_array)

                    thruster_left, thruster_right = player.act(obs_array)







                elif player.name == "PPO_noisy":
                    xt = targets[player.target_counter][0]
                    yt = targets[player.target_counter][1]

                    angle_to_up = player.angle / 180 * pi
                    velocity = sqrt(player.x_speed**2 + player.y_speed**2)
                    angle_velocity = player.angular_speed
                    dist_val = sqrt(
                        (xt - player.x_position) ** 2 + (yt - player.y_position) ** 2
                    )
                    distance_to_target = dist_val / 500
                    angle_to_target = np.arctan2(
                        yt - player.y_position, xt - player.x_position
                    )
                    angle_target_and_velocity = np.arctan2(
                        yt - player.y_position, xt - player.x_position
                    ) - np.arctan2(player.y_speed, player.x_speed)

                    obs_array = np.array(
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

                    # PPO_noisy vede osservazioni disturbate, come in training
                    obs_array = add_sensor_noise(obs_array)

                    thruster_left, thruster_right = player.act(obs_array)






                elif player.name == "PPO_curriculum":
                    xt = targets[player.target_counter][0]
                    yt = targets[player.target_counter][1]

                    angle_to_up = player.angle / 180 * pi
                    velocity = sqrt(player.x_speed**2 + player.y_speed**2)
                    angle_velocity = player.angular_speed
                    dist_val = sqrt(
                        (xt - player.x_position) ** 2 + (yt - player.y_position) ** 2
                    )
                    distance_to_target = dist_val / 500
                    angle_to_target = np.arctan2(
                        yt - player.y_position, xt - player.x_position
                    )
                    angle_target_and_velocity = np.arctan2(
                        yt - player.y_position, xt - player.x_position
                    ) - np.arctan2(player.y_speed, player.x_speed)

                    obs_array = np.array(
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

                    # PPO_noisy vede osservazioni disturbate, come in training
                    obs_array = add_sensor_noise(obs_array)

                    thruster_left, thruster_right = player.act(obs_array)








                elif player.name == "SAC":
                    angle_to_up = player.angle / 180 * pi
                    velocity = sqrt(player.x_speed**2 + player.y_speed**2)
                    angle_velocity = player.angular_speed
                    distance_to_target = (
                        sqrt(
                            (targets[player.target_counter][0] - player.x_position) ** 2
                            + (targets[player.target_counter][1] - player.y_position)
                            ** 2
                        )
                        / 500
                    )
                    angle_to_target = np.arctan2(
                        targets[player.target_counter][1] - player.y_position,
                        targets[player.target_counter][0] - player.x_position,
                    )
                    angle_target_and_velocity = np.arctan2(
                        targets[player.target_counter][1] - player.y_position,
                        targets[player.target_counter][0] - player.x_position,
                    ) - np.arctan2(player.y_speed, player.x_speed)
                    distance_to_target = (
                        sqrt(
                            (targets[player.target_counter][0] - player.x_position) ** 2
                            + (targets[player.target_counter][1] - player.y_position)
                            ** 2
                        )
                        / 500
                    )

                    obs_array = np.array(
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

                    # se il SAC è stato addestrato con rumore, lo riapplichi qui
                    # obs_array = add_sensor_noise(obs_array)

                    thruster_left, thruster_right = player.act(obs_array)






                else:
                    # Human player
                    thruster_left, thruster_right = player.act([])

                # Calculate accelerations according to Newton's laws of motion
                player.x_acceleration += (
                    -(thruster_left + thruster_right)
                    * sin(player.angle * pi / 180)
                    / mass
                )
                player.y_acceleration += (
                    -(thruster_left + thruster_right)
                    * cos(player.angle * pi / 180)
                    / mass
                )
                player.angular_acceleration += (
                    arm * (thruster_right - thruster_left) / mass
                )

                # aggiungi contributo del vento
                if WIND_ENABLED:
                    player.x_acceleration += wind_ax
                    player.y_acceleration += wind_ay

                # Calculate speed
                player.x_speed += player.x_acceleration
                player.y_speed += player.y_acceleration
                player.angular_speed += player.angular_acceleration

                # Calculate position
                player.x_position += player.x_speed
                player.y_position += player.y_speed
                player.angle += player.angular_speed

                # Calculate distance to target
                dist = sqrt(
                    (player.x_position - targets[player.target_counter][0]) ** 2
                    + (player.y_position - targets[player.target_counter][1]) ** 2
                )

                # If target reached, respawn target
                if dist < 50:
                    player.target_counter += 1

                # If too far, die and respawn after timer
                elif dist > 1000:
                    player.dead = True
                    player.respawn_timer = respawn_timer_max
            else:
                # Display respawn timer
                if player.name == "Human":
                    respawn_text = respawn_timer_font.render(
                        str(int(player.respawn_timer) + 1), True, (255, 255, 255)
                    )
                    respawn_text.set_alpha(124)
                    screen.blit(
                        respawn_text,
                        (
                            WIDTH / 2 - respawn_text.get_width() / 2,
                            HEIGHT / 2 - respawn_text.get_height() / 2,
                        ),
                    )

                player.respawn_timer -= 1 / 60
                # Respawn
                if player.respawn_timer < 0:
                    player.dead = False
                    (
                        player.angle,
                        player.angular_speed,
                        player.angular_acceleration,
                    ) = (0, 0, 0)
                    (player.x_position, player.x_speed, player.x_acceleration) = (
                        400,
                        0,
                        0,
                    )
                    (player.y_position, player.y_speed, player.y_acceleration) = (
                        400,
                        0,
                        0,
                    )

            # Display target and player
            target_sprite = target_animation[
                int(step * target_animation_speed) % len(target_animation)
            ]
            target_sprite.set_alpha(player.alpha)
            screen.blit(
                target_sprite,
                (
                    targets[player.target_counter][0]
                    - int(target_sprite.get_width() / 2),
                    targets[player.target_counter][1]
                    - int(target_sprite.get_height() / 2),
                ),
            )

            player_sprite = player_animation[
                int(step * player_animation_speed) % len(player_animation)
            ]
            player_copy = pygame.transform.rotate(player_sprite, player.angle)
            player_copy.set_alpha(player.alpha)
            screen.blit(
                player_copy,
                (
                    player.x_position - int(player_copy.get_width() / 2),
                    player.y_position - int(player_copy.get_height() / 2),
                ),
            )

            # Display player name
            name_hud_text = name_hud_font.render(player.name, True, (255, 255, 255))
            screen.blit(
                name_hud_text,
                (
                    player.x_position - int(name_hud_text.get_width() / 2),
                    player.y_position - 30 - int(name_hud_text.get_height() / 2),
                ),
            )

            # Display player info
            if player_index == 0:  # Human
                display_info(20)
            elif player_index == 1:  # PID
                display_info(130)
            elif player_index == 2:  # SAC
                display_info(240)
            elif player_index == 3:  # A2C
                display_info(350)
            elif player_index == 4:  # PPO
                display_info(460)
            elif player_index == 5:  # PPO_noisy
                display_info(570)
            elif player_index == 6:  # PPO_curriculum
                display_info(680)

            time_text = time_font.render(
                "Time : " + str(int(time_limit - time)), True, (255, 255, 255)
            )
            screen.blit(time_text, (670, 30))

        # Ending conditions
        if time > time_limit:
            break

        pygame.display.update()
        FramePerSec.tick(FPS)

    # Print scores and who won
    print("")
    scores = []
    for player in players:
        print(player.name + " collected : " + str(player.target_counter))
        scores.append(player.target_counter)
    winner = players[np.argmax(scores)].name

    print("")
    print("Winner is : " + winner + " !")
