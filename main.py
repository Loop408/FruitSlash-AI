import pygame
import random
import cv2
import mediapipe as mp
import threading
import os
import sys
import math

# ---- MediaPipe Hand Tracking ----
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.85,
                       min_tracking_confidence=0.85)
mp_draw = mp.solutions.drawing_utils
hand_x, hand_y = 0, 0
prev_hand_x, prev_hand_y = 0, 0
finger_trail = []
running_tracker = True
frame_for_display = None

# Track hand in separate thread
def track_hand():
    global hand_x, hand_y, prev_hand_x, prev_hand_y, running_tracker, finger_trail, frame_for_display
    while running_tracker:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        frame_for_display = frame.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                x = int(hand_landmarks.landmark[8].x * w)
                y = int(hand_landmarks.landmark[8].y * h)
                prev_hand_x, prev_hand_y = hand_x, hand_y
                hand_x = int(x * 800 / w)
                hand_y = int(y * 600 / h)
                finger_trail.append((hand_x, hand_y))
                if len(finger_trail) > 15:
                    finger_trail.pop(0)
        else:
            finger_trail = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running_tracker = False
            break

tracker_thread = threading.Thread(target=track_hand, daemon=True)
tracker_thread.start()

# ---- Pygame Setup ----
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Fruit Ninja AI")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 40)

# Load background
bg = pygame.image.load("background.jpg")
bg = pygame.transform.scale(bg, (800, 600))

# Load slash effect
try:
    slash_img = pygame.image.load("slash.png").convert_alpha()
    slash_img = pygame.transform.scale(slash_img, (80, 80))
except:
    slash_img = None

# Load fruit images
fruit_imgs = []
fruit_folder = "fruit_images"
os.makedirs(fruit_folder, exist_ok=True)
for f in os.listdir(fruit_folder):
    if f.endswith(".png"):
        img = pygame.image.load(os.path.join(fruit_folder, f)).convert_alpha()
        fruit_imgs.append(pygame.transform.scale(img, (100, 100)))

# Load bomb images
bomb_imgs = []
for f in os.listdir(fruit_folder):
    if "bomb" in f.lower() and f.endswith(".png"):
        img = pygame.image.load(os.path.join(fruit_folder, f)).convert_alpha()
        bomb_imgs.append(pygame.transform.scale(img, (100, 100)))

# ---- Game Variables ----
score = 0
bomb_hits = 0  # Counter for bombs hit
GRAVITY = 0.35
fruits = []
slice_effects = []
slash_effects = []
timer = 0
paused = False
spawn_interval = 30
speed_multiplier = 1.0

# Buttons
pause_btn = pygame.Rect(680, 10, 100, 40)
resume_btn = pygame.Rect(340, 250, 120, 50)
quit_btn = pygame.Rect(340, 320, 120, 50)

# Spawn a fruit or bomb
def spawn_fruit():
    if random.random() < 0.15 and bomb_imgs:  # 15% chance of bomb
        img = random.choice(bomb_imgs)
        is_bomb = True
    else:
        img = random.choice(fruit_imgs)
        is_bomb = False

    x = random.randint(300, 500)  # spawn around the center
    vel_x = random.uniform(-2, 2) * speed_multiplier
    vel_y = random.uniform(-20, -17) * speed_multiplier
    return {
        "img": img,
        "pos": [x, 600],
        "vel": [vel_x, vel_y],
        "active": True,
        "sliced": False,
        "bomb": is_bomb
    }

# Handle slicing
def handle_slice(fruit):
    global score, bomb_hits, running
    fx, fy = fruit["pos"]
    w, h = fruit["img"].get_size()

    if fruit["bomb"]:
        bomb_hits += 1
        if bomb_hits >= 10:  # Game Over condition
            running = False
        if slash_img:
            sx = hand_x - slash_img.get_width() // 2
            sy = hand_y - slash_img.get_height() // 2
            slash_effects.append({"img": slash_img, "pos": [sx, sy], "timer": 5})
        for i in range(25):
            spark = {
                "pos": [fx + w // 2, fy + h // 2],
                "vel": [random.uniform(-7, 7), random.uniform(-7, 7)],
                "timer": 25,
                "radius": random.randint(3, 6),
                "color": [random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)]  # Random color
            }
            slash_effects.append(spark)
        return

    # Normal fruit slice
    left = pygame.Surface((w // 2, h), pygame.SRCALPHA)
    right = pygame.Surface((w // 2, h), pygame.SRCALPHA)
    left.blit(fruit["img"], (0, 0), (0, 0, w // 2, h))
    right.blit(fruit["img"], (0, 0), (w // 2, 0, w // 2, h))

    slice_effects.append({"img": left, "pos": [fx, fy], "vel": [-4, -6], "rot": 0, "rot_speed": -10, "timer": 30})
    slice_effects.append({"img": right, "pos": [fx + w // 2, fy], "vel": [4, -6], "rot": 0, "rot_speed": 10, "timer": 30})

    for i in range(20):
        spark = {
            "pos": [fx + w // 2, fy + h // 2],
            "vel": [random.uniform(-6, 6), random.uniform(-6, 6)],
            "timer": 20,
            "radius": random.randint(2, 5),
            "color": [random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)]  # Random color
        }
        slash_effects.append(spark)

    if slash_img:
        sx = hand_x - slash_img.get_width() // 2
        sy = hand_y - slash_img.get_height() // 2
        slash_effects.append({"img": slash_img, "pos": [sx, sy], "timer": 5})

    score += 1

# ---- Main Game Loop ----
running = True
while running:
    screen.blit(bg, (0, 0))

    if frame_for_display is not None:
        cam_surf = pygame.surfarray.make_surface(cv2.cvtColor(cv2.resize(frame_for_display, (160, 120)), cv2.COLOR_BGR2RGB).swapaxes(0, 1))
        screen.blit(cam_surf, (10, 470))

    if not paused:
        timer += 1

        if score >= 50:
            speed_multiplier = 1.4
        elif score >= 40:
            speed_multiplier = 1.3
        elif score >= 20:
            speed_multiplier = 1.2
        else:
            speed_multiplier = 1.0

        if timer % int(spawn_interval / speed_multiplier) == 0:
            fruits.append(spawn_fruit())

        for effect in slice_effects[:]:
            effect["pos"][0] += effect["vel"][0]
            effect["pos"][1] += effect["vel"][1]
            effect["vel"][1] += GRAVITY
            effect["rot"] += effect["rot_speed"]
            rotated = pygame.transform.rotate(effect["img"], effect["rot"])
            rect = rotated.get_rect(center=(effect["pos"][0] + effect["img"].get_width() // 2,
                                            effect["pos"][1] + effect["img"].get_height() // 2))
            screen.blit(rotated, rect.topleft)
            effect["timer"] -= 1
            if effect["timer"] <= 0:
                slice_effects.remove(effect)

        for eff in slash_effects[:]:
            if "img" in eff:
                screen.blit(eff["img"], eff["pos"])
                eff["timer"] -= 1
                if eff["timer"] <= 0:
                    slash_effects.remove(eff)
            else:
                pygame.draw.circle(screen, eff["color"], eff["pos"], eff["radius"])
                eff["pos"][0] += eff["vel"][0]
                eff["pos"][1] += eff["vel"][1]
                eff["timer"] -= 1
                if eff["timer"] <= 0:
                    slash_effects.remove(eff)

        for fruit in fruits[:]:
            if fruit["active"]:
                fruit["vel"][1] += GRAVITY
                fruit["pos"][0] += fruit["vel"][0]
                fruit["pos"][1] += fruit["vel"][1]
                screen.blit(fruit["img"], fruit["pos"])
                fx, fy = fruit["pos"]
                w, h = fruit["img"].get_size()
                if not fruit["sliced"] and len(finger_trail) > 1:
                    for i in range(1, len(finger_trail)):
                        x1, y1 = finger_trail[i - 1]
                        x2, y2 = finger_trail[i]
                        dx, dy = x2 - x1, y2 - y1
                        dist = math.hypot(dx, dy)
                        if dist > 20:
                            for j in range(11):
                                ix = x1 + j * dx / 10
                                iy = y1 + j * dy / 10
                                if fx < ix < fx + w and fy < iy < fy + h:
                                    handle_slice(fruit)
                                    fruit["active"] = False
                                    fruit["sliced"] = True
                                    break
            else:
                fruits.remove(fruit)

        pygame.draw.rect(screen, (0, 128, 0), pause_btn)
        screen.blit(font.render("Pause", True, (255, 255, 255)), (pause_btn.x + 10, pause_btn.y + 5))
        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(score_text, (350, 10))
    else:
        pygame.draw.rect(screen, (0, 0, 0, 180), pygame.Rect(0, 0, 800, 600))
        pygame.draw.rect(screen, (0, 128, 255), resume_btn)
        pygame.draw.rect(screen, (255, 0, 0), quit_btn)
        screen.blit(font.render("Resume", True, (255, 255, 255)), (resume_btn.x + 10, resume_btn.y + 10))
        screen.blit(font.render("Quit", True, (255, 255, 255)), (quit_btn.x + 25, quit_btn.y + 10))

    if bomb_hits >= 10:
        game_over_text = font.render("Game Over! You hit 10 bombs!", True, (255, 0, 0))
        screen.blit(game_over_text, (200, 200))

    if len(finger_trail) > 1:
        pygame.draw.lines(screen, (0, 0, 255), False, finger_trail, 3)

    pygame.display.flip()
    clock.tick(60)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if pause_btn.collidepoint(event.pos):
                paused = not paused
            elif resume_btn.collidepoint(event.pos):
                paused = False
            elif quit_btn.collidepoint(event.pos):
                running = False

cap.release()
pygame.quit()
sys.exit()

