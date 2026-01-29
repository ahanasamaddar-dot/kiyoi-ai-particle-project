import cv2
import mediapipe as mp
import pygame
import random
import math
import time
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Configuration ---
WIDTH, HEIGHT = 1280, 720
MODEL_FILE = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

def download_model():
    if not os.path.exists(MODEL_FILE):
        print(f"Downloading {MODEL_FILE}...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
            print("Download Complete.")
        except Exception as e:
            print(f"Error downloading model: {e}")

class Snowflake:
    def __init__(self, x, y, style="normal"):
        self.x = x
        self.y = y
        self.style = style
        self.life = 255
        self.spin = random.uniform(0, 360)
        self.spin_speed = random.uniform(-5, 5)
        self.size = random.randint(4, 10)
        
        if style == "beam":
            self.vx = random.uniform(-1, 1)
            self.vy = random.uniform(-15, -10)
            self.color = (150, 255, 255)
        elif style == "vortex":
            self.vx, self.vy = random.uniform(-4, 4), random.uniform(-4, 4)
            self.color = (100, 255, 150)
        elif style == "gravity":
            self.vx, self.vy = 0, 0
            self.color = (200, 100, 255)
            self.size = random.randint(2, 6)
        else: # normal
            self.vx = random.uniform(-1, 1)
            self.vy = random.uniform(2, 5)
            self.color = (200, 230, 255)

    def update(self, target_x=None, target_y=None):
        if self.style == "gravity" and target_x is not None:
            # Move towards target
            dx = target_x - self.x
            dy = target_y - self.y
            dist = math.hypot(dx, dy) + 0.1
            self.vx = dx / dist * 15
            self.vy = dy / dist * 15
            self.x += self.vx
            self.y += self.vy
            if dist < 20: self.life = 0 # Die when reaching center
        else:
            self.x += self.vx
            self.y += self.vy
            
        self.life -= 4
        self.spin += self.spin_speed

    def draw(self, surface):
        if self.life > 0:
            color = (*self.color, int(self.life))
            
            # Draw a 6-pointed star/snowflake
            s_size = int(self.size * 3)
            s = pygame.Surface((s_size, s_size), pygame.SRCALPHA)
            center = s_size // 2
            
            rad = math.radians(self.spin)
            for i in range(3): 
                angle = rad + (i * math.pi / 3)
                dx = math.cos(angle) * self.size
                dy = math.sin(angle) * self.size
                pygame.draw.line(s, color, (center - dx, center - dy), (center + dx, center + dy), 2)
            
            surface.blit(s, (self.x - center, self.y - center))

def get_gesture(lm):
    # Tip IDs: 8=Index, 12=Middle, 16=Ring, 20=Pinky
    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pink_up = lm[20].y < lm[18].y

    if index_up and middle_up and ring_up and pink_up: return "open" 
    if index_up and middle_up and not ring_up and not pink_up: return "peace"
    if not index_up and not middle_up and not ring_up and not pink_up: return "fist"
    if index_up and not middle_up and not ring_up and not pink_up: return "point"
    return "other"

def main():
    download_model()
    
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Magical Snowflakes")
    clock = pygame.time.Clock()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    
    if not cap.isOpened():
        print("Error: Webcam not found.")
        return

    if not os.path.exists(MODEL_FILE):
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2)
    landmarker = vision.HandLandmarker.create_from_options(options)

    particles = []
    running = True

    while running:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(time.time() * 1000)
        
        result = landmarker.detect_for_video(mp_image, timestamp)

        bg_surface = pygame.surfarray.make_surface(cv2.transpose(rgb_frame))
        screen.blit(bg_surface, (0, 0))

        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                gesture = get_gesture(hand_lms)
                
                # Center of palm for shield
                palm_x = int(hand_lms[9].x * WIDTH) # Middle finger MCP roughly center
                palm_y = int(hand_lms[9].y * HEIGHT)
                
                if gesture == "open":
                    # BLIZZARD: Emit from all 5 fingertips
                    tips = [4, 8, 12, 16, 20]
                    for t in tips:
                        tx, ty = int(hand_lms[t].x * WIDTH), int(hand_lms[t].y * HEIGHT)
                        for _ in range(3):
                            particles.append(Snowflake(tx, ty, "normal"))
                            
                elif gesture == "point":
                    # ICE BEAM: Fast stream from index tip
                    tx, ty = int(hand_lms[8].x * WIDTH), int(hand_lms[8].y * HEIGHT)
                    for _ in range(8):
                        particles.append(Snowflake(tx, ty, "beam"))
                        
                elif gesture == "peace":
                    # FROST VORTEX: Between index and middle
                    ix, iy = int(hand_lms[8].x * WIDTH), int(hand_lms[8].y * HEIGHT)
                    mx, my = int(hand_lms[12].x * WIDTH), int(hand_lms[12].y * HEIGHT)
                    cx, cy = (ix + mx) // 2, (iy + my) // 2
                    for _ in range(6):
                        particles.append(Snowflake(cx, cy, "vortex"))
                        
                elif gesture == "fist":
                    # GRAVITY WELL: Particles spawn outside and suck in
                    for _ in range(10):
                        angle = random.uniform(0, 2 * math.pi)
                        dist = random.uniform(100, 200)
                        sx = palm_x + math.cos(angle) * dist
                        sy = palm_y + math.sin(angle) * dist
                        particles.append(Snowflake(sx, sy, "gravity"))
                    
                    # Store palm center for update
                    for p in particles: 
                        if p.style == "gravity": p.target = (palm_x, palm_y)
                
                else: 
                    # Default
                    pass

        # Update & Draw Particles
        for p in particles[:]:
            if p.style == "gravity" and hasattr(p, 'target'):
                 p.update(p.target[0], p.target[1])
            else:
                 p.update()
            
            p.draw(screen)
            if p.life <= 0 or p.y > HEIGHT + 50 or p.y < -50:
                particles.remove(p)

        pygame.display.flip()
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
