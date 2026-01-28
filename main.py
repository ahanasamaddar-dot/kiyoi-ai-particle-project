import cv2
import mediapipe as mp
import pygame
import random
import math
import numpy as np
import time

# --- MIGRATION: Imports for Tasks API ---
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Configuration ---
WIDTH, HEIGHT = 1280, 720
BG_COLOR = (10, 10, 20)

# --- Initialize Pygame ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Kiyoi AI Particle Project")
clock = pygame.time.Clock()

class Particle:
    def __init__(self, x, y, color, behavior="trail"):
        self.x = x
        self.y = y
        self.color = color
        self.behavior = behavior
        self.size = random.uniform(4, 8)
        self.life = 255
        self.decay = random.uniform(4, 8)
        
        if behavior == "burst":
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(4, 12)
            self.vx, self.vy = math.cos(angle) * speed, math.sin(angle) * speed
        elif behavior == "swirl":
            self.vx, self.vy = random.uniform(-1, 1), random.uniform(-1, 1)
        else: # Trail
            self.vx, self.vy = random.uniform(-1, 1), random.uniform(-3, -1)

    def update(self):
        if self.behavior == "swirl":
            t = pygame.time.get_ticks() * 0.01
            # Spiral motion
            self.x += math.sin(t) * 6 + self.vx
            self.y += math.cos(t) * 6 + self.vy
        else:
            self.x += self.vx
            self.y += self.vy
            
        self.life -= self.decay
        self.size = max(0, self.size - 0.1)

    def draw(self, surface):
        if self.life > 0:
            s = pygame.Surface((int(self.size * 2), int(self.size * 2)), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.color, int(self.life)), (int(self.size), int(self.size)), int(self.size))
            surface.blit(s, (self.x - self.size, self.y - self.size))

def get_gesture(landmarks):
    # Logic: Finger is 'up' if tip Y < joint Y
    # Indices: 8=Index, 12=Middle, 16=Ring, 20=Pinky
    ids = [8, 12, 16, 20] 
    up = [landmarks[i].y < landmarks[i-2].y for i in ids]
    
    if all(up): return "burst"   
    if up[0] and up[1] and not any(up[2:]): return "swirl"
    return "trail"

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # --- Initialize Hand Landmarker (Tasks API) ---
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO)
    landmarker = vision.HandLandmarker.create_from_options(options)

    particles = []
    running = True

    print("Starting Magical Hands... Press 'q' to quit.")
    start_time_ms = int(time.time() * 1000)

    while running:
        # 1. Capture Frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        # Mirror
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MP Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect (VIDEO mode requires timestamp)
        current_time_ms = int(time.time() * 1000)
        detection_result = landmarker.detect_for_video(mp_image, current_time_ms)

        # 2. Render Background
        bg_image = cv2.transpose(rgb_frame)
        bg_surface = pygame.surfarray.make_surface(bg_image)
        screen.blit(bg_surface, (0, 0))

        # 3. Handle Hand Tracking & Particles
        if detection_result.hand_landmarks:
            for hand_lms in detection_result.hand_landmarks:
                # hand_lms is a list of NormalizedLandmark
                tip = hand_lms[8] 
                x, y = int(tip.x * WIDTH), int(tip.y * HEIGHT)
                
                gesture = get_gesture(hand_lms)
                
                if gesture == "burst":
                    color, count = (255, 100, 255), 5 
                elif gesture == "swirl":
                    color, count = (100, 255, 255), 3 
                else: 
                    color, count = (255, 200, 50), 2

                for _ in range(count):
                    particles.append(Particle(x, y, color, gesture))

        # 4. Update and Render Particles
        for p in particles[:]:
            p.update()
            p.draw(screen)
            if p.life <= 0: particles.remove(p)

        # 5. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()