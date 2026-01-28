import cv2
import mediapipe as mp
import pygame
import random
import math
import time
import os
import urllib.request
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Configuration ---
WIDTH, HEIGHT = 1280, 720
FACE_MODEL_FILE = "face_landmarker.task"
HAND_MODEL_FILE = "hand_landmarker.task"
SEGMENTATION_MODEL_FILE = "selfie_segmenter.tflite"

FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
SEGMENTATION_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite"

BACKGROUND_IMAGE_PATH = "ice_background.jpg"

# --- Constants for Face Landmarks ---
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
FOREHEAD_TOP = 10

def download_models():
    models = [
        (FACE_MODEL_FILE, FACE_MODEL_URL),
        (HAND_MODEL_FILE, HAND_MODEL_URL),
        (SEGMENTATION_MODEL_FILE, SEGMENTATION_MODEL_URL)
    ]
    for file, url in models:
        if not os.path.exists(file):
            print(f"Downloading {file}...")
            try:
                urllib.request.urlretrieve(url, file)
                print("Download Complete.")
            except Exception as e:
                print(f"Error downloading {file}: {e}")

# --- Classes ---
class Diamond:
    def __init__(self, x, y, size, color, glow_intensity=1.0):
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.glow_intensity = glow_intensity
        self.angle = 0
        self.pulse_phase = random.uniform(0, 6.28)

    def draw(self, surface):
        self.pulse_phase += 0.1
        pulse = (math.sin(self.pulse_phase) + 1) * 0.2 + 0.8 
        current_size = self.size * pulse
        alpha = int(200 * self.glow_intensity)
        
        points = [
            (self.x, self.y - current_size),
            (self.x + current_size * 0.7, self.y),
            (self.x, self.y + current_size),
            (self.x - current_size * 0.7, self.y) 
        ]
        
        glow_surf = pygame.Surface((int(current_size * 4), int(current_size * 4)), pygame.SRCALPHA)
        glow_color = (*self.color, 50)
        center = int(current_size * 2)
        glow_points = [
            (center, center - current_size * 1.5),
            (center + current_size, center),
            (center, center + current_size * 1.5),
            (center - current_size, center)
        ]
        pygame.draw.polygon(glow_surf, glow_color, glow_points)
        surface.blit(glow_surf, (self.x - center, self.y - center))

        poly_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(poly_surf, (*self.color, alpha), points)
        surface.blit(poly_surf, (0,0))
        
        pygame.draw.circle(surface, (255, 255, 255), (self.x, self.y), current_size * 0.2)

class Snowflake:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.life = 255
        self.spin = random.uniform(0, 360)
        self.spin_speed = random.uniform(-5, 5)
        self.size = random.randint(4, 8)
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(2, 6) 
        self.color = (200, 230, 255)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 4
        self.spin += self.spin_speed
        self.size = max(0, self.size - 0.05)

    def draw(self, surface):
        if self.life > 0:
            color = (*self.color, int(self.life))
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

def draw_makeup_eye(surface, landmarks, indices, color):
    points = []
    for idx in indices:
        lm = landmarks[idx]
        points.append((int(lm.x * WIDTH), int(lm.y * HEIGHT)))
    
    if len(points) > 2:
        poly_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(poly_surf, color, points)
        surface.blit(poly_surf, (0, 0))

def is_palm_open(landmarks):
    return (landmarks[8].y < landmarks[6].y and
            landmarks[12].y < landmarks[10].y and
            landmarks[16].y < landmarks[14].y and
            landmarks[20].y < landmarks[18].y)

def main():
    download_models()
    
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Elsa Magic: Clap for Ice World!")
    clock = pygame.time.Clock()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Load Background Image
    ice_bg = None
    if os.path.exists(BACKGROUND_IMAGE_PATH):
        ice_bg = cv2.imread(BACKGROUND_IMAGE_PATH)
        ice_bg = cv2.resize(ice_bg, (WIDTH, HEIGHT))
        ice_bg = cv2.cvtColor(ice_bg, cv2.COLOR_BGR2RGB)
    else:
        print(f"Warning: {BACKGROUND_IMAGE_PATH} not found.")

    # Initialize Landmarkers
    face_base = python.BaseOptions(model_asset_path=FACE_MODEL_FILE)
    face_options = vision.FaceLandmarkerOptions(
        base_options=face_base,
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1)
    
    hand_base = python.BaseOptions(model_asset_path=HAND_MODEL_FILE)
    hand_options = vision.HandLandmarkerOptions(
        base_options=hand_base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2)

    # Initialize Segmenter
    segmenter_base = python.BaseOptions(model_asset_path=SEGMENTATION_MODEL_FILE)
    segmenter_options = vision.ImageSegmenterOptions(
        base_options=segmenter_base,
        running_mode=vision.RunningMode.VIDEO,
        output_category_mask=False,
        output_confidence_masks=True)

    try:
        face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
        hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        segmenter = vision.ImageSegmenter.create_from_options(segmenter_options)
    except Exception as e:
        print(f"Failed to load MediaPipe tasks: {e}")
        return

    particles = []
    
    # State
    show_ice_background = False
    frame_count = 0 # Strictly increasing timestamps for MediaPipe VIDEO mode

    running = True

    while running:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # Use frame_count as a strictly increasing timestamp
        timestamp = frame_count
        frame_count += 1
        
        # Detect
        face_result = face_landmarker.detect_for_video(mp_image, timestamp)
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp)
        


        # BACKGROUND RENDERING
        final_frame_rgb = rgb_frame
        
        if show_ice_background and ice_bg is not None:
             # Segmentation
             segmentation_result = segmenter.segment_for_video(mp_image, timestamp)
             # Get confidence mask for 'person'
             if segmentation_result.confidence_masks:
                 # Usually: [0]=Background, [1]=Person. If only one, assume it's the person mask?
                 if len(segmentation_result.confidence_masks) > 1:
                     confidence_mask = segmentation_result.confidence_masks[1]
                 else:
                     confidence_mask = segmentation_result.confidence_masks[0]
                     
                 mask_np = confidence_mask.numpy_view()
                 
                 # Robustly ensure shape is (HEIGHT, WIDTH, 1) for broadcasting with (HEIGHT, WIDTH, 3)
                 person_mask = mask_np.reshape(HEIGHT, WIDTH, 1)
                 
                 # Blend: pixel * mask + bg * (1 - mask)
                 final_frame_rgb = (rgb_frame * person_mask + ice_bg * (1 - person_mask)).astype(np.uint8)

        # Draw to Pygame
        # Transpose for Pygame (W, H, 3) -> (H, W, 3) ? No, Pygame needs (W, H, 3) but surfarray expects transpose usually?
        # Actually cv2 images are (H, W, C). Pygame surfaces are (W, H). 
        # Correct way usually: transpose swap axes 0 and 1 -> (W, H, C)
        bg_surface = pygame.surfarray.make_surface(cv2.transpose(final_frame_rgb))
        screen.blit(bg_surface, (0, 0))

        # --- FACE EFFECTS ---
        if face_result.face_landmarks:
            face_lms = face_result.face_landmarks[0]
            
            # Makeup
            shimmer_color = (0, 255, 255, 80) 
            draw_makeup_eye(screen, face_lms, LEFT_EYE, shimmer_color)
            draw_makeup_eye(screen, face_lms, RIGHT_EYE, shimmer_color)
            
            # Crown
            head_top = face_lms[FOREHEAD_TOP]
            hx, hy = int(head_top.x * WIDTH), int(head_top.y * HEIGHT)
            crown_y_base = hy - 60
            
            diamonds = [
                Diamond(hx, crown_y_base, 25, (100, 200, 255)),
                Diamond(hx - 50, crown_y_base + 10, 15, (100, 230, 255)),
                Diamond(hx + 50, crown_y_base + 10, 15, (100, 230, 255)),
                Diamond(hx - 90, crown_y_base + 30, 10, (100, 250, 255)),
                Diamond(hx + 90, crown_y_base + 30, 10, (100, 250, 255))
            ]
            for d in diamonds:
                d.draw(screen)

        # --- HAND EFFECTS (SNOW BEAM) ---
        if hand_result.hand_landmarks:
            for hand_lms in hand_result.hand_landmarks:
                if is_palm_open(hand_lms):
                    palm_x = int(hand_lms[9].x * WIDTH)
                    palm_y = int(hand_lms[9].y * HEIGHT)
                    
                    for _ in range(25):
                        p = Snowflake(palm_x, palm_y)
                        p.vx = random.uniform(-5, 5) 
                        p.vy = random.uniform(5, 12) 
                        particles.append(p)

        # Update & Draw Particles
        for p in particles[:]:
            p.update()
            p.draw(screen)
            if p.life <= 0 or p.y > HEIGHT + 50:
                particles.remove(p)

        pygame.display.flip()
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_b:
                    show_ice_background = not show_ice_background
                    print(f"Key 'B' Pressed! Background: {show_ice_background}")

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
