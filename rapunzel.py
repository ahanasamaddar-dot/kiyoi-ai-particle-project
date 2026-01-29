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

BACKGROUND_IMAGE_PATH = "rapunzel_lantern.jpg"
BACKGROUND_AUDIO_PATH = "i_see_the_light.mp3"

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
class Petal:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = random.uniform(5, 12)
        self.life = 255
        self.vy = random.uniform(2, 5)
        self.vx = random.uniform(-2, 2)
        self.angle = random.uniform(0, 360)
        self.spin = random.uniform(-5, 5)
        self.color = (255, 215, 0) # Gold

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.angle += self.spin
        self.life -= 3

    def draw(self, surface):
        if self.life > 0:
            s_size = int(self.size * 2)
            s = pygame.Surface((s_size, s_size), pygame.SRCALPHA)
            pygame.draw.ellipse(s, (*self.color, self.life), (0, 0, self.size, self.size*0.6))
            rotated = pygame.transform.rotate(s, self.angle)
            surface.blit(rotated, (self.x, self.y))

class SunSpark:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.life = 255
        self.angle = random.uniform(0, 6.28)
        self.speed = random.uniform(4, 10)
        self.length = random.uniform(10, 20)
        self.color = (255, 255, 200)

    def update(self):
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        self.life -= 10

    def draw(self, surface):
        if self.life > 0:
            dx = math.cos(self.angle) * self.length
            dy = math.sin(self.angle) * self.length
            pygame.draw.line(surface, (*self.color, self.life), (self.x, self.y), (self.x + dx, self.y + dy), 2)

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

def is_peace_sign(landmarks):
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_down = landmarks[16].y > landmarks[14].y
    pinky_down = landmarks[20].y > landmarks[18].y
    spread = abs(landmarks[8].x - landmarks[12].x)
    return index_up and middle_up and ring_down and pinky_down and spread > 0.04

def is_ok_sign(landmarks):
    d = math.sqrt((landmarks[4].x - landmarks[8].x)**2 + (landmarks[4].y - landmarks[8].y)**2)
    tips_close = d < 0.05
    return tips_close and landmarks[12].y < landmarks[10].y # Index/Thumb touch + middle up

def is_fist(landmarks):
    return (landmarks[8].y > landmarks[6].y and
            landmarks[12].y > landmarks[10].y and
            landmarks[16].y > landmarks[14].y and
            landmarks[20].y > landmarks[18].y and
            landmarks[4].y > landmarks[2].y)

def main():
    download_models()
    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Rapunzel Magic: Floating Lanterns!")
    clock = pygame.time.Clock()

    # Load Audio
    try:
        if os.path.exists(BACKGROUND_AUDIO_PATH):
            pygame.mixer.music.load(BACKGROUND_AUDIO_PATH)
            pygame.mixer.music.set_volume(1.0)
            music_loaded = True
            print(f"DEBUG: Audio loaded: {BACKGROUND_AUDIO_PATH}")
        else:
            print(f"Warning: {BACKGROUND_AUDIO_PATH} not found.")
    except Exception as e:
        print(f"Error loading audio: {e}")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Background
    lantern_bg = None
    if os.path.exists(BACKGROUND_IMAGE_PATH):
        lantern_bg = cv2.imread(BACKGROUND_IMAGE_PATH)
        lantern_bg = cv2.resize(lantern_bg, (WIDTH, HEIGHT))
        lantern_bg = cv2.cvtColor(lantern_bg, cv2.COLOR_BGR2RGB)

    # Tiara
    tiara_img = None
    if os.path.exists("rapunzel_tiara.jpg"):
        # Load and set colorkey/alpha if needed, but since it's a JPG, we might need to handle black/white background
        tiara_img = pygame.image.load("rapunzel_tiara.jpg").convert_alpha()
        # Resize to reasonable size
        tiara_img = pygame.transform.scale(tiara_img, (300, 150))

    # MediaPipe Setup
    face_landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=FACE_MODEL_FILE),
        running_mode=vision.RunningMode.VIDEO, num_faces=1))
    hand_landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL_FILE),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
    segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=SEGMENTATION_MODEL_FILE),
        running_mode=vision.RunningMode.VIDEO, output_confidence_masks=True))

    particles = []
    show_background = False
    music_loaded = False
    music_started = False
    frame_count = 0
    last_gesture_time = 0

    running = True
    while running:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = frame_count
        frame_count += 1

        face_result = face_landmarker.detect_for_video(mp_image, timestamp)
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp)

        # Gesture Toggle
        if hand_result.hand_landmarks:
            for hand_lms in hand_result.hand_landmarks:
                if is_peace_sign(hand_lms):
                    current_time = time.time()
                    if current_time - last_gesture_time > 1.0:
                        show_background = not show_background
                        last_gesture_time = current_time

        # Background Rendering
        final_frame_rgb = rgb_frame
        if show_background and lantern_bg is not None:
             seg_result = segmenter.segment_for_video(mp_image, timestamp)
             if seg_result.confidence_masks:
                 mask = seg_result.confidence_masks[1 if len(seg_result.confidence_masks) > 1 else 0].numpy_view()
                 person_mask = mask.reshape(HEIGHT, WIDTH, 1)
                 final_frame_rgb = (rgb_frame * person_mask + lantern_bg * (1 - person_mask)).astype(np.uint8)

        # Audio
        if show_background and music_loaded:
            if not music_started:
                pygame.mixer.music.play(-1)
                music_started = True
            else: pygame.mixer.music.unpause()
        elif music_loaded:
            pygame.mixer.music.pause()

        bg_surface = pygame.surfarray.make_surface(cv2.transpose(final_frame_rgb))
        screen.blit(bg_surface, (0, 0))

        # Face Effects
        if face_result.face_landmarks:
            face_lms = face_result.face_landmarks[0]
            draw_makeup_eye(screen, face_lms, LEFT_EYE, (255, 215, 0, 70))
            draw_makeup_eye(screen, face_lms, RIGHT_EYE, (255, 215, 0, 70))
            
            # Rapunzel Tiara
            if tiara_img:
                top = face_lms[FOREHEAD_TOP]
                tx, ty = int(top.x * WIDTH), int(top.y * HEIGHT)
                # Offset to sit on forehead
                screen.blit(tiara_img, (tx - 150, ty - 130))

        # Hand Effects
        if hand_result.hand_landmarks:
            for hand_lms in hand_result.hand_landmarks:
                px, py = int(hand_lms[9].x * WIDTH), int(hand_lms[9].y * HEIGHT)
                if is_ok_sign(hand_lms):
                    for _ in range(5): particles.append(SunSpark(px, py))
                elif is_fist(hand_lms):
                    for _ in range(3): particles.append(Petal(px, py))

        for p in particles[:]:
            p.update()
            p.draw(screen)
            if p.life <= 0 or p.y < -100 or p.y > HEIGHT + 100: particles.remove(p)

        pygame.display.flip()
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
