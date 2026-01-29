import cv2
import mediapipe as mp
import pygame
import random
import math
import time
import os
import urllib.request
import threading
import numpy as np
import speech_recognition as sr
from PIL import Image, ImageSequence
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
BACKGROUND_AUDIO_PATH = "let_it_go.mp3"
OLAF_GIF_PATH = "olaf_gif2.gif"

# --- Constants for Face Landmarks ---
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

def load_gif_frames(path, size=None):
    if not os.path.exists(path):
        return []
    with Image.open(path) as img:
        frames = []
        for frame in ImageSequence.Iterator(img):
            frame_rgba = frame.convert('RGBA')
            if size:
                frame_rgba = frame_rgba.resize(size, Image.Resampling.LANCZOS)
            
            # Simple Background Removal: turn white/near-white to transparent
            data = np.array(frame_rgba)
            r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
            # Threshold for "white": all channels > 240
            white_mask = (r > 240) & (g > 240) & (b > 240)
            data[white_mask, 3] = 0 # Set alpha to 0 for white pixels
            
            frames.append(data)
        return frames

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
    def __init__(self, x, y, color=(200, 230, 255), gravity=0.0):
        self.x = x
        self.y = y
        self.life = 255
        self.spin = random.uniform(0, 360)
        self.spin_speed = random.uniform(-5, 5)
        self.size = random.uniform(4, 8)
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(2, 6) 
        self.color = color
        self.gravity = gravity

    def update(self):
        self.vy += self.gravity
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


def is_palm_open(landmarks):
    return (landmarks[8].y < landmarks[6].y and
            landmarks[12].y < landmarks[10].y and
            landmarks[16].y < landmarks[14].y and
            landmarks[20].y < landmarks[18].y)

def is_peace_sign(landmarks):
    # Tip IDs: 8=Index, 12=Middle, 16=Ring, 20=Pinky
    # PIP IDs: 6, 10, 14, 18
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_down = landmarks[16].y > landmarks[14].y
    pinky_down = landmarks[20].y > landmarks[18].y
    
    # Spread check: Index and Middle tips should be apart horizontally
    spread = abs(landmarks[8].x - landmarks[12].x)
    is_spread = spread > 0.04 # 4% of screen width
    
    return index_up and middle_up and ring_down and pinky_down and is_spread

def is_ok_sign(landmarks):
    # Thumb and Index tips touching
    d = math.sqrt((landmarks[4].x - landmarks[8].x)**2 + (landmarks[4].y - landmarks[8].y)**2)
    tips_close = d < 0.05
    # Others extended
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y
    return tips_close and middle_up and ring_up and pinky_up

def is_thumbs_up(landmarks):
    # Thumb tip well above IP joint
    thumb_up = landmarks[4].y < landmarks[3].y and landmarks[4].y < landmarks[2].y
    # Other fingers folded
    index_down = landmarks[8].x > landmarks[6].x if landmarks[0].x < landmarks[5].x else landmarks[8].x < landmarks[6].x
    # Simpler: tips closer to wrist than base joints
    others_folded = (landmarks[8].y > landmarks[6].y and 
                     landmarks[12].y > landmarks[10].y and 
                     landmarks[16].y > landmarks[14].y and 
                     landmarks[20].y > landmarks[18].y)
    return thumb_up and others_folded

def is_fist(landmarks):
    # All tips below PIP joints
    return (landmarks[8].y > landmarks[6].y and
            landmarks[12].y > landmarks[10].y and
            landmarks[16].y > landmarks[14].y and
            landmarks[20].y > landmarks[18].y and
            landmarks[4].y > landmarks[2].y)

class OlafActor:
    def __init__(self, frames):
        self.frames = frames
        self.active = False
        self.frame_idx = 0
        self.pos = (0, 0)
        self.spawn_time = 0
        self.duration = 4.0 # Seconds he stays visible
        
    def trigger(self):
        if not self.active:
            print("OLAF TRIGGERED BY VOICE!")
            self.active = True
            self.spawn_time = time.time()
            width = self.frames[0].shape[1]
            height = self.frames[0].shape[0]
            self.pos = (random.randint(50, WIDTH - width - 50), random.randint(HEIGHT // 2, HEIGHT - height - 50))
            self.frame_idx = 0

    def update(self):
        if self.active:
            now = time.time()
            if now - self.spawn_time > self.duration:
                self.active = False
            else:
                self.frame_idx = (self.frame_idx + 1) % len(self.frames)

    def draw(self, bg_img):
        if self.active and self.frames:
            frame = self.frames[self.frame_idx]
            h, w, _ = frame.shape
            x, y = self.pos
            
            # Extract RGB and Alpha
            olaf_rgb = frame[:, :, :3]
            olaf_alpha = frame[:, :, 3] / 255.0  # Normalize to [0.0, 1.0]
            olaf_alpha = np.expand_dims(olaf_alpha, axis=2) # Shape (H, W, 1)

            # Extract Background ROI
            bg_roi = bg_img[y:y+h, x:x+w]
            
            # Blending: Olaf * Alpha + Background * (1 - Alpha)
            blended_roi = (olaf_rgb * olaf_alpha + bg_roi * (1 - olaf_alpha)).astype(np.uint8)
            
            # Paste back
            bg_img[y:y+h, x:x+w] = blended_roi

def listen_for_olaf(olaf_actor):
    recognizer = sr.Recognizer()
    # Performance Tweaks
    recognizer.pause_threshold = 0.5  # Detect end of speech Faster (default 0.8)
    recognizer.energy_threshold = 300 
    recognizer.dynamic_energy_threshold = True
    
    mic = sr.Microphone()
    print("Voice Recognition for Elsa3 Started. Say 'Hi Olaf'!")
    
    # Pre-adjust for noise once at startup
    with mic as source:
        print("Adjusting for ambient noise... please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
    
    keywords = ["hi olaf", "hey olaf", "hi all of", "hey all of", "high olaf", "hay olaf"]
    
    while True:
        with mic as source:
            try:
                # listen for 4 seconds max, phrase length 3s
                audio = recognizer.listen(source, timeout=4, phrase_time_limit=3)
                text = recognizer.recognize_google(audio).lower()
                print(f"Elsa3 Heard: {text}")
                
                if any(kw in text for kw in keywords):
                    olaf_actor.trigger()
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except Exception as e:
                print(f"Voice Error: {e}")
                time.sleep(1)


def main():
    download_models()
    
    pygame.init()
    # High-quality audio init
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Elsa Magic: Clap for Ice World!")
    clock = pygame.time.Clock()

    # Load Audio
    try:
        if os.path.exists(BACKGROUND_AUDIO_PATH):
            pygame.mixer.music.load(BACKGROUND_AUDIO_PATH)
            pygame.mixer.music.set_volume(1.0)
            print(f"DEBUG: Audio loaded: {BACKGROUND_AUDIO_PATH}")
        else:
            print(f"Warning: {BACKGROUND_AUDIO_PATH} not found.")
    except Exception as e:
        print(f"Error loading audio: {e}")

    cap = cv2.VideoCapture(0)
    # Set to high res (1080p) to avoid driver cropping
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
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

    # Load Olaf Frames
    olaf_frames = load_gif_frames(OLAF_GIF_PATH, size=(200, 200))
    olaf = OlafActor(olaf_frames)

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

    # Start voice listening thread
    voice_thread = threading.Thread(target=listen_for_olaf, args=(olaf,), daemon=True)
    voice_thread.start()

    particles = []
    
    # State
    show_ice_background = False
    music_started = False
    frame_count = 0 # Strictly increasing timestamps for MediaPipe VIDEO mode
    last_gesture_time = 0
    gesture_cooldown = 1.0 # Seconds

    running = True

    while running:
        ret, frame = cap.read()
        if not ret: break

        # For OBSBOT and some webcams, capturing at 1080p and resizing 
        # manually avoids a "center crop" that looks like a zoom-in.
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # Use frame_count as a strictly increasing timestamp
        timestamp = frame_count
        frame_count += 1
        
        # Detect
        face_result = face_landmarker.detect_for_video(mp_image, timestamp)
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp)
        segmentation_result = None
        


        # GESTURE TOGGLE (Peace Sign)
        if hand_result.hand_landmarks:
            for hand_lms in hand_result.hand_landmarks:
                if is_peace_sign(hand_lms):
                    current_time = time.time()
                    if current_time - last_gesture_time > gesture_cooldown:
                        show_ice_background = not show_ice_background # Toggle
                        last_gesture_time = current_time
                        print(f"PEACE SIGN! Background: {show_ice_background}")

        # VISUAL FEEDBACK (Yellow Text)
        if time.time() - last_gesture_time < 0.5:
             font = pygame.font.SysFont(None, 80)
             text = font.render("PEACE SIGN DETECTED!", True, (255, 255, 0))
             screen.blit(text, (WIDTH//2 - 350, 100))


        # BACKGROUND & OLAF RENDERING
        base_bg = ice_bg if (show_ice_background and ice_bg is not None) else rgb_frame
        final_frame_rgb = base_bg
        
        # Always update Olaf
        olaf.update()
        
        # We need segmentation if we are showing the virtual background OR if Olaf is active
        # to keep Olaf behind the person.
        should_segment = show_ice_background or olaf.active
        
        if olaf.active or show_ice_background:
            temp_bg = base_bg.copy()
            if olaf.active:
                olaf.draw(temp_bg)
            
            if should_segment:
                segmentation_result = segmenter.segment_for_video(mp_image, timestamp)
                if segmentation_result.confidence_masks:
                    if len(segmentation_result.confidence_masks) > 1:
                        confidence_mask = segmentation_result.confidence_masks[1]
                    else:
                        confidence_mask = segmentation_result.confidence_masks[0]
                    
                    mask_np = confidence_mask.numpy_view()
                    person_mask = mask_np.reshape(HEIGHT, WIDTH, 1)
                    
                    # Blend: original camera person + temp_bg (with Olaf/Virtual BG) * (1 - mask)
                    final_frame_rgb = (rgb_frame * person_mask + temp_bg * (1 - person_mask)).astype(np.uint8)
                else:
                    final_frame_rgb = temp_bg
            else:
                final_frame_rgb = temp_bg



        # AUDIO TRIGGER
        if show_ice_background:
            if not music_started:
                print("DEBUG: Starting music playback for the first time...")
                pygame.mixer.music.play(-1)
                music_started = True
            else:
                # If already started, just unpause
                pygame.mixer.music.set_volume(1.0)
                pygame.mixer.music.unpause()
        else:
            # We use pause() instead of stop() to keep the position
            if music_started:
                pygame.mixer.music.pause()

        # Draw to Pygame
        # Transpose for Pygame (W, H, 3) -> (H, W, 3) ? No, Pygame needs (W, H, 3) but surfarray expects transpose usually?
        # Actually cv2 images are (H, W, C). Pygame surfaces are (W, H). 
        # Correct way usually: transpose swap axes 0 and 1 -> (W, H, C)
        bg_surface = pygame.surfarray.make_surface(cv2.transpose(final_frame_rgb))
        screen.blit(bg_surface, (0, 0))

        # --- FACE EFFECTS ---
        if face_result.face_landmarks:
            face_lms = face_result.face_landmarks[0]
            
            
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
                    
                    for _ in range(60):
                        p = Snowflake(palm_x, palm_y)
                        p.vx = random.uniform(-6, 6) 
                        p.vy = random.uniform(5, 15)
                        p.size = random.uniform(3, 10)
                        particles.append(p)
                
                elif is_ok_sign(hand_lms):
                    # Magical Pixels: Crystalline Blue
                    palm_x = int(hand_lms[9].x * WIDTH)
                    palm_y = int(hand_lms[9].y * HEIGHT)
                    color = (180, 240, 255) # Crystalline Blue
                    for _ in range(30):
                        p = Snowflake(palm_x, palm_y, color=color)
                        p.vx = random.uniform(-3, 3)
                        p.vy = random.uniform(1, 4)
                        p.size = random.uniform(5, 15)
                        particles.append(p)

                elif is_thumbs_up(hand_lms):
                    # Frozen Sun: Bright White/Ice Blue
                    palm_x = int(hand_lms[9].x * WIDTH)
                    palm_y = int(hand_lms[9].y * HEIGHT)
                    color = (255, 255, 255) # Pure White
                    for _ in range(40):
                        p = Snowflake(palm_x, palm_y, color=color)
                        p.vx = random.uniform(-10, 10)
                        p.vy = random.uniform(-5, 5)
                        p.size = random.uniform(2, 6)
                        particles.append(p)

                elif is_fist(hand_lms):
                    # Hail Storm: Slate Grey
                    palm_x = int(hand_lms[9].x * WIDTH)
                    palm_y = int(hand_lms[9].y * HEIGHT)
                    for _ in range(25):
                        p = Snowflake(palm_x, palm_y, color=(150, 150, 170), gravity=0.5)
                        p.vx = random.uniform(-2, 2)
                        p.vy = random.uniform(5, 12)
                        p.size = random.uniform(8, 14)
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