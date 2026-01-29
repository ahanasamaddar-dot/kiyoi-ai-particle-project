import cv2
import mediapipe as mp
import pygame
import random
import math
import time
import os
import urllib.request
import numpy as np
import ssl
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- macOS SSL Fix ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- Configuration ---
WIDTH, HEIGHT = 1280, 720
POSE_MODEL_FILE = "pose_landmarker_full.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
SEGMENTATION_MODEL_FILE = "selfie_segmenter.tflite"
SEGMENTATION_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite"
COSTUME_IMAGE_PATH = "elsa_costume.png"

# --- Landmark Groups ---
ARM_NODES = [11, 12, 13, 14, 15, 16]
HAND_NODES = [17, 18, 19, 20, 21, 22]
BODY_NODES = [23, 24]
LEG_NODES = [25, 26, 27, 28]
FOOT_NODES = [29, 30, 31, 32]

ALL_MAPPED_NODES = ARM_NODES + HAND_NODES + BODY_NODES + LEG_NODES + FOOT_NODES

# Connections for the "Map" look
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Arms
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22), # Hands
    (11, 23), (12, 24), (23, 24), # Torso
    (23, 25), (25, 27), (27, 29), (29, 31), # Left Leg/Foot
    (24, 26), (26, 28), (28, 30), (30, 32)  # Right Leg/Foot
]

# Midpoint calculation pairs for "Synthesis"
MIDPOINT_PAIRS = [
    (11, 12, "Chest"), (23, 24, "Pelvis"), # Core
    (11, 13, "L-UpperArm"), (12, 14, "R-UpperArm"),
    (13, 15, "L-Forearm"), (14, 16, "R-Forearm"),
    (23, 25, "L-Thigh"), (24, 26, "R-Thigh"),
    (25, 27, "L-Shin"), (26, 28, "R-Shin")
]

def download_models():
    models = [
        (POSE_MODEL_FILE, POSE_MODEL_URL),
        (SEGMENTATION_MODEL_FILE, SEGMENTATION_MODEL_URL)
    ]
    for file, url in models:
        if not os.path.exists(file):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(url, file)
            print("Download Complete.")

class Diamond:
    def __init__(self, x, y, size, color):
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.pulse_phase = random.uniform(0, 6.28)

    def draw(self, surface):
        self.pulse_phase += 0.1
        pulse = (math.sin(self.pulse_phase) + 1) * 0.2 + 0.8
        current_size = self.size * pulse
        
        points = [
            (self.x, self.y - current_size),
            (self.x + current_size * 0.7, self.y),
            (self.x, self.y + current_size),
            (self.x - current_size * 0.7, self.y)
        ]
        
        # Draw Glow
        glow_surf = pygame.Surface((int(current_size * 4), int(current_size * 4)), pygame.SRCALPHA)
        pygame.draw.polygon(glow_surf, (*self.color, 60), [
            (current_size*2, current_size*0.5),
            (current_size*3, current_size*2),
            (current_size*2, current_size*3.5),
            (current_size*1, current_size*2)
        ])
        surface.blit(glow_surf, (self.x - current_size*2, self.y - current_size*2))
        
        # Draw Main Diamond
        pygame.draw.polygon(surface, (*self.color, 200), points)
        pygame.draw.circle(surface, (255, 255, 255), (self.x, self.y), current_size * 0.2)

def main():
    download_models()
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Magical Body Mapping")
    clock = pygame.time.Clock()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Load Costume Image
    costume_img = None
    if os.path.exists(COSTUME_IMAGE_PATH):
        costume_img = cv2.imread(COSTUME_IMAGE_PATH)
        costume_img = cv2.cvtColor(costume_img, cv2.COLOR_BGR2RGB)
        print(f"DEBUG: Costume loaded successfully: {COSTUME_IMAGE_PATH}")
    else:
        print(f"Warning: {COSTUME_IMAGE_PATH} not found.")

    # Initialize MediaPipe Tasks
    pose_base = python.BaseOptions(model_asset_path=POSE_MODEL_FILE)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1)
    
    seg_base = python.BaseOptions(model_asset_path=SEGMENTATION_MODEL_FILE)
    seg_options = vision.ImageSegmenterOptions(
        base_options=seg_base,
        running_mode=vision.RunningMode.VIDEO,
        output_category_mask=False,
        output_confidence_masks=True)
    
    landmarker = vision.PoseLandmarker.create_from_options(pose_options)
    segmenter = vision.ImageSegmenter.create_from_options(seg_options)
    frame_count = 0

    running = True
    while running:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(cv2.resize(frame, (WIDTH, HEIGHT)), 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        pose_result = landmarker.detect_for_video(mp_image, frame_count)
        seg_result = segmenter.segment_for_video(mp_image, frame_count)
        frame_count += 1

        # Draw Frame to Pygame
        bg_surface = pygame.surfarray.make_surface(cv2.transpose(rgb_frame))
        screen.blit(bg_surface, (0, 0))

        if pose_result.pose_landmarks and seg_result.confidence_masks:
            pose_lms = pose_result.pose_landmarks[0]
            
            # 0. Get the Person Mask (Smoothed)
            conf_mask = seg_result.confidence_masks[0].numpy_view()
            conf_mask = cv2.GaussianBlur(conf_mask, (15, 15), 0)
            person_mask = conf_mask.reshape(HEIGHT, WIDTH)
            
            # 1. Dress Mapping Logic
            if costume_img is not None:
                try:
                    # Precision Alignment: Spine-Centered
                    # Landmarks: 11, 12 (Shoulders), 23, 24 (Hips), 25, 26 (Knees)
                    l_sh = pose_lms[11]
                    r_sh = pose_lms[12]
                    
                    # Center dress on your spine
                    spine_x = (l_sh.x + r_sh.x) / 2 * WIDTH
                    
                    # Scale width relative to shoulder distance (fitted look)
                    sh_dist = abs(r_sh.x - l_sh.x) * WIDTH
                    target_w = int(sh_dist * 2.8) 
                    
                    # Vertical range: Shoulders to slightly below Knees
                    y_top = int(min(l_sh.y, r_sh.y) * HEIGHT) - 40
                    y_bottom = int(max(pose_lms[25].y, pose_lms[26].y) * HEIGHT) + 50
                    
                    if y_bottom > y_top and target_w > 0:
                        target_h = y_bottom - y_top
                        target_x = int(spine_x - target_w / 2)
                        
                        # Use segmentation mask to cut out the person within this center box
                        v_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                        # Clip target area to screen dimensions
                        tx1 = max(0, target_x)
                        ty1 = max(0, y_top)
                        tx2 = min(WIDTH, target_x + target_w)
                        ty2 = min(HEIGHT, y_bottom)
                        
                        v_mask[ty1:ty2, tx1:tx2] = 255
                        
                        # Final mask combines the center-locked area with the person's real shape
                        dress_mask_float = person_mask * (v_mask / 255.0)
                        dress_mask = (dress_mask_float * 255).astype(np.uint8)
                        
                        dx, dy, dw, dh = cv2.boundingRect(dress_mask)
                        if dw > 0 and dh > 0:
                            local_mask = dress_mask[dy:dy+dh, dx:dx+dw]
                            
                            # Center-aware texture resize
                            th, tw = costume_img.shape[:2]
                            if (dh/dw) > (th/tw):
                                nw = int(th * (dw/dh))
                                sx = (tw - nw) // 2
                                crop = costume_img[:, sx:sx+nw]
                            else:
                                nh = int(tw * (dh/dw))
                                sy = (th - nh) // 2
                                crop = costume_img[sy:sy+nh, :]
                            
                            tex = cv2.resize(crop, (dw, dh))
                            alpha_channel = (local_mask * 0.95).astype(np.uint8)
                            rgba = np.dstack((tex, alpha_channel))
                            
                            dress_surf = pygame.image.frombuffer(rgba.tobytes(), (dw, dh), 'RGBA')
                            screen.blit(dress_surf, (dx, dy))
                except Exception as e:
                    print(f"Dress Error: {e}")

            # 2. Calculate Synthetic Midpoints
            synthetic_points = {}
            for idx1, idx2, name in MIDPOINT_PAIRS:
                if idx1 < len(pose_lms) and idx2 < len(pose_lms):
                    lm1 = pose_lms[idx1]
                    lm2 = pose_lms[idx2]
                    mx = (lm1.x + lm2.x) / 2
                    my = (lm1.y + lm2.y) / 2
                    synthetic_points[name] = (int(mx * WIDTH), int(my * HEIGHT))
            
            # Spine Node: Midpoint of Chest and Pelvis
            if "Chest" in synthetic_points and "Pelvis" in synthetic_points:
                cx, cy = synthetic_points["Chest"]
                px, py = synthetic_points["Pelvis"]
                synthetic_points["Spine"] = ((cx + px) // 2, (cy + py) // 2)

            # 2. Draw Connections (The "Map" Lines)
            for start_idx, end_idx in CONNECTIONS:
                if start_idx < len(pose_lms) and end_idx < len(pose_lms):
                    start = (int(pose_lms[start_idx].x * WIDTH), int(pose_lms[start_idx].y * HEIGHT))
                    end = (int(pose_lms[end_idx].x * WIDTH), int(pose_lms[end_idx].y * HEIGHT))
                    pygame.draw.line(screen, (100, 200, 255, 100), start, end, 2)

            # 3. Draw Original Magical Nodes
            for idx in ALL_MAPPED_NODES:
                if idx < len(pose_lms):
                    lm = pose_lms[idx]
                    if lm.presence > 0.5:
                        px, py = int(lm.x * WIDTH), int(lm.y * HEIGHT)
                        Diamond(px, py, 12, (150, 230, 255)).draw(screen)
            
            # 4. Draw Synthetic Nodes (Glowier!)
            for name, (sx, sy) in synthetic_points.items():
                Diamond(sx, sy, 10, (180, 250, 255)).draw(screen)

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