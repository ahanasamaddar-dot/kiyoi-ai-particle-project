import cv2
import pygame
import random
import time
import os
import threading
import numpy as np
from PIL import Image, ImageSequence
import speech_recognition as sr

# --- Configuration ---
WIDTH, HEIGHT = 640, 480
OLAF_GIF_PATH = "olaf_gif2.gif"

def load_gif_frames(path, size=None):
    if not os.path.exists(path):
        return []
    with Image.open(path) as img:
        frames = []
        for frame in ImageSequence.Iterator(img):
            frame_rgba = frame.convert('RGBA')
            if size:
                frame_rgba = frame_rgba.resize(size, Image.Resampling.LANCZOS)
            
            data = np.array(frame_rgba)
            r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
            white_mask = (r > 240) & (g > 240) & (b > 240)
            data[white_mask, 3] = 0 
            frames.append(data)
        return frames

class OlafActor:
    def __init__(self, frames):
        self.frames = frames
        self.active = False
        self.frame_idx = 0
        self.pos = (WIDTH // 2 - 100, HEIGHT // 2 - 100)
        self.spawn_time = 0
        self.duration = 5.0 
        
    def trigger(self):
        if not self.active:
            print("OLAF TRIGGERED!")
            self.active = True
            self.spawn_time = time.time()
            self.frame_idx = 0

    def update(self):
        if self.active:
            now = time.time()
            if now - self.spawn_time > self.duration:
                self.active = False
            else:
                self.frame_idx = (self.frame_idx + 1) % len(self.frames)

    def draw(self, surface):
        if self.active and self.frames:
            frame = self.frames[self.frame_idx]
            # Convert to Pygame surface with Alpha
            # frame is (H, W, 4) in RGBA
            olaf_surf = pygame.image.frombuffer(frame.flatten(), (frame.shape[1], frame.shape[0]), 'RGBA')
            surface.blit(olaf_surf, self.pos)

def listen_for_olaf(olaf_actor):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    print("Voice Recognition Started. Say 'Hi Olaf'!")
    
    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
                text = recognizer.recognize_google(audio).lower()
                print(f"Heard: {text}")
                if "hi olaf" in text or "hey olaf" in text:
                    olaf_actor.trigger()
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except Exception as e:
                print(f"Voice Error: {e}")
                time.sleep(1)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Say 'Hi Olaf'!")
    clock = pygame.time.Clock()

    olaf_frames = load_gif_frames(OLAF_GIF_PATH, size=(200, 200))
    olaf = OlafActor(olaf_frames)

    # Start voice listening in a separate thread
    voice_thread = threading.Thread(target=listen_for_olaf, args=(olaf,), daemon=True)
    voice_thread.start()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        olaf.update()
        
        screen.fill((30, 30, 50)) # Dark blue background
        
        font = pygame.font.SysFont(None, 36)
        text = font.render("Say 'Hi Olaf' to summon him!", True, (200, 200, 255))
        screen.blit(text, (WIDTH // 2 - 180, 50))
        
        olaf.draw(screen)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
