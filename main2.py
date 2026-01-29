import cv2
import mediapipe as mp
import pygame
import random
import math

# 1. SETUP
WIDTH, HEIGHT = 1280, 720
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# MediaPipe Initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

class Snowflake:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = random.randint(5, 12)
        self.life = 255 
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(2, 5) 

    def draw(self, surface):
        # Create a color with fading alpha
        color = (200, 230, 255)
        for i in range(6):
            ang = i * math.pi / 3
            end_x = self.x + math.cos(ang) * self.size
            end_y = self.y + math.sin(ang) * self.size
            pygame.draw.line(surface, color, (self.x, self.y), (end_x, end_y), 2)
        
        self.x += self.vx
        self.y += self.vy
        self.life -= 5 

particles = []

while True:
    screen.fill((10, 20, 50)) 
    
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Mini Camera Feed
    small_frame = cv2.resize(frame, (200, 150))
    cv_surface = pygame.surfarray.make_surface(small_frame.swapaxes(0, 1))
    screen.blit(cv_surface, (0, 0))

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            lm = hand_lms.landmark
            
            # GESTURE DETECTION: Check which fingers are up
            # Landmark indices: Index(8), Middle(12), Ring(16), Pinky(20)
            # A finger is "up" if its tip Y is less than its PIP joint Y
            index_up = lm[8].y < lm[6].y
            middle_up = lm[12].y < lm[10].y
            ring_up = lm[16].y < lm[14].y
            pinky_up = lm[20].y < lm[18].y

            # If ONLY the index finger is up
            if index_up and not middle_up and not ring_up and not pinky_up:
                ix, iy = int(lm[8].x * WIDTH), int(lm[8].y * HEIGHT)
                
                # Feedback circle
                pygame.draw.circle(screen, (0, 255, 255), (ix, iy), 15, 2)
                
                # Spawn snowflakes
                for _ in range(3):
                    particles.append(Snowflake(ix, iy))

    # 5. UPDATE PARTICLES
    for p in particles[:]:
        p.draw(screen)
        if p.life <= 0 or p.y > HEIGHT:
            particles.remove(p)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            exit()

    pygame.display.flip()
    clock.tick(30)

