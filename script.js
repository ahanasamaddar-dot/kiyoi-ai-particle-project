import {
    HandLandmarker,
    FaceLandmarker,
    ImageSegmenter,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs";

// --- Constants & Config ---
const WIDTH = 1280;
const HEIGHT = 720;
const GESTURE_COOLDOWN = 1000; // ms

// --- Elements ---
const video = document.getElementById('webcam');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('start-btn');
const overlay = document.getElementById('overlay');
const loading = document.getElementById('loading');
const arView = document.getElementById('ar-view');
const statusBadge = document.getElementById('gesture-status');
const bgMusic = document.getElementById('bg-music');
const iceBgImg = document.getElementById('ice-bg');
const olafGif = document.getElementById('olaf-gif');

// --- State ---
let handLandmarker;
let faceLandmarker;
let imageSegmenter;
let lastVideoTime = -1;
let lastGestureTime = 0;
let showIceBackground = false;
let isMuted = false;
let particles = [];
let voiceActive = false;
let olafActive = false;
let olafPos = { x: 0, y: 0 };
let olafStartTime = 0;
let olafDuration = 4000; // ms

// --- Initialization ---
async function init() {
    loading.classList.remove('hidden');

    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );

    [handLandmarker, faceLandmarker, imageSegmenter] = await Promise.all([
        HandLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" },
            runningMode: "VIDEO",
            numHands: 2
        }),
        FaceLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" },
            runningMode: "VIDEO"
        }),
        ImageSegmenter.createFromOptions(vision, {
            baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite" },
            runningMode: "VIDEO",
            outputConfidenceMasks: true
        })
    ]);

    await startWebcam();
    loading.classList.add('hidden');
    requestAnimationFrame(renderLoop);
    initVoiceRecognition();
}

async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: WIDTH, height: HEIGHT }
        });
        video.srcObject = stream;
        canvas.width = WIDTH;
        canvas.height = HEIGHT;
    } catch (err) {
        console.error("Webcam Error:", err);
        showStatus("Please allow camera access!");
    }
}

// --- Classes ---
class Snowflake {
    constructor(x, y, color = 'rgba(200, 230, 255, 1)', gravity = 0) {
        this.x = x;
        this.y = y;
        this.life = 1.0;
        this.size = Math.random() * 4 + 4;
        this.vx = (Math.random() - 0.5) * 4;
        this.vy = Math.random() * 4 + 2;
        this.color = color;
        this.gravity = gravity;
        this.spin = Math.random() * 360;
        this.spinSpeed = (Math.random() - 0.5) * 5;
    }

    update() {
        this.vy += this.gravity;
        this.x += this.vx;
        this.y += this.vy;
        this.life -= 0.015;
        this.spin += this.spinSpeed;
    }

    draw(ctx) {
        if (this.life <= 0) return;
        ctx.save();
        ctx.translate(this.x, this.y);
        ctx.rotate(this.spin * Math.PI / 180);
        ctx.strokeStyle = this.color.replace('1)', `${this.life})`);
        ctx.lineWidth = 2;
        for (let i = 0; i < 3; i++) {
            ctx.rotate(Math.PI / 3);
            ctx.beginPath();
            ctx.moveTo(-this.size, 0);
            ctx.lineTo(this.size, 0);
            ctx.stroke();
        }
        ctx.restore();
    }
}

class Diamond {
    constructor(x, y, size, color) {
        this.x = x;
        this.size = size;
        this.y = y;
        this.color = color;
        this.phase = Math.random() * Math.PI * 2;
    }

    draw(ctx) {
        this.phase += 0.1;
        const pulse = Math.sin(this.phase) * 0.2 + 0.8;
        const currentSize = this.size * pulse;

        ctx.beginPath();
        ctx.moveTo(this.x, this.y - currentSize);
        ctx.lineTo(this.x + currentSize * 0.7, this.y);
        ctx.lineTo(this.x, this.y + currentSize);
        ctx.lineTo(this.x - currentSize * 0.7, this.y);
        ctx.closePath();

        ctx.fillStyle = this.color;
        ctx.fill();
        ctx.shadowBlur = 15;
        ctx.shadowColor = this.color;
    }
}

// --- Gesture Logic ---
function isPalmOpen(landmarks) {
    return (landmarks[8].y < landmarks[6].y &&
        landmarks[12].y < landmarks[10].y &&
        landmarks[16].y < landmarks[14].y &&
        landmarks[20].y < landmarks[18].y);
}

function isPeaceSign(landmarks) {
    const indexUp = landmarks[8].y < landmarks[6].y;
    const middleUp = landmarks[12].y < landmarks[10].y;
    const ringDown = landmarks[16].y > landmarks[14].y;
    const pinkyDown = landmarks[20].y > landmarks[18].y;
    const spread = Math.abs(landmarks[8].x - landmarks[12].x);
    return indexUp && middleUp && ringDown && pinkyDown && spread > 0.04;
}

function isOkSign(landmarks) {
    const d = Math.sqrt(Math.pow(landmarks[4].x - landmarks[8].x, 2) + Math.pow(landmarks[4].y - landmarks[8].y, 2));
    return d < 0.05 && landmarks[12].y < landmarks[10].y && landmarks[16].y < landmarks[14].y;
}

// --- Render Loop ---
function renderLoop() {
    if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const startTimeMs = performance.now();

        // 1. Detections
        const handResults = handLandmarker.detectForVideo(video, startTimeMs);
        const faceResults = faceLandmarker.detectForVideo(video, startTimeMs);

        // 2. Clear Canvas
        ctx.clearRect(0, 0, WIDTH, HEIGHT);

        // 3. Handle Segmentation & Background
        if (showIceBackground) {
            imageSegmenter.segmentForVideo(video, startTimeMs, (result) => {
                const mask = result.confidenceMasks[0].getAsUint8Array();

                // Draw Ice Background
                ctx.drawImage(iceBgImg, 0, 0, WIDTH, HEIGHT);

                // Create a temporary canvas for the person mask
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = WIDTH;
                tempCanvas.height = HEIGHT;
                const tempCtx = tempCanvas.getContext('2d');

                // Draw Video Frame (flipped)
                tempCtx.save();
                tempCtx.scale(-1, 1);
                tempCtx.drawImage(video, -WIDTH, 0, WIDTH, HEIGHT);
                tempCtx.restore();

                // Apply Mask
                const imgData = tempCtx.getImageData(0, 0, WIDTH, HEIGHT);
                for (let i = 0; i < mask.length; i++) {
                    imgData.data[i * 4 + 3] = mask[i]; // Set alpha from mask
                }
                tempCtx.putImageData(imgData, 0, 0);

                // Composite onto main canvas
                ctx.drawImage(tempCanvas, 0, 0);
            });
        } else {
            // Standard flipped view
            ctx.save();
            ctx.scale(-1, 1);
            ctx.drawImage(video, -WIDTH, 0, WIDTH, HEIGHT);
            ctx.restore();
        }

        // 4. Draw Olaf
        if (olafActive) {
            const now = Date.now();
            if (now - olafStartTime > olafDuration) {
                olafActive = false;
            } else {
                ctx.drawImage(olafGif, olafPos.x, olafPos.y, 200, 200);
            }
        }

        // 5. Handle Hand Gestures
        if (handResults.landmarks) {
            handResults.landmarks.forEach(landmarks => {
                // Flip X for particles
                const rawPalmX = landmarks[9].x * WIDTH;
                const palmX = WIDTH - rawPalmX; // Mirrored
                const palmY = landmarks[9].y * HEIGHT;

                if (isPeaceSign(landmarks)) {
                    const now = Date.now();
                    if (now - lastGestureTime > GESTURE_COOLDOWN) {
                        toggleBackground();
                        lastGestureTime = now;
                        showStatus("Peace Sign Detected! ❄️");
                    }
                } else if (isPalmOpen(landmarks)) {
                    for (let i = 0; i < 5; i++) {
                        particles.push(new Snowflake(palmX, palmY));
                    }
                } else if (isOkSign(landmarks)) {
                    for (let i = 0; i < 3; i++) {
                        particles.push(new Snowflake(palmX, palmY, 'rgba(180, 240, 255, 1)'));
                    }
                }
            });
        }

        // 6. Handle Face Crown
        if (faceResults.faceLandmarks && faceResults.faceLandmarks.length > 0) {
            const lms = faceResults.faceLandmarks[0];
            const forehead = lms[10];
            const fx = WIDTH - (forehead.x * WIDTH);
            const fy = forehead.y * HEIGHT - 80;

            const crown = [
                new Diamond(fx, fy, 20, '#64c8ff'),
                new Diamond(fx - 40, fy + 10, 12, '#64e6ff'),
                new Diamond(fx + 40, fy + 10, 12, '#64e6ff')
            ];
            crown.forEach(d => d.draw(ctx));
        }

        // 7. Update & Draw Particles
        particles.forEach((p, idx) => {
            p.update();
            p.draw(ctx);
            if (p.life <= 0) particles.splice(idx, 1);
        });
    }

    requestAnimationFrame(renderLoop);
}

// --- Utilities ---
function toggleBackground() {
    showIceBackground = !showIceBackground;
    if (showIceBackground) {
        bgMusic.play();
        bgMusic.volume = 1;
    } else {
        bgMusic.pause();
    }
}

function showStatus(text) {
    statusBadge.innerText = text;
    setTimeout(() => {
        statusBadge.innerText = "Ready for Magic";
    }, 2000);
}

function triggerOlaf() {
    if (!olafActive) {
        olafActive = true;
        olafStartTime = Date.now();
        olafPos = {
            x: Math.random() * (WIDTH - 250) + 50,
            y: Math.random() * (HEIGHT - 300) + HEIGHT / 2
        };
        showStatus("Hi Olaf! ☃️");
        console.log("Olaf Triggered!");
    }
}

// --- Voice Recognition ---
function initVoiceRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.lang = 'en-US';

    recognition.onresult = (event) => {
        const text = event.results[event.results.length - 1][0].transcript.toLowerCase();
        console.log("Heard:", text);
        if (text.includes("hi olaf") || text.includes("hey olaf") || text.includes("olaf")) {
            triggerOlaf();
        }
    };

    recognition.onerror = (e) => console.log("Speech error:", e);
    recognition.start();
}

// --- Listeners ---
startBtn.addEventListener('click', () => {
    overlay.classList.add('fade-out');
    arView.classList.remove('hidden');
    init();
});

document.getElementById('toggle-bg-btn').addEventListener('click', toggleBackground);
document.getElementById('mute-btn').addEventListener('click', () => {
    isMuted = !isMuted;
    bgMusic.muted = isMuted;
    document.getElementById('mute-btn').innerText = isMuted ? '🔇' : '🔊';
});
