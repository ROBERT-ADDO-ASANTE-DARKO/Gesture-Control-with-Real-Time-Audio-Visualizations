import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import random
import time
import math
import pyaudio
import struct
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import numpy.fft as fft

def compute_spectrum(audio_data):
    spectrum = fft.fft(audio_data)
    magnitude = np.abs(spectrum)
    return magnitude

def draw_spectrum_analyzer(img, spectrum):
    num_bins = len(spectrum)
    bin_width = img.shape[1] // num_bins
    for i in range(num_bins):
        height = int(spectrum[i] * 0.1)  # Scale the height
        cv2.rectangle(img, (i * bin_width, img.shape[0] - height),
                      ((i + 1) * bin_width, img.shape[0]), (0, 255, 0), -1)

'''
def draw_3d_waves(img, audio_level):
    wave_height = int(audio_level * 150)
    for i in range(100):
        x = 200 + i * 4
        y = 250 + int(math.sin(i / 5 + time.time() * 3) * wave_height)
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
'''

def draw_circular_visualizer(img, audio_level):
    center = (img.shape[1] // 2, img.shape[0] // 2)
    num_bars = 36
    for i in range(num_bars):
        angle = 2 * np.pi * i / num_bars
        length = int(audio_level * 100)
        end_point = (int(center[0] + length * np.cos(angle)),
                     int(center[1] + length * np.sin(angle)))
        cv2.line(img, center, end_point, (255, 255, 0), 2)

particles = []

def draw_particles(img, audio_level):
    global particles

    # Add new particles if there are fewer than 100
    if len(particles) < 100:
        # Each particle is represented as (x, y, color)
        # where color is a tuple (B, G, R)
        particles.append((
            np.random.randint(0, img.shape[1]),  # x position
            np.random.randint(0, img.shape[0]),  # y position
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Random color (B, G, R)
        ))

    # Update and draw particles
    for i, (x, y, color) in enumerate(particles):
        # Draw the particle with its assigned color
        cv2.circle(img, (x, y), 2, color, -1)

        # Update particle position with random motion
        particles[i] = (
            x + np.random.randint(-2, 3),  # Random horizontal movement
            y + np.random.randint(-2, 3),  # Random vertical movement
            color  # Keep the same color
        )

'''
def draw_histogram(img, audio_level):
    num_bins = 10
    bin_width = img.shape[1] // num_bins
    for i in range(num_bins):
        height = int(audio_level * np.random.randint(50, 200))
        cv2.rectangle(img, (i * bin_width, 450 - height),
                      ((i + 1) * bin_width, 450), (0, 255, 255), -1)
'''

def draw_water_ripple(img, audio_level):
    center = (img.shape[1] // 2, img.shape[0] // 2)
    radius = int(audio_level * 100)
    cv2.circle(img, center, radius, (0, 0, 255), 2)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)
print(hands)
mpDraw = mp.solutions.drawing_utils

# Initialize audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()  # (-65.25, 0.0)
minVol, maxVol = volRange[0], volRange[1]

# Audio Stream Setup
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                input=True,
                frames_per_buffer=1024)

# Finger landmarks
index_tip = 8
thumb_tip = 4
middle_tip = 12
ring_tip = 16
pinky_tip = 20
wrist = 0

# Gesture detection variables
gesture_detected = {"left": False, "right": False}
last_gesture_time = {"left": 0, "right": 0}
gesture_cooldown = 1  # Cooldown time in seconds


def is_fist(lmList):
    """Check if the hand is in a fist gesture."""
    if len(lmList) < 21:
        return False

    # Check if fingertips are close to the palm
    thumb_tip_dist = np.hypot(lmList[thumb_tip][0] - lmList[wrist][0], lmList[thumb_tip][1] - lmList[wrist][1])
    index_tip_dist = np.hypot(lmList[index_tip][0] - lmList[wrist][0], lmList[index_tip][1] - lmList[wrist][1])
    middle_tip_dist = np.hypot(lmList[middle_tip][0] - lmList[wrist][0], lmList[middle_tip][1] - lmList[wrist][1])
    ring_tip_dist = np.hypot(lmList[ring_tip][0] - lmList[wrist][0], lmList[ring_tip][1] - lmList[wrist][1])
    pinky_tip_dist = np.hypot(lmList[pinky_tip][0] - lmList[wrist][0], lmList[pinky_tip][1] - lmList[wrist][1])

    # If all fingertips are close to the wrist, it's a fist
    return (thumb_tip_dist < 50 and index_tip_dist < 50 and
            middle_tip_dist < 50 and ring_tip_dist < 50 and
            pinky_tip_dist < 50)


def detect_swipe(lmList, prev_lmList):
    """Detect swipe gestures (left or right)."""
    if len(lmList) < 21 or len(prev_lmList) < 21:
        return None

    # Calculate horizontal movement of the wrist
    wrist_x = lmList[wrist][0]
    prev_wrist_x = prev_lmList[wrist][0]

    if wrist_x - prev_wrist_x > 50:  # Swipe right
        return "right"
    elif wrist_x - prev_wrist_x < -50:  # Swipe left
        return "left"
    return None

vol = 0
volBar = 400
volPercentage = 0
pTime = 0  # Previous time for FPS calculation

prev_lmList = {"left": [], "right": []}
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Process audio data
    audio_data = stream.read(1024, exception_on_overflow=False)
    audio_data = struct.unpack(str(1024) + 'h', audio_data)
    audio_level = np.abs(np.array(audio_data)).mean() / 5000  # Normalize

    # Draw advanced visualizations
    draw_spectrum_analyzer(img, compute_spectrum(audio_data))
    #draw_3d_waves(img, audio_level)
    draw_circular_visualizer(img, audio_level)
    draw_particles(img, audio_level)
    #draw_histogram(img, audio_level)
    draw_water_ripple(img, audio_level)

    if results.multi_hand_landmarks:
        for hand_idx, handLms in enumerate(results.multi_hand_landmarks):
            # Determine if it's the left or right hand
            handedness = results.multi_handedness[hand_idx].classification[0].label  # "Left" or "Right"
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

            # Draw hand landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Gesture detection for playback control (right hand)
            if handedness == "Right":
                if is_fist(lmList):
                    if not gesture_detected["right"] and (time.time() - last_gesture_time["right"]) > gesture_cooldown:
                        pyautogui.press('playpause')  # Toggle play/pause
                        gesture_detected["right"] = True
                        last_gesture_time["right"] = time.time()
                        cv2.putText(img, "Play/Pause", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                else:
                    gesture_detected["right"] = False

                # Swipe detection for next/previous track
                if prev_lmList["right"]:
                    swipe = detect_swipe(lmList, prev_lmList["right"])
                    if swipe and (time.time() - last_gesture_time["right"]) > gesture_cooldown:
                        if swipe == "right":
                            pyautogui.press('nexttrack')  # Next track
                            cv2.putText(img, "Next Track", (50, 150),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        elif swipe == "left":
                            pyautogui.press('prevtrack')  # Previous track
                            cv2.putText(img, "Previous Track", (50, 150),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        last_gesture_time["right"] = time.time()

                prev_lmList["right"] = lmList

            # Gesture detection for volume control (left hand)
            elif handedness == "Left":
                # Volume Control Logic
                if len(lmList) > index_tip:
                    index_x, index_y = lmList[index_tip]
                    thumb_x, thumb_y = lmList[thumb_tip]

                    # Measure distance between thumb & index finger
                    distance = np.hypot(index_x - thumb_x, index_y - thumb_y)

                    # Convert distance to volume range
                    vol = np.interp(distance, [30, 200], [minVol, maxVol])
                    volBar = np.interp(distance, [30, 200], [400, 150])
                    volPercentage = np.interp(distance, [30, 200], [0, 100])

                    # Set volume
                    volume.SetMasterVolumeLevel(vol, None)

                    # Color-Changing Line
                    if distance < 60:
                        color = (0, 0, 255)  # Red (Low Volume)
                    elif distance < 130:
                        color = (0, 255, 255)  # Yellow (Medium Volume)
                    else:
                        color = (0, 255, 0)  # Green (High Volume)

                    cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), color, 4)
                    cv2.circle(img, (thumb_x, thumb_y), 7, color, -1)
                    cv2.circle(img, (index_x, index_y), 7, color, -1)

                    # Draw volume control bar
                    cv2.rectangle(img, (50, 150), (85, 400), (200, 0, 0), 3)  # Volume bar outline
                    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 200, 0), cv2.FILLED)  # Volume level
                    cv2.putText(img, f'Vol: {int(volPercentage)}%', (40, 430),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                    # Display gesture label
                    if distance < 50:
                        cv2.putText(img, "Mute", (index_x - 20, index_y - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                pass

    # FPS Calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display FPS counter
    cv2.putText(img, f'FPS: {int(fps)}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    # 🎵 Real-Time Audio Visualization
    audio_data = stream.read(1024, exception_on_overflow=False)
    audio_data = struct.unpack(str(1024) + 'h', audio_data)
    audio_level = np.abs(np.array(audio_data)).mean() / 5000  # Normalize

    # 🔥 Dynamic Wave Animation
    wave_height = int(audio_level * 150)  # Scale wave height
    wave_color = (0, int(volPercentage * 2.55), 255 - int(volPercentage * 2.55))  # Color shifts

    for i in range(100):  # Generate 100 points for the wave
        x = 200 + i * 4
        y = 250 + int(math.sin(i / 5 + time.time() * 3) * wave_height)
        cv2.circle(img, (x, y), 3, wave_color, -1)

    # 🎛️ Equalizer Bars
    for i in range(10):
        bar_x = 400 + i * 15
        bar_height = int(audio_level * np.random.randint(50, 200))
        cv2.rectangle(img, (bar_x, 450 - bar_height), (bar_x + 10, 450),
                      (int(255 * (i / 10)), 100, 255 - int(255 * (i / 10))), -1)

    # 🌟 Glow Effect (Soft Blurred Overlay)
    overlay = img.copy()
    cv2.GaussianBlur(overlay, (15, 15), 10, dst=overlay)
    img = cv2.addWeighted(img, 0.9, overlay, 0.1, 0)

    # Show window
    cv2.imshow("Gesture-Based Volume Control + Audio Visualizer", img)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
stream.stop_stream()
stream.close()
p.terminate()
cv2.destroyAllWindows()
