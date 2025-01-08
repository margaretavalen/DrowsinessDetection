import streamlit as st
import cv2
import numpy as np
import pygame
import os
import mediapipe as mp
import time
import threading
from tensorflow.keras.models import load_model
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class Config:
    MODEL_PATH: str = "dataset/DatasetFinal/model/fcnn_model.h5"
    EYE_CLOSED_THRESHOLD: float = 0.2
    CLOSED_FRAMES_THRESHOLD: int = 30
    ALARM_DURATION: int = 3
    AUDIO_FILE: str = 'alarm.wav'

    LEFT_EYE_INDICES: List[int] = (362, 385, 387, 263, 373, 380)
    RIGHT_EYE_INDICES: List[int] = (33, 160, 158, 133, 153, 144)

class FaceMeshDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )

    def calculate_ear(self, eye_landmarks: List[int], landmarks: List[Tuple[int, int]]) -> float:
        def distance(p1, p2):
            return np.linalg.norm(np.array(p1) - np.array(p2))

        coords_points = [landmarks[idx] for idx in eye_landmarks]
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        # Compute the eye aspect ratio
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)
        return ear

    def process_frame(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        h, w = image.shape[:2]
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return image, None

        landmarks = [(int(lm.x * w), int(lm.y * h)) 
                     for lm in results.multi_face_landmarks[0].landmark]

        left_ear = self.calculate_ear(Config.LEFT_EYE_INDICES, landmarks)
        right_ear = self.calculate_ear(Config.RIGHT_EYE_INDICES, landmarks)
        avg_ear = (left_ear + right_ear) / 2.0

        status = "Closed" if avg_ear < Config.EYE_CLOSED_THRESHOLD else "Open"
        color = (0, 0, 255) if status == "Closed" else (0, 255, 0)
        cv2.putText(image, f"Eye Status: {status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return image, avg_ear

class AlarmSystem:
    def __init__(self):
        pygame.mixer.init()
        self.is_playing = False
        self.start_time = None

    def play(self):
        if not os.path.exists(Config.AUDIO_FILE):
            st.error("Sound file not found!")
            return

        if not self.is_playing:
            pygame.mixer.music.load(Config.AUDIO_FILE)
            pygame.mixer.music.play(-1)  # Loop alarm sound
            self.is_playing = True
            self.start_time = time.time()

    def stop(self):
        if self.is_playing:
            pygame.mixer.music.stop()
            self.is_playing = False

    def update(self):
        if self.is_playing and time.time() - self.start_time > Config.ALARM_DURATION:
            self.stop()

class DrowsinessDetectionPage:
    def __init__(self):
        self.model = load_model(Config.MODEL_PATH)
        self.face_detector = FaceMeshDetector()
        self.alarm = AlarmSystem()
        self.closed_frames = 0

    def process_webcam(self):
        FRAME_WINDOW = st.image([])
        run = st.checkbox("Start Detection")

        if not run:
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to open webcam.")
            return

        try:
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video.")
                    break

                processed_frame, avg_ear = self.face_detector.process_frame(frame)

                if avg_ear and avg_ear < Config.EYE_CLOSED_THRESHOLD:
                    self.closed_frames += 1
                    if self.closed_frames > Config.CLOSED_FRAMES_THRESHOLD and not self.alarm.is_playing:
                        self.alarm.play()
                else:
                    self.closed_frames = 0

                self.alarm.update()
                FRAME_WINDOW.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

        finally:
            cap.release()
            FRAME_WINDOW.image([])

    def render(self):
        # Streamlit UI
        st.title("Drowsiness Detection System")
        st.write("Real-time video feed to detect drowsiness.")
        self.process_webcam()

# Export class for modular use
def drowsiness_detection_page():
    page = DrowsinessDetectionPage()
    page.render()
