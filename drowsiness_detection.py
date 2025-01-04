import streamlit as st
import cv2
import numpy as np
import pygame
import os
import mediapipe as mp
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class Config:
    MODEL_PATH: str = "dataset/DatasetFinal/model/fcnn_model.h5"
    EYE_CLOSED_THRESHOLD: float = 0.2
    CLOSED_FRAMES_THRESHOLD: int = 30
    ALARM_DURATION: int = 4
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

        for eye_indices in [Config.LEFT_EYE_INDICES, Config.RIGHT_EYE_INDICES]:
            eye_coords = np.array([landmarks[idx] for idx in eye_indices], dtype=np.int32)
            x_min = min(eye_coords[:, 0])
            y_min = min(eye_coords[:, 1])
            x_max = max(eye_coords[:, 0])
            y_max = max(eye_coords[:, 1])

            eyebrow_indices = [17, 18, 19, 20, 21, 22]  
            eyebrow_coords = np.array([landmarks[idx] for idx in eyebrow_indices], dtype=np.int32)
            y_min_eyebrow = min(eyebrow_coords[:, 1])

            y_min = min(y_min, y_min_eyebrow)

            width = x_max - x_min
            height = y_max - y_min
            aspect_ratio = width / height if height != 0 else 0

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

            cv2.putText(image, f"Ratio: {aspect_ratio:.2f}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
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

        try:
            pygame.mixer.music.load(Config.AUDIO_FILE)
            pygame.mixer.music.play()
            self.is_playing = True
            self.start_time = time.time()
        except Exception as e:
            st.error(f"Error playing sound: {e}")

    def update(self):
        if self.is_playing and time.time() - self.start_time > Config.ALARM_DURATION:
            pygame.mixer.music.stop()
            self.is_playing = False
            self.start_time = None

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

    def process_image(self):
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_image is None:
            return

        image = cv2.imdecode(
            np.frombuffer(uploaded_image.read(), np.uint8),
            cv2.IMREAD_COLOR
        )
        processed_image, _ = self.face_detector.process_frame(image)
        st.image(processed_image, channels="BGR", caption="Processed Image", 
                 use_column_width=True)

    def render(self):
        # Streamlit Sidebar Menu
        menu_options = ['Webcam', 'Upload Image']
        selected_option = st.sidebar.selectbox("Choose Input Type", menu_options)

        # Streamlit UI
        st.markdown("""<div style="text-align: center; color: #fffff; font-size: 40px; font-weight: bold;">
            <h1>Drowsiness Detection System</h1>
            <p>Real-time video feed to detect drowsiness using CNN.</p>
        </div>""", unsafe_allow_html=True)

        if selected_option == 'Webcam':
            self.process_webcam()
        elif selected_option == 'Upload Image':
            self.process_image()

# Export class for modular use
def drowsiness_detection_page():
    page = DrowsinessDetectionPage()
    page.render()
