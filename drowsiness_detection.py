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
    EYE_CLOSED_THRESHOLD: float = 0.3
    CLOSED_FRAMES_THRESHOLD: int = 60 #60 fps setara 2 detik
    ALARM_DURATION: int = 4
    AUDIO_FILE: str = 'alarm.wav'
    
    LEFT_EYE_INDICES: List[int] = (362, 385, 387, 263, 373, 380)
    RIGHT_EYE_INDICES: List[int] = (33, 160, 158, 133, 153, 144)

class FaceMeshDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    def calculate_ear(self, eye_indices: List[int], landmarks: List[Tuple[int, int]]) -> float:
        # Calculate EAR using the given eye indices
        p2_p6 = np.linalg.norm(np.array(landmarks[eye_indices[1]]) - np.array(landmarks[eye_indices[5]]))
        p3_p5 = np.linalg.norm(np.array(landmarks[eye_indices[2]]) - np.array(landmarks[eye_indices[4]]))
        p1_p4 = np.linalg.norm(np.array(landmarks[eye_indices[0]]) - np.array(landmarks[eye_indices[3]]))
        return (p2_p6 + p3_p5) / (2.0 * p1_p4)

    def process_frame(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        h, w = image.shape[:2]
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return image, None

        landmarks = [(int(lm.x * w), int(lm.y * h)) 
                     for lm in results.multi_face_landmarks[0].landmark]

        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(Config.LEFT_EYE_INDICES, landmarks)
        right_ear = self.calculate_ear(Config.RIGHT_EYE_INDICES, landmarks)
        avg_ear = (left_ear + right_ear) / 2.0

        # Display EAR status
        status = "Mengantuk" if avg_ear < Config.EYE_CLOSED_THRESHOLD else "Tidak Mengantuk"
        color = (0, 0, 255) if status == "Mengantuk" else (0, 255, 0)
        cv2.putText(image, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Display EAR for each eye
        cv2.putText(image, f"EAR (Kiri): {left_ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"EAR (Kanan): {right_ear:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw rectangles and display EAR for each eye region
        for eye_indices, ear_value, label in zip(
            [Config.LEFT_EYE_INDICES, Config.RIGHT_EYE_INDICES],
            [left_ear, right_ear],
            ["Kiri", "Kanan"]
        ):
            eye_coords = np.array([landmarks[idx] for idx in eye_indices], dtype=np.int32)
            x_min = min(eye_coords[:, 0])
            y_min = min(eye_coords[:, 1])
            x_max = max(eye_coords[:, 0])
            y_max = max(eye_coords[:, 1])
            
            # Draw rectangle around the eye
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Display EAR above each eye
            cv2.putText(image, f"{label}: {ear_value:.2f}", (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
        st.markdown(
            """
            ### Fitur Webcam
            - Klik checkbox **Start Detection** untuk memulai deteksi.
            - Pastikan webcam aktif dan memberikan akses ke aplikasi.
            - Aplikasi akan mendeteksi status mata dan memberikan peringatan jika mata tertutup lebih dari 2 detik.
            - Alarm suara akan berbunyi selama 4 detik ketika mata terdeteksi mengantuk lebih dari 2 detik. 
            - Alarm suara akan berhenti berbunyi ketika mata sudah tidak terdeteksi mengantuk. 
            - Hapus checkbox **Start Detection** untuk menghentikan deteksi.
            """
        )

        FRAME_WINDOW = st.image([])
        run = st.checkbox("Start Detection")
        if not run:
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Unable to access the webcam. Try changing the camera index.")
            cap = cv2.VideoCapture(-1)

        try:
            while run:
                ret, frame = cap.read()
                if not ret or frame is None:
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
        st.markdown(
            """
            ### Fitur Upload Gambar
            - Unggah file gambar dengan format `.jpg`, `.jpeg`, atau `.png`.
            - Aplikasi akan memproses gambar untuk mendeteksi status mata.
            - Hasil deteksi akan ditampilkan dengan status "Mengantuk" atau "Tidak Mengantuk".
            """
        )

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
        selected_option = st.sidebar.selectbox("Pilih", menu_options)

        # Sidebar Description
        st.sidebar.markdown(
            """
            ### Petunjuk:
            - **Webcam**: Gunakan kamera untuk deteksi kantuk secara real-time.
            - **Upload Image**: Unggah gambar untuk mendeteksi status mata pada gambar.
            """
        )

        # Streamlit UI
        st.markdown(
            """
            <div style="text-align: center; font-size: 18px;">
                <h1>Drowsiness Detection System</h1>
                <p>Real-time video feed to detect drowsiness using CNN.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

        if selected_option == 'Webcam':
            self.process_webcam()
        elif selected_option == 'Upload Image':
            self.process_image()

# Export class for modular use
def drowsiness_detection_page():
    page = DrowsinessDetectionPage()
    page.render()
