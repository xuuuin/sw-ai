import streamlit as st
import cv2
import mediapipe as mp
import math
import time
import threading
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ===== 설정 =====
EYE_CLOSED_THRESHOLD = 0.21
EYE_CLOSED_TIME_SEC  = 1.0
NO_FACE_THRESHOLD    = 5.0

LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

mp_face_mesh = mp.solutions.face_mesh

def calc_EAR(landmarks, eye_idx_list, img_w, img_h):
    points = []
    for idx in eye_idx_list:
        lm = landmarks[idx]
        x = int(lm.x * img_w)
        y = int(lm.y * img_h)
        points.append((x, y))

    p1, p2, p3, p4, p5, p6 = points

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    ear = (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4) + 1e-6)
    return ear, points

def format_time(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"

# ========== Processor ==========

class StudyCamProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.session_start = time.time()
        self.prev_time = time.time()

        self.current_ear = 0.0
        self.eyes_closed_time = 0.0
        self.no_face_time = 0.0
        self.focused_time = 0.0

        self.state = "INIT"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now

        img = cv2.flip(img, 1)
        img_h, img_w, _ = img.shape

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        eyes_open = False

        if results.multi_face_landmarks:
            self.no_face_time = 0.0
            face_landmarks = results.multi_face_landmarks[0].landmark

            left_ear, left_pts = calc_EAR(face_landmarks, LEFT_EYE_IDX, img_w, img_h)
            right_ear, right_pts = calc_EAR(face_landmarks, RIGHT_EYE_IDX, img_w, img_h)
            self.current_ear = (left_ear + right_ear) / 2.0

            # Landmark 점 표시
            for (x, y) in left_pts + right_pts:
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

            if self.current_ear > EYE_CLOSED_THRESHOLD:
                eyes_open = True
                self.eyes_closed_time = 0.0
            else:
                eyes_open = False
                self.eyes_closed_time += dt

            # 상태 판정
            if eyes_open:
                self.state = "FOCUS"
                self.focused_time += dt
            else:
                if self.eyes_closed_time >= EYE_CLOSED_TIME_SEC:
                    self.state = "DROWSY"
                    cv2.rectangle(img, (0, 0), (img_w, img_h), (0, 0, 255), 5)
                else:
                    self.state = "BLINK / WARNING"

        else:
            self.current_ear = 0.0
            self.eyes_closed_time = 0.0
            self.no_face_time += dt

            if self.no_face_time >= NO_FACE_THRESHOLD:
                self.state = "AWAY"
            else:
                self.state = "LOST"

        # -----------------------------
        # 화면에 상태/시간 표시
        # -----------------------------
        elapsed = now - self.session_start

        cv2.putText(img, f"State: {self.state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(img, f"Session: {format_time(elapsed)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(img, f"Focused: {format_time(self.focused_time)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.putText(img, f"EAR: {self.current_ear:.3f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.putText(img, f"EyesClosedTime: {self.eyes_closed_time:.1f}s", (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ===== Streamlit UI =====
st.set_page_config(page_title="StudyCam - Streamlit", layout="wide")
st.title("📚 StudyCam (Browser + Streamlit WebRTC)")

st.write("➡️ 브라우저에서 카메라 허용 후, EAR / Drowsy 상태가 바로 영상에 표시돼요.")

webrtc_streamer(
    key="studycam",
    video_processor_factory=StudyCamProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
