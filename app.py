import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import math
import time
import threading

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ========================
# 1. 상수 및 설정
# ========================

APP_STATE_SETUP = "SETUP"
APP_STATE_MONITORING = "MONITORING"
APP_STATE_PAUSED = "PAUSED"
APP_STATE_ENDED = "ENDED"

EYE_CLOSED_THRESHOLD = 0.21
NO_FACE_THRESHOLD = 5.0

APP_STATE_IDLE = "IDLE"
APP_STATE_MONITORING = "MONITORING"

def init_session_state():
    """졸음 감지에 필요한 상태 값들을 한 번에 초기화"""
    defaults = {
        "app_state": APP_STATE_IDLE,       # 처음에는 대기 상태
        "last_update_time": 0.0,
        "eye_closed_time_sec": 2.0,        # 눈 감은 시간 임계값(초)
        "eyes_closed_time": 0.0,
        "no_face_time": 0.0,
        "current_ear": 0.0,
        "drowsiness_state": "INIT",
        "alarm_active": False,
        "focused_time": 0.0,
        "drowsy_time": 0.0,
        "away_time": 0.0,
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

SENSITIVITY_MAP = {
    1: {"label": "😴 1단계 (피곤해요)", "time": 5.0},
    2: {"label": "😐 2단계 (보통이에요)", "time": 3.0},
    3: {"label": "😤 3단계 (집중할래요)", "time": 2.0},
    4: {"label": "🔥 4단계 (스파르타)", "time": 1.0},
}

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

BASE64_SOUND = (
    "data:audio/wav;base64,"
    "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQQAAAAAAABeLwE6+QnO9h7k+P/4/PPw5O3s8uTk3tvPz4+40Qy3XQ4l+QAA"
)

def play_alarm_js(volume: float = 0.5):
    volume = max(0.0, min(1.0, volume))
    audio_html = f"""
    <audio id="alarm-sound" autoplay>
        <source src="{BASE64_SOUND}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    <script>
        var audio = document.getElementById('alarm-sound');
        if (audio) {{
            audio.volume = {volume};
            audio.play().catch(e => console.error("Audio playback blocked:", e));
            setTimeout(() => {{
                var element = document.getElementById('alarm-sound');
                if(element) element.remove();
            }}, 2000);
        }}
    </script>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# ========================
# 2. 유틸 함수
# ========================

def format_time(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"

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

# ========================
# 3. session_state 초기화
# ========================

if "app_state" not in st.session_state:
    st.session_state.app_state = APP_STATE_SETUP

    st.session_state.study_goal = ""
    st.session_state.sensitivity_level = 2
    st.session_state.eye_closed_time_sec = SENSITIVITY_MAP[2]["time"]

    st.session_state.session_start = 0.0
    st.session_state.focused_time = 0.0
    st.session_state.drowsy_time = 0.0
    st.session_state.away_time = 0.0
    st.session_state.last_update_time = 0.0

    st.session_state.drowsiness_state = "INIT"
    st.session_state.eyes_closed_time = 0.0
    st.session_state.no_face_time = 0.0
    st.session_state.current_ear = 0.0
    st.session_state.alarm_active = False
    st.session_state.alarm_played_in_cycle = False

# ========================
# 4. VideoProcessor (MediaPipe + EAR)
# ========================

class FaceMeshDrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            img_h, img_w, _ = img.shape
            now = time.time()

            with self.lock:
                state = st.session_state.get("app_state", APP_STATE_IDLE)
                last_update_time = st.session_state.get("last_update_time", 0.0)
                eye_closed_time_sec = st.session_state.get("eye_closed_time_sec", 2.0)
                eyes_closed_time = st.session_state.get("eyes_closed_time", 0.0)
                no_face_time = st.session_state.get("no_face_time", 0.0)

            dt = 0.0
            if last_update_time != 0.0:
                dt = now - last_update_time

            current_ear = st.session_state.current_ear
            drowsiness_state = st.session_state.drowsiness_state
            alarm_active = False

            if state == APP_STATE_MONITORING:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0].landmark

                    left_ear, _ = calc_EAR(face_landmarks, LEFT_EYE_IDX, img_w, img_h)
                    right_ear, _ = calc_EAR(face_landmarks, RIGHT_EYE_IDX, img_w, img_h)
                    current_ear = (left_ear + right_ear) / 2.0

                    if current_ear > EYE_CLOSED_THRESHOLD:
                        eyes_closed_time = 0.0
                        drowsiness_state = "FOCUS (집중)"
                        st.session_state.focused_time += dt
                    else:
                        eyes_closed_time += dt
                        if eyes_closed_time >= eye_closed_time_sec:
                            drowsiness_state = "DROWSY (졸음 감지!)"
                            st.session_state.drowsy_time += dt
                            alarm_active = True
                            cv2.rectangle(img, (0, 0), (img_w, img_h), (0, 0, 255), 10)
                        else:
                            drowsiness_state = "BLINK / WARNING (경고)"

                    no_face_time = 0.0
                else:
                    eyes_closed_time = 0.0
                    no_face_time += dt
                    if no_face_time >= NO_FACE_THRESHOLD:
                        drowsiness_state = "AWAY (자리 비움)"
                        st.session_state.away_time += dt
                        cv2.rectangle(img, (0, 0), (img_w, img_h), (255, 100, 100), 10)
                    else:
                        drowsiness_state = "LOST (얼굴 인식 실패)"

                with self.lock:
                    st.session_state.last_update_time = now
                    st.session_state.current_ear = current_ear
                    st.session_state.eyes_closed_time = eyes_closed_time
                    st.session_state.no_face_time = no_face_time
                    st.session_state.drowsiness_state = drowsiness_state
                    st.session_state.alarm_active = alarm_active

            cv2.putText(
                img,
                f"State: {st.session_state.drowsiness_state}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            if st.session_state.alarm_active:
                cv2.putText(
                    img,
                    "🚨 ALARM! (소리 활성화)",
                    (img_w // 2 - 250, img_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                )

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        except Exception as e:
            print("ERROR in recv:", repr(e))
            # 에러 나도 스트림이 완전 죽지 않게 원본 프레임 반환
            return frame
    
    

# ========================
# 5. UI 함수들
# ========================

def render_setup_ui():
    st.title("StudyCam: 대화형 학습 모니터링")
    st.write("---")

    st.markdown("## 오늘의 공부 목표는 무엇인가요? 🎯")
    goal = st.text_input("목표 입력", value="파이썬 프로젝트 완성하기", label_visibility="collapsed")

    st.markdown("## 현재 컨디션에 맞는 집중 모드(민감도)를 선택하세요 ⭐")

    sensitivity_options = [SENSITIVITY_MAP[k]["label"] for k in sorted(SENSITIVITY_MAP.keys())]
    selected_label = st.radio(
        "감지 강도 선택",
        options=sensitivity_options,
        index=1,
        help="눈 감음 허용 시간이 짧을수록 엄격한 모드입니다.",
    )

    selected_level = next(k for k, v in SENSITIVITY_MAP.items() if v["label"] == selected_label)

    st.markdown("---")

    if st.button("🚀 공부 시작하기", use_container_width=True, type="primary", key="start_study_btn"):
        st.session_state.study_goal = goal
        st.session_state.sensitivity_level = selected_level
        st.session_state.eye_closed_time_sec = SENSITIVITY_MAP[selected_level]["time"]
        st.session_state.app_state = APP_STATE_MONITORING
        st.session_state.session_start = time.time()
        st.session_state.last_update_time = time.time()
        st.rerun()

def handle_pause_resume():
    if st.session_state.app_state == APP_STATE_MONITORING:
        st.session_state.app_state = APP_STATE_PAUSED
        st.session_state.last_update_time = time.time()
    elif st.session_state.app_state == APP_STATE_PAUSED:
        st.session_state.app_state = APP_STATE_MONITORING
        pause_duration = time.time() - st.session_state.last_update_time
        st.session_state.session_start += pause_duration
        st.session_state.last_update_time = time.time()
    st.rerun()

def handle_end_session():
    if st.session_state.app_state != APP_STATE_ENDED:
        if st.session_state.app_state == APP_STATE_MONITORING:
            st.session_state.last_update_time = time.time()
        st.session_state.app_state = APP_STATE_ENDED
        st.rerun()

def render_monitoring_ui():
    # 알람 재생 한 번만
    if st.session_state.alarm_active and not st.session_state.alarm_played_in_cycle:
        extra = max(0.0, st.session_state.eyes_closed_time - st.session_state.eye_closed_time_sec)
        ratio = min(1.0, extra / 5.0)
        volume = 0.3 + (1.0 - 0.3) * ratio
        play_alarm_js(volume=volume)
        st.session_state.alarm_played_in_cycle = True
    elif not st.session_state.alarm_active:
        st.session_state.alarm_played_in_cycle = False

    st.title("StudyCam: 집중 모니터링 중")
    st.markdown(f"## 🎯 목표: **{st.session_state.study_goal}**")

    col_status, col_ear, col_sensitivity = st.columns([2, 1, 1])

    if st.session_state.session_start > 0:
        elapsed_time = time.time() - st.session_state.session_start
    else:
        elapsed_time = 0.0

    status_text = st.session_state.drowsiness_state
    if st.session_state.app_state == APP_STATE_PAUSED:
        status_text = "일시 정지됨 (PAUSED)"

    col_status.metric(
        "⏳ 총 학습 시간",
        format_time(elapsed_time),
        delta=status_text,
        delta_color="off" if "FOCUS" in st.session_state.drowsiness_state else "inverse",
    )
    col_ear.metric("EAR", f"{st.session_state.current_ear:.3f}")
    col_sensitivity.metric(
        "민감도",
        f"{st.session_state.sensitivity_level}단계",
        f"{st.session_state.eye_closed_time_sec:.1f}초 허용",
    )

    st.markdown("---")

    st.subheader("웹캠 모니터링 (졸음/자리 비움 감지)")
    webrtc_streamer(
        key="drowsiness_monitor",
        video_processor_factory=FaceMeshDrowsinessProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 30},
            },
            "audio": False,
        },
        async_processing=True,
    )

    st.markdown("### 실시간 감지 상태 및 누적 시간")
    st.text(f"  집중 시간: {format_time(st.session_state.focused_time)}")
    st.text(f"  졸음 시간: {format_time(st.session_state.drowsy_time)}")
    st.text(f"  자리 비움 시간: {format_time(st.session_state.away_time)}")

    st.markdown("---")
    col_pause, col_end = st.columns(2)

    if st.session_state.app_state == APP_STATE_MONITORING:
        if col_pause.button("⏸️ 잠시 멈춤", use_container_width=True, type="secondary", key="pause_btn_active"):
            handle_pause_resume()
    else:
        if col_pause.button("▶️ 다시 시작", use_container_width=True, type="primary", key="resume_btn_active"):
            handle_pause_resume()

    if col_end.button("🛑 공부 종료", use_container_width=True, type="primary", key="end_session_btn"):
        handle_end_session()

def render_ended_ui():
    st.title("StudyCam: 학습 결과 요약 📝")
    st.markdown("---")

    total_session_time = st.session_state.last_update_time - st.session_state.session_start
    total_active_time = (
        st.session_state.focused_time
        + st.session_state.drowsy_time
        + st.session_state.away_time
    )
    focus_ratio = (
        (st.session_state.focused_time / total_active_time) * 100 if total_active_time > 0 else 0
    )

    st.markdown(f"## 🎯 목표: {st.session_state.study_goal}")
    st.markdown(
        f"**⭐ 민감도 설정:** {st.session_state.sensitivity_level}단계 "
        f"({st.session_state.eye_closed_time_sec:.1f}초 허용)"
    )
    st.markdown("---")

    st.metric("총 세션 시간", format_time(total_session_time))
    st.metric("✅ 최종 집중도", f"{focus_ratio:.1f}%")

    st.markdown("### 상세 시간 분석")

    col_f, col_d, col_a = st.columns(3)
    col_f.metric("집중 시간", format_time(st.session_state.focused_time))
    col_d.metric("졸음 시간", format_time(st.session_state.drowsy_time))
    col_a.metric("자리 비움 시간", format_time(st.session_state.away_time))

    st.markdown("---")

    if st.button("처음으로 돌아가기", use_container_width=True, key="reset_btn"):
        st.session_state.app_state = APP_STATE_SETUP
        st.session_state.session_start = 0.0
        st.session_state.focused_time = 0.0
        st.session_state.drowsy_time = 0.0
        st.session_state.away_time = 0.0
        st.session_state.last_update_time = 0.0
        st.rerun()

# ========================
# 6. 메인
# ========================

def main():
    
    init_session_state()   # 🔴 이 줄 추가!

    st.title("졸음 감지 AI 모니터링")
    # ... 버튼/슬라이더 등 UI 코드 ...
    # webrtc_streamer(...) 호출 ...
    st.set_page_config(
        page_title="StudyCam Drowsiness Monitor (팀 9조)",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if st.session_state.app_state == APP_STATE_SETUP:
        render_setup_ui()
    elif st.session_state.app_state in (APP_STATE_MONITORING, APP_STATE_PAUSED):
        render_monitoring_ui()
    elif st.session_state.app_state == APP_STATE_ENDED:
        render_ended_ui()

if __name__ == "__main__":
    main()
