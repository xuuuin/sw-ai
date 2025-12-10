import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import math
import time
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer # <-- 수정: webrtc_stream -> webrtc_streamer
import threading
from streamlit.components.v1 import html

# ========================
# 1. 상수 및 설정
# ========================

# StudyCam 앱 상태 정의
APP_STATE_SETUP = "SETUP"
APP_STATE_MONITORING = "MONITORING"
APP_STATE_PAUSED = "PAUSED"
APP_STATE_ENDED = "ENDED"

# 졸음 감지 기준
EYE_CLOSED_THRESHOLD = 0.21 
NO_FACE_THRESHOLD = 5.0     

# 민감도 단계 설정
SENSITIVITY_MAP = {
    1: {"label": "😴 1단계 (피곤해요)", "time": 5.0},
    2: {"label": "😐 2단계 (보통이에요)", "time": 3.0},
    3: {"label": "😤 3단계 (집중할래요)", "time": 2.0},
    4: {"label": "🔥 4단계 (스파르타)", "time": 1.0}
}

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 눈 랜드마크 인덱스 (FaceMesh 기준)
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# --- 오디오 설정 ---
# Base64 인코딩된 더미 WAV 파일 (경고음)
# 실제 알람 소리("사당로.wav")를 사용하려면 해당 파일을 Base64로 인코딩하여 여기에 넣어주세요.
BASE64_SOUND = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQQAAAAAAABeLwE6+QnO9h7k+P/4/PPw5O3s8uTk3tvPz4+40Qy3XQ4l+QAA"


def play_alarm_js():
    """HTML 오디오 태그를 삽입하여 경고음을 재생합니다."""
    # JavaScript를 사용하여 오디오를 재생하고 즉시 HTML을 제거 (깨끗하게)
    audio_html = f"""
    <audio id="alarm-sound" autoplay>
        <source src="{BASE64_SOUND}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    <script>
        var audio = document.getElementById('alarm-sound');
        if (audio) {{
            audio.volume = 0.5; // 볼륨 설정
            audio.play().catch(e => console.error("Audio playback blocked:", e));
            // 재생 후 요소 제거 (Streamlit 재실행 시마다 다시 삽입)
            setTimeout(() => {{ 
                var element = document.getElementById('alarm-sound'); 
                if(element) element.remove(); 
            }}, 2000); 
        }}
    </script>
    """
    # Streamlit에 HTML 코드 삽입
    st.markdown(audio_html, unsafe_allow_html=True)

# ========================
# 2. 유틸리티 함수
# ========================

def format_time(sec):
    """시간을 'MM:SS' 형식으로 포맷"""
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"

def calc_EAR(landmarks, eye_idx_list, img_w, img_h):
    """눈 랜드마크 좌표로 EAR 계산"""
    points = []
    for idx in eye_idx_list:
        lm = landmarks[idx]
        x = int(lm.x * img_w)
        y = int(lm.y * img_h)
        points.append((x, y))
    
    # p1, p2, p3, p4, p5, p6
    p1, p2, p3, p4, p5, p6 = points

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # 수직 거리와 수평 거리 비율 계산
    ear = (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4) + 1e-6)
    return ear, points

# ========================
# 3. Streamlit Session State 초기화
# ========================

if 'app_state' not in st.session_state:
    # 앱 상태
    st.session_state.app_state = APP_STATE_SETUP
    
    # 설정 값
    st.session_state.study_goal = ""
    st.session_state.sensitivity_level = 2 # 기본 2단계
    st.session_state.eye_closed_time_sec = SENSITIVITY_MAP[2]["time"]
    
    # 시간 및 상태 누적 변수
    st.session_state.session_start = 0.0
    st.session_state.focused_time = 0.0
    st.session_state.drowsy_time = 0.0
    st.session_state.away_time = 0.0
    st.session_state.last_update_time = 0.0 # dt 계산용
    
    # 실시간 모니터링 값
    st.session_state.drowsiness_state = "INIT"
    st.session_state.eyes_closed_time = 0.0
    st.session_state.no_face_time = 0.0
    st.session_state.current_ear = 0.0
    st.session_state.alarm_active = False
    # 알람 재생을 한 번만 하기 위한 플래그
    st.session_state.alarm_played_in_cycle = False 

# ========================
# 4. VideoTransformer (MediaPipe 처리)
# ========================

class FaceMeshDrowsinessTransformer(VideoTransformerBase):
    """웹캠 프레임을 처리하고 졸음 상태를 감지하는 클래스"""
    
    def __init__(self):
        # MediaPipe FaceMesh는 인스턴스마다 하나씩 생성
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # 락 (Lock) 객체를 사용하여 session_state 접근 시 동시성 문제 방지
        self.lock = threading.Lock()

    def transform(self, frame: np.ndarray) -> np.ndarray:
        # Streamlit session state 복사 (프레임 단위 상태 업데이트)
        with self.lock:
            state = st.session_state.app_state
            last_update_time = st.session_state.last_update_time
            eye_closed_time_sec = st.session_state.eye_closed_time_sec
            eyes_closed_time = st.session_state.eyes_closed_time
            no_face_time = st.session_state.no_face_time
            
        img = frame.copy()
        img_h, img_w, _ = img.shape
        now = time.time()
        
        # dt 계산
        dt = 0.0
        if last_update_time != 0.0:
            dt = now - last_update_time

        # 상태 업데이트 및 시간 누적은 MONITORING 상태에서만 진행
        if state == APP_STATE_MONITORING:
            
            # MediaPipe 처리
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            current_ear = 0.0
            drowsiness_state = "LOST (얼굴 인식 실패)"
            alarm_active = False
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark

                # EAR 계산
                left_ear, _ = calc_EAR(face_landmarks, LEFT_EYE_IDX, img_w, img_h)
                right_ear, _ = calc_EAR(face_landmarks, RIGHT_EYE_IDX, img_w, img_h)
                current_ear = (left_ear + right_ear) / 2.0
                
                # 졸음 감지 로직
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
                        
                        # 빨간색 테두리 표시 (시각적 알람)
                        cv2.rectangle(img, (0, 0), (img_w, img_h), (0, 0, 255), 10) 
                    else:
                        drowsiness_state = "BLINK / WARNING (경고)"

                no_face_time = 0.0
                
            else:
                # 얼굴 안 보임
                eyes_closed_time = 0.0
                no_face_time += dt
                
                if no_face_time >= NO_FACE_THRESHOLD:
                    drowsiness_state = "AWAY (자리 비움)"
                    st.session_state.away_time += dt
                    # 파란색 테두리 표시
                    cv2.rectangle(img, (0, 0), (img_w, img_h), (255, 100, 100), 10)
                else:
                    drowsiness_state = "LOST (얼굴 인식 실패)"
            
            # --- 상태 업데이트 (Lock 사용) ---
            with self.lock:
                st.session_state.last_update_time = now
                st.session_state.current_ear = current_ear
                st.session_state.eyes_closed_time = eyes_closed_time
                st.session_state.no_face_time = no_face_time
                st.session_state.drowsiness_state = drowsiness_state
                st.session_state.alarm_active = alarm_active
        
        # 텍스트 오버레이
        cv2.putText(img, f"State: {st.session_state.drowsiness_state}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if st.session_state.alarm_active:
             cv2.putText(img, "🚨 ALARM! (소리 활성화)", (img_w // 2 - 250, img_h // 2), 
                         cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # BGR 프레임 반환
        return img

# ========================
# 5. Streamlit UI 렌더링 함수
# ========================

def render_setup_ui():
    """초기 설정 화면"""
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
        help="눈 감음 허용 시간이 짧을수록 엄격한 모드입니다."
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
    """일시정지/재개 상태 변경 및 시간 조정"""
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
    """세션을 종료하고 통계 화면으로 전환"""
    if st.session_state.app_state != APP_STATE_ENDED:
        if st.session_state.app_state == APP_STATE_MONITORING:
            st.session_state.last_update_time = time.time()
        
        st.session_state.app_state = APP_STATE_ENDED
        st.rerun()

def render_monitoring_ui():
    """모니터링 및 학습 진행 화면"""
    
    # 1. 알람 재생 로직
    # 졸음 감지 시, 한 번만 알람 소리 재생을 시도합니다.
    if st.session_state.alarm_active and not st.session_state.alarm_played_in_cycle:
        play_alarm_js()
        st.session_state.alarm_played_in_cycle = True
    elif not st.session_state.alarm_active:
        st.session_state.alarm_played_in_cycle = False


    st.title("StudyCam: 집중 모니터링 중")
    st.markdown(f"## 🎯 목표: **{st.session_state.study_goal}**")

    # 상단 정보 요약
    col_status, col_ear, col_sensitivity = st.columns([2, 1, 1])
    
    elapsed_time = time.time() - st.session_state.session_start if st.session_state.app_state != APP_STATE_PAUSED else st.session_state.last_update_time - st.session_state.session_start
    
    status_text = st.session_state.drowsiness_state
    if st.session_state.app_state == APP_STATE_PAUSED:
        status_text = "일시 정지됨 (PAUSED)"

    col_status.metric(
        "⏳ 총 학습 시간", 
        format_time(elapsed_time), 
        delta=status_text, 
        delta_color="off" if "FOCUS" in st.session_state.drowsiness_state else "inverse"
    )
    col_ear.metric("EAR", f"{st.session_state.current_ear:.3f}")
    col_sensitivity.metric(
        "민감도", 
        f"{st.session_state.sensitivity_level}단계", 
        f"{st.session_state.eye_closed_time_sec:.1f}초 허용"
    )
    
    st.markdown("---")

    # 웹캠 스트림 (PIP 역할)
    st.subheader("웹캠 모니터링 (졸음/자리 비움 감지)")
    webrtc_streamer( # <-- 수정: webrtc_stream -> webrtc_streamer
        key="drowsiness_monitor",
        video_processor_factory=FaceMeshDrowsinessTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 30}
            },
            "audio": False
        },
        async_transform=True,
    )

    # 실시간 상태 및 누적 시간
    st.markdown("### 실시간 감지 상태 및 누적 시간")
    st.text(f"  집중 시간: {format_time(st.session_state.focused_time)}")
    st.text(f"  졸음 시간: {format_time(st.session_state.drowsy_time)}")
    st.text(f"  자리 비움 시간: {format_time(st.session_state.away_time)}")

    # 버튼
    st.markdown("---")
    col_pause, col_end = st.columns(2)
    
    # 일시정지 / 다시 시작 버튼
    if st.session_state.app_state == APP_STATE_MONITORING:
        pause_label = "⏸️ 잠시 멈춤"
        pause_type = "secondary"
        if col_pause.button(pause_label, use_container_width=True, type=pause_type, key="pause_btn_active"):
            handle_pause_resume()
    else: # APP_STATE_PAUSED
        pause_label = "▶️ 다시 시작"
        pause_type = "primary"
        if col_pause.button(pause_label, use_container_width=True, type=pause_type, key="resume_btn_active"):
            handle_pause_resume()
            
    # 공부 종료 버튼
    if col_end.button("🛑 공부 종료", use_container_width=True, type="primary", key="end_session_btn"):
        handle_end_session()

def render_ended_ui():
    """최종 학습 통계 화면"""
    st.title("StudyCam: 학습 결과 요약 📝")
    st.markdown("---")
    
    # 총 학습 시간 = 세션 종료 시점 - 세션 시작 시점 (일시정지 시간 포함)
    total_session_time = st.session_state.last_update_time - st.session_state.session_start
    # 총 활동 시간 = 집중 + 졸음 + 자리 비움
    total_active_time = st.session_state.focused_time + st.session_state.drowsy_time + st.session_state.away_time
    
    # 집중도 계산
    focus_ratio = (st.session_state.focused_time / total_active_time) * 100 if total_active_time > 0 else 0
    
    st.markdown(f"## 🎯 목표: {st.session_state.study_goal}")
    st.markdown(f"**⭐ 민감도 설정:** {st.session_state.sensitivity_level}단계 ({st.session_state.eye_closed_time_sec:.1f}초 허용)")
    st.markdown("---")

    # 통계 메트릭
    st.metric("총 세션 시간", format_time(total_session_time))
    st.metric("✅ 최종 집중도", f"{focus_ratio:.1f}%")

    st.markdown("### 상세 시간 분석")
    
    col_f, col_d, col_a = st.columns(3)
    col_f.metric("집중 시간", format_time(st.session_state.focused_time))
    col_d.metric("졸음 시간", format_time(st.session_state.drowsy_time))
    col_a.metric("자리 비움 시간", format_time(st.session_state.away_time))

    st.markdown("---")
    
    if st.button("처음으로 돌아가기", use_container_width=True, key="reset_btn"):
        # 상태 초기화
        st.session_state.app_state = APP_STATE_SETUP
        st.session_state.session_start = 0.0
        st.session_state.focused_time = 0.0
        st.session_state.drowsy_time = 0.0
        st.session_state.away_time = 0.0
        st.session_state.last_update_time = 0.0
        st.rerun()

# ========================
# 6. 메인 앱 실행
# ========================

def main():
    """메인 실행 함수"""
    st.set_page_config(
        page_title="StudyCam Drowsiness Monitor (팀 9조)",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    # st.sidebar.markdown("© 2025 sw_ai 9조 (Team 9)")
    
    if st.session_state.app_state == APP_STATE_SETUP:
        render_setup_ui()
    elif st.session_state.app_state == APP_STATE_MONITORING or st.session_state.app_state == APP_STATE_PAUSED:
        render_monitoring_ui()
    elif st.session_state.app_state == APP_STATE_ENDED:
        render_ended_ui()

if __name__ == "__main__":
    main()