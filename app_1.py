import streamlit as st
import cv2
import time
import pygame
import numpy as np
import mediapipe as mp
import math
import base64
import os
from datetime import datetime, date # [추가] 날짜 계산을 위한 모듈

# Base64 파일 경로 설정
ALARM_FILE_PATH = "alarm_b64.txt" 
ALARM_WAV_FILENAME = "alarm.wav"

# ========================
# 1. Pygame 초기화 & 알람
# ========================
def decode_alarm_sound(file_path, output_filename):
    """Base64 파일에서 데이터를 읽어와 WAV 파일로 디코딩"""
    try:
        with open(file_path, "r") as f:
            b64_data = f.read().strip()
        
        # 파일이 비어있는 경우 처리
        if not b64_data:
            st.error(f"'{file_path}' 파일이 비어 있습니다. Base64 문자열을 확인해주세요.")
            return False

        decoded_data = base64.b64decode(b64_data)
        with open(output_filename, "wb") as f:
            f.write(decoded_data)
        return True
    except FileNotFoundError:
        st.error(f"알람 파일 '{file_path}'를 찾을 수 없습니다. Base64 파일을 만들고 같은 폴더에 넣어주세요.")
        return False
    except Exception as e:
        st.error(f"Base64 디코딩 중 오류 발생: {e}")
        return False

# Base64 디코딩 및 Pygame 초기화
if decode_alarm_sound(ALARM_FILE_PATH, ALARM_WAV_FILENAME):
    try:
        pygame.mixer.init()
        ALARM_SOUND = pygame.mixer.Sound(ALARM_WAV_FILENAME)
    except pygame.error as e:
        st.error(f"Pygame 사운드 초기화 실패: {e}")
else:
    pass


# 알람 및 볼륨 설정
alarm_playing = False
last_alarm_time = 0.0
ALARM_INTERVAL = 1
BASE_VOLUME = 0.3
MAX_VOLUME = 1.0
RAMP_DURATION = 2.0

def play_alarm(now, eyes_closed_time, EYE_CLOSED_TIME_SEC):
    global last_alarm_time, alarm_playing
    
    if 'ALARM_SOUND' not in globals():
        return

    extra = max(0.0, eyes_closed_time - EYE_CLOSED_TIME_SEC)
    ratio = min(1.0, extra / RAMP_DURATION)
    volume = BASE_VOLUME + (MAX_VOLUME - BASE_VOLUME) * ratio
    volume = max(0.0, min(1.0, volume)) 

    if now - last_alarm_time >= ALARM_INTERVAL:
        ALARM_SOUND.stop()
        ALARM_SOUND.set_volume(volume)
        ALARM_SOUND.play()
        last_alarm_time = now
        alarm_playing = True

def stop_alarm():
    global alarm_playing
    if 'ALARM_SOUND' in globals() and alarm_playing:
        ALARM_SOUND.stop()
        alarm_playing = False

# ========================
# 2. MediaPipe 준비
# ========================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

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
# 3. Streamlit UI & 메인 루프 
# ========================

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    .stTextInput > div > div > input {
        text-align: center;
        font-size: 1.5rem;
        padding: 10px;
        width: 100%;
    }
    .stRadio > label {
        justify-content: center;
    }
    .stButton > button {
        width: 150px;
        height: 50px;
        font-size: 1.2rem;
        margin: 10px;
    }
    .study-goal {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
        margin-bottom: 20px;
        text-align: center;
    }
    .main-timer {
        font-size: 5rem;
        font-weight: bold;
        color: #f44336;
        margin-bottom: 20px;
        text-align: center;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
        font-size: 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)


if 'study_started' not in st.session_state:
    st.session_state.study_started = False
if 'study_goal' not in st.session_state:
    st.session_state.study_goal = ""
if 'EYE_CLOSED_TIME_SEC' not in st.session_state:
    st.session_state.EYE_CLOSED_TIME_SEC = 3.0
if 'study_session_start_time' not in st.session_state:
    st.session_state.study_session_start_time = 0.0
if 'focused_time' not in st.session_state:
    st.session_state.focused_time = 0.0
if 'drowsy_time' not in st.session_state:
    st.session_state.drowsy_time = 0.0
if 'is_paused' not in st.session_state:
    st.session_state.is_paused = False
if 'total_elapsed_time' not in st.session_state:
    st.session_state.total_elapsed_time = 0.0
if 'd_day_target' not in st.session_state: # [추가] D-Day 날짜 저장 변수
    st.session_state.d_day_target = date.today()


def format_time(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"

if not st.session_state.study_started:
    # --- 설정 화면 ---
    st.markdown("<h2 style='text-align: center;'>오늘의 공부 목표는 무엇인가요?</h2>", unsafe_allow_html=True)
    study_goal_input = st.text_input("", placeholder="예: Streamlit 앱 개발, 선형대수학 복습")

    # [추가] --- D-Day 날짜 입력 ---
    st.markdown("<h4 style='text-align: center; margin-top: 20px;'>📅 시험(목표) 날짜를 선택하세요</h4>", unsafe_allow_html=True)
    col_d1, col_d2, col_d3 = st.columns([1, 2, 1])
    with col_d2:
        d_day_input = st.date_input(
            "날짜 선택",
            value=date.today(),
            min_value=date.today(),
            label_visibility="collapsed"
        )
    # ---------------------------

    st.markdown("<h3 style='text-align: center;'>집중 모드(졸음 감지 민감도)를 선택하세요.</h3>", unsafe_allow_html=True)
    sensitivity_options = {
        "1단계 (피곤해요): 눈 감음 5초 허용 (느슨한 감지)": 5.0,
        "2단계 (보통이에요): 눈 감음 3초 허용 (기본)": 3.0,
        "3단계 (집중할래요): 눈 감음 2초 허용 (엄격)": 2.0,
        "4단계 (스파르타): 눈 깜빡임이 느려지기만 해도 경고 (초고강도)": 0.5,
    }
    selected_option = st.radio(
        "",
        list(sensitivity_options.keys()),
        index=1,
        key="sensitivity_radio"
    )

    if st.button("공부 시작", key="start_study_button"):
        if study_goal_input:
            st.session_state.study_goal = study_goal_input
            st.session_state.d_day_target = d_day_input # [추가] 목표 날짜 저장
            st.session_state.EYE_CLOSED_TIME_SEC = sensitivity_options[selected_option]
            st.session_state.study_started = True
            st.session_state.study_session_start_time = time.time()
            st.session_state.focused_time = 0.0
            st.session_state.drowsy_time = 0.0
            st.session_state.is_paused = False
            st.session_state.total_elapsed_time = 0.0
            st.rerun()
        else:
            st.warning("공부 목표를 입력해주세요!")
else:
    # --- 학습 진행 화면 ---
    
    # [추가] --- D-Day 계산 및 표시 ---
    today = date.today()
    target = st.session_state.d_day_target
    delta = target - today
    days_left = delta.days

    if days_left > 0:
        d_day_str = f"D-{days_left}"
        d_day_color = "#FF5722" # 주황색
    elif days_left == 0:
        d_day_str = "D-Day"
        d_day_color = "#F44336" # 빨간색
    else:
        d_day_str = f"D+{abs(days_left)}" # 지난 날짜
        d_day_color = "#9E9E9E" # 회색

    st.markdown(f"""
        <div style='text-align: center; margin-bottom: 10px;'>
            <span style='font-size: 1.5rem; font-weight: bold; background-color: {d_day_color}; color: white; padding: 5px 15px; border-radius: 20px; vertical-align: middle; margin-right: 10px;'>
                {d_day_str}
            </span>
            <div class='study-goal' style='margin-top: 10px;'>
                {st.session_state.study_goal}
            </div>
        </div>
    """, unsafe_allow_html=True)
    # ---------------------------

    study_timer_placeholder = st.empty()
    webcam_placeholder = st.empty()
    status_placeholder = st.empty()

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("잠시 멈춤" if not st.session_state.is_paused else "다시 시작", key="pause_button"):
            st.session_state.is_paused = not st.session_state.is_paused
            stop_alarm()
            if not st.session_state.is_paused:
                # 멈춤 시간만큼 시작 시간 보정
                st.session_state.study_session_start_time = time.time() - st.session_state.total_elapsed_time
            st.rerun()

    with col3:
        if st.button("공부 종료", key="end_study_button"):
            stop_alarm()
            st.session_state.study_started = False
            st.session_state.is_paused = True 
            st.rerun()

    # 웹캠 처리
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("웹캠을 열 수 없습니다. 웹캠이 연결되어 있는지 확인하고 다른 프로그램에서 사용 중이 아닌지 확인해주세요.")
        if st.session_state.total_elapsed_time > 0:
            st.button("통계 보기/종료", on_click=lambda: st.session_state.update(study_started=False, is_paused=True))
        st.stop()

    eyes_closed_time = 0.0
    no_face_time = 0.0
    prev_time = time.time()
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while st.session_state.study_started:
            if st.session_state.is_paused:
                study_timer_placeholder.markdown(f"<p class='main-timer'>{format_time(st.session_state.total_elapsed_time)}</p>", unsafe_allow_html=True)
                
                # Paused UI
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    h, w, _ = frame.shape
                    
                    # 반투명 검은색 오버레이
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                    alpha = 0.7
                    combined_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                    # 텍스트
                    cv2.putText(combined_frame, "PAUSED", (w // 2 - 150, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3, cv2.LINE_AA)
                    
                    webcam_placeholder.image(combined_frame, channels="BGR", use_container_width=True)
                
                status_placeholder.text("일시 정지됨")
                time.sleep(0.1)
                continue

            now = time.time()
            dt = now - prev_time
            prev_time = now

            st.session_state.total_elapsed_time = now - st.session_state.study_session_start_time
            study_timer_placeholder.markdown(f"<p class='main-timer'>{format_time(st.session_state.total_elapsed_time)}</p>", unsafe_allow_html=True)


            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img_h, img_w, _ = frame.shape
            results = face_mesh.process(rgb)

            face_detected = False
            eyes_open = False
            current_ear = 0.0
            
            # --- 졸음/자리비움 감지 로직 ---
            if results.multi_face_landmarks:
                face_detected = True
                no_face_time = 0.0

                face_landmarks = results.multi_face_landmarks[0].landmark
                left_ear, left_points = calc_EAR(face_landmarks, LEFT_EYE_IDX, img_w, img_h)
                right_ear, right_points = calc_EAR(face_landmarks, RIGHT_EYE_IDX, img_w, img_h)
                current_ear = (left_ear + right_ear) / 2.0

                # 눈 주변 점 찍기
                for (x, y) in left_points + right_points:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # EAR 임계값 설정
                EAR_THRESHOLD = 0.21 
                if st.session_state.EYE_CLOSED_TIME_SEC <= 1.0: 
                      EAR_THRESHOLD = 0.25 
                    
                if current_ear > EAR_THRESHOLD:
                    eyes_open = True
                    eyes_closed_time = 0.0
                else:
                    eyes_open = False
                    eyes_closed_time += dt

                if eyes_open:
                    stop_alarm()
                    state = "FOCUS"
                    st.session_state.focused_time += dt
                else:
                    if eyes_closed_time >= st.session_state.EYE_CLOSED_TIME_SEC:
                        state = "DROWSY"
                        play_alarm(now, eyes_closed_time, st.session_state.EYE_CLOSED_TIME_SEC)
                        st.session_state.drowsy_time += dt
                    else:
                        state = "BLINK / WARNING"

            else:
                current_ear = 0.0
                eyes_closed_time = 0.0
                no_face_time += dt
                stop_alarm()

                NO_FACE_THRESHOLD = 5.0
                if no_face_time >= NO_FACE_THRESHOLD:
                    state = "AWAY"
                else:
                    state = "LOST"

            # 화면에 상태/시간 표시
            status_text = f"State: {state} | EAR: {current_ear:.3f} | EyesClosed: {eyes_closed_time:.1f}s"
            if state == "DROWSY":
                status_color = (0, 0, 255) # Red
            elif state == "AWAY":
                status_color = (0, 165, 255) # Orange
            elif state == "FOCUS":
                status_color = (0, 255, 0) # Green
            else:
                status_color = (255, 255, 255) # White

            cv2.putText(
                frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
            )
            
            webcam_placeholder.image(frame, channels="BGR", use_container_width=True)
            status_placeholder.text(
                f"집중 시간: {format_time(st.session_state.focused_time)} | "
                f"졸음/이석 시간: {format_time(st.session_state.drowsy_time + no_face_time)} | "
                f"민감도(눈 감음 허용 시간): {st.session_state.EYE_CLOSED_TIME_SEC}초"
            )

            time.sleep(0.01)

# =======================
# 공부 종료 시 통계 출력
# =======================
if not st.session_state.study_started and st.session_state.total_elapsed_time > 0:

    st.markdown("<h3 style='text-align: center;'>📊 공부 결과</h3>", unsafe_allow_html=True)

    total_time = st.session_state.total_elapsed_time
    focus_time = st.session_state.focused_time
    drowsy_time = total_time - focus_time

    st.write(f"**총 공부 시간:** {format_time(total_time)}")
    st.write(f"**집중 시간:** {format_time(focus_time)}")
    st.write(f"**졸음/이석 시간:** {format_time(drowsy_time)}")

    if total_time > 0:
        focus_ratio = (focus_time / total_time) * 100
        st.progress(focus_ratio / 100)
        st.write(f"**집중도:** {focus_ratio:.1f}%")

    # 다시 시작하기 버튼
    if st.button("새로운 공부 시작"):
        st.session_state.clear()
        st.rerun()

# 웹캠 리소스 해제
try:
    if 'cap' in locals() or 'cap' in globals():
        cap.release()
except:
    pass

cv2.destroyAllWindows()
    
# 임시 파일 삭제
if os.path.exists(ALARM_WAV_FILENAME):
        try:
            os.remove(ALARM_WAV_FILENAME)
        except PermissionError:
            pass

st.markdown("""
    <div class="footer">
        © 2025 9조 (Team 9)
    </div>
""", unsafe_allow_html=True)
