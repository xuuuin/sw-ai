import cv2
import time
import pygame
import numpy as np
import mediapipe as mp
import math

# ========================
# 1. Pygame 초기화 & 알람
# ========================
pygame.mixer.init()
ALARM_SOUND = pygame.mixer.Sound("사당로.wav")

alarm_playing = False
last_alarm_time = 0.0
ALARM_INTERVAL = 1
BASE_VOLUME = 0.3          # 최소 볼륨
MAX_VOLUME = 5.0           # 최대 볼륨
RAMP_DURATION = 2.0      # 볼륨이 최대치에 도달하는 데 걸리는 시간(초)


def play_alarm(now, eyes_closed_time):
    """
    눈 감은 시간이 길어질수록 볼륨을 키운다.
    eyes_closed_time: 눈 감고 있는 누적 시간 (초)
    """
    global last_alarm_time

    # 졸음 기준(EYE_CLOSED_TIME_SEC) 이후부터 증가분 계산
    extra = max(0.0, eyes_closed_time - EYE_CLOSED_TIME_SEC)

    # 0 ~ 1 사이 비율로 압축 (RAMP_DURATION초 동안 서서히 0→1)
    ratio = min(1.0, extra / RAMP_DURATION)

    # BASE_VOLUME ~ MAX_VOLUME 사이로 보간
    volume = BASE_VOLUME + (MAX_VOLUME - BASE_VOLUME) * ratio
    volume = max(0.0, min(1.0, volume))  # 안전하게 0~1로 클램프

    # 일정 간격마다만 울리게 하기
    if now - last_alarm_time >= ALARM_INTERVAL:
        ALARM_SOUND.stop()
        ALARM_SOUND.set_volume(volume)   # 🔊 여기서 볼륨 설정
        ALARM_SOUND.play()
        last_alarm_time = now

def stop_alarm():
    global alarm_playing
    if alarm_playing:
        ALARM_SOUND.stop()
        alarm_playing = False

# ========================
# 2. MediaPipe 준비
# ========================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 눈 랜드마크 인덱스 (FaceMesh 기준)
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

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

    ear = (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4) + 1e-6)
    return ear, points

# ========================
# 3. 웹캠 열기
# ========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# ========================
# 4. 시간 / 상태 변수
# ========================
EYE_CLOSED_THRESHOLD = 0.21   # EAR 이 이 값보다 작으면 눈 감은 상태로 간주
EYE_CLOSED_TIME_SEC  = 1.0    # 이렇게 1초 이상 지속되면 졸음
NO_FACE_THRESHOLD    = 5.0    # 얼굴 없음 5초 이상이면 자리비움

eyes_closed_time = 0.0
no_face_time = 0.0
focused_time = 0.0

session_start = time.time()
prev_time = time.time()

state = "INIT"
current_ear = 0.0

print("q 를 누르면 종료합니다.")

# ========================
# 5. FaceMesh 사용
# ========================
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        now = time.time()
        dt = now - prev_time
        prev_time = now

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_h, img_w, _ = frame.shape

        results = face_mesh.process(rgb)

        face_detected = False
        eyes_open = False

        if results.multi_face_landmarks:
            face_detected = True
            no_face_time = 0.0  # 얼굴 보이면 리셋

            face_landmarks = results.multi_face_landmarks[0].landmark

            # 왼쪽/오른쪽 EAR 계산
            left_ear, left_points = calc_EAR(face_landmarks, LEFT_EYE_IDX, img_w, img_h)
            right_ear, right_points = calc_EAR(face_landmarks, RIGHT_EYE_IDX, img_w, img_h)

            current_ear = (left_ear + right_ear) / 2.0

            # 눈 주변 점 찍어보기 (디버그용)
            for (x, y) in left_points + right_points:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # EAR 기준 눈 뜬/감김 판정
            if current_ear > EYE_CLOSED_THRESHOLD:
                eyes_open = True
                eyes_closed_time = 0.0
            else:
                eyes_open = False
                eyes_closed_time += dt

            # 상태 결정
            if eyes_open:
                stop_alarm()
                state = "FOCUS"
                focused_time += dt
            else:
                if eyes_closed_time >= EYE_CLOSED_TIME_SEC:
                    state = "DROWSY"
                    play_alarm(now, eyes_closed_time)
                else:
                    state = "BLINK / WARNING"

        else:
            # 얼굴 안 보임
            current_ear = 0.0
            eyes_closed_time = 0.0
            no_face_time += dt
            stop_alarm()

            if no_face_time >= NO_FACE_THRESHOLD:
                state = "AWAY"
            else:
                state = "LOST"

        # ========================
        # 화면에 상태/시간 표시
        # ========================
        def format_time(sec):
            m = int(sec // 60)
            s = int(sec % 60)
            return f"{m:02d}:{s:02d}"

        elapsed = now - session_start

        cv2.putText(
            frame,
            f"State: {state}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Session: {format_time(elapsed)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Focused: {format_time(focused_time)}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
        )

        cv2.putText(
            frame,
            f"EAR: {current_ear:.3f}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )

        cv2.putText(
            frame,
            f"EyesClosedTime: {eyes_closed_time:.1f}s",
            (10, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
        )

        cv2.imshow("StudyCam - MediaPipe Drowsiness Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

# ========================
# 6. 종료 처리
# ========================
stop_alarm()
cap.release()
cv2.destroyAllWindows()

def format_time(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"

print("총 세션 시간:", format_time(time.time() - session_start))
print("집중 시간:", format_time(focused_time))