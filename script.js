// === Mediapipe Face Landmarker ===
import { FilesetResolver, FaceLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

// === Firebase (Firestore) 설정 ===
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js";
import {
  getFirestore,
  collection,
  addDoc,
  getDocs,
  query,
  orderBy,
  limit,
  serverTimestamp
} from "https://www.gstatic.com/firebasejs/10.12.0/firebase-firestore.js";

// 🔥 Firebase 콘솔에서 복사한 설정 값
const firebaseConfig = {
  apiKey: "AIzaSyBTj5FitusmvFRNrkcJwrGEI3a80MUhnvw",
  authDomain: "swai09.firebaseapp.com",
  projectId: "swai09",
  storageBucket: "swai09.firebasestorage.app",
  messagingSenderId: "731455422892",
  appId: "1:731455422892:web:9c1b1c466aa7f24a56ba09",
  measurementId: "G-BYSM5M8847"
};

// Firebase 초기화
const firebaseApp = initializeApp(firebaseConfig);
const db = getFirestore(firebaseApp);
const rankingColRef = collection(db, "studySessions");

// === 1. DOM 요소 참조 ===
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

const nicknameInput = document.getElementById("nicknameInput"); // HTML에 새로 추가할 예정

// === 2. 상태 변수 ===
let faceLandmarker;
let lastVideoTime = -1;

// 알람
const alarmAudio = new Audio("alarm.wav");
alarmAudio.loop = true;

// 공부 상태
let isStudying = false;
let isPaused = false;

// 타이머 관련 (초 단위)
let totalElapsedTime = 0;
let focusedTime = 0;
let drowsyTime = 0;
let noFaceTime = 0;
let eyesClosedTime = 0;
let lastFrameTime = 0;

// 설정값
let eyeClosedThresholdSec = 3.0; // 눈 감은 상태 유지 시간 임계값(민감도)
const EAR_THRESHOLD = 0.21;      // 눈 감김 판단 기준 (EAR)
let sensitivityLevel = 2;        // 1~4단계 정도로 저장용
let studyGoal = "";              // 현재 세션 목표 텍스트

// Mediapipe 눈 랜드마크 인덱스
const LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380];

// === 3. 초기화 ===
async function createFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU"
    },
    outputFaceBlendshapes: true,
    runningMode: "VIDEO",
    numFaces: 1
  });
}
createFaceLandmarker().catch((e) => console.error("FaceLandmarker init error:", e));

// D-Day 오늘 날짜로 세팅
document.getElementById("dDayInput").valueAsDate = new Date();

// === 4. 유틸 함수 ===
function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function calcEAR(landmarks, indices, width, height) {
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const p = indices.map((idx) => ({
    x: landmarks[idx].x * width,
    y: landmarks[idx].y * height
  }));
  return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2.0 * dist(p[0], p[3]) + 1e-6);
}

function stopAlarm() {
  if (!alarmAudio.paused) {
    alarmAudio.pause();
    alarmAudio.currentTime = 0;
  }
}

function formatSecondsForRanking(seconds) {
  const min = Math.floor(seconds / 60);
  const sec = Math.floor(seconds % 60);
  return `${min}분 ${sec}초`;
}

// === 5. 공부 시작 ===
async function startStudy() {
  const goalInput = document.getElementById("studyGoal");
  const goal = goalInput.value.trim();
  if (!goal) {
    alert("공부 목표를 입력해주세요!");
    return;
  }
  studyGoal = goal;

  // 닉네임은 없어도 되지만 있으면 랭킹에 표시
  const nickname = nicknameInput ? nicknameInput.value.trim() : "";

  // D-Day 계산
  const dDayTarget = new Date(document.getElementById("dDayInput").value);
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  dDayTarget.setHours(0, 0, 0, 0);

  const diffTime = dDayTarget - today;
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

  let dDayText =
    diffDays > 0
      ? `D-${diffDays}`
      : diffDays === 0
      ? "D-Day"
      : `D+${Math.abs(diffDays)}`;
  let dDayColor =
    diffDays > 0 ? "#FF5722" : diffDays === 0 ? "#F44336" : "#9E9E9E";

  document.getElementById("dDayDisplay").innerText = dDayText;
  document.getElementById("dDayDisplay").style.backgroundColor = dDayColor;
  document.getElementById("goalDisplay").innerText = goal;

  // 민감도 설정 (라디오 버튼 value = 허용 초)
  const radios = document.getElementsByName("sensitivity");
  for (let radio of radios) {
    if (radio.checked) {
      eyeClosedThresholdSec = parseFloat(radio.value);
    }
  }
  // 허용 시간이 짧을수록 더 빡센 단계라고 가정
  if (eyeClosedThresholdSec >= 5) sensitivityLevel = 1;
  else if (eyeClosedThresholdSec >= 3) sensitivityLevel = 2;
  else if (eyeClosedThresholdSec >= 2) sensitivityLevel = 3;
  else sensitivityLevel = 4;

  // 화면 전환
  document.getElementById("setupScreen").classList.remove("active");
  document.getElementById("studyScreen").classList.add("active");
  document.getElementById("resultScreen").classList.remove("active");

  // 타이머/상태 초기화
  totalElapsedTime = 0;
  focusedTime = 0;
  drowsyTime = 0;
  noFaceTime = 0;
  eyesClosedTime = 0;
  lastFrameTime = performance.now();
  stopAlarm();

  // 웹캠 시작
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  } catch (err) {
    alert("웹캠을 찾을 수 없습니다.");
    console.error(err);
    return;
  }

  isStudying = true;
  isPaused = false;
}

// === 6. 일시 정지 / 재시작 ===
function togglePause() {
  isPaused = !isPaused;
  const overlay = document.getElementById("pauseOverlay");
  overlay.style.display = isPaused ? "flex" : "none";

  if (isPaused) {
    stopAlarm();
  } else {
    lastFrameTime = performance.now();
  }
}

// === 7. Firestore 저장 ===
async function saveSessionToRanking({
  nickname,
  goal,
  focusedTime,
  totalElapsedTime,
  drowsyTime,
  noFaceTime,
  sensitivityLevel
}) {
  try {
    await addDoc(rankingColRef, {
      nickname: nickname || "익명",
      goal: goal || "",
      focusedTime, // 초 단위
      totalElapsedTime,
      drowsyTime,
      noFaceTime,
      sensitivityLevel,
      createdAt: serverTimestamp()
    });
    console.log("✅ 랭킹에 세션 기록 저장 완료");
  } catch (err) {
    console.error("🔥 세션 저장 중 오류:", err);
  }
}

// Firestore에서 집중 시간 기준 TOP 10 불러오기
async function loadRanking() {
  const listEl = document.getElementById("rankingList");
  if (!listEl) return; // HTML에 없으면 그냥 패스

  listEl.innerHTML = "<li>랭킹을 불러오는 중입니다...</li>";

  try {
    const q = query(rankingColRef, orderBy("focusedTime", "desc"), limit(10));
    const snap = await getDocs(q);

    if (snap.empty) {
      listEl.innerHTML =
        "<li>아직 기록이 없어요. 첫 번째 집중왕이 되어 보세요!</li>";
      return;
    }

    listEl.innerHTML = "";
    let rank = 1;

    snap.forEach((doc) => {
      const data = doc.data();
      const li = document.createElement("li");
      li.className = "ranking-item";

      const nameSpan = document.createElement("span");
      nameSpan.className = "ranking-name";
      nameSpan.textContent = `${rank}. ${data.nickname || "익명"}`;

      const timeSpan = document.createElement("span");
      timeSpan.className = "ranking-time";
      timeSpan.textContent = formatSecondsForRanking(data.focusedTime || 0);

      li.appendChild(nameSpan);
      li.appendChild(timeSpan);
      listEl.appendChild(li);

      rank++;
    });
  } catch (err) {
    console.error("🔥 랭킹 불러오기 오류:", err);
    listEl.innerHTML =
      "<li>랭킹을 불러오는 데 실패했어요. 잠시 후 다시 시도해 주세요.</li>";
  }
}

// === 8. 공부 종료 ===
async function endStudy() {
  if (!isStudying) return;
  isStudying = false;
  stopAlarm();

  // 화면 전환
  document.getElementById("studyScreen").classList.remove("active");
  document.getElementById("resultScreen").classList.add("active");

  // 결과 표시 (totalElapsedTime 은 이미 초 단위 누적)
  document.getElementById("resTotal").innerText = formatTime(totalElapsedTime);
  document.getElementById("resFocus").innerText = formatTime(focusedTime);
  document.getElementById("resDrowsy").innerText = formatTime(
    drowsyTime + noFaceTime
  );

  const focusRatio =
    totalElapsedTime > 0 ? (focusedTime / totalElapsedTime) * 100 : 0;

  document.getElementById("resProgress").style.width = `${focusRatio}%`;
  document.getElementById(
    "resScore"
  ).innerText = `집중도: ${focusRatio.toFixed(1)}%`;

  // Firestore에 기록 저장 후 랭킹 갱신
  try {
    await saveSessionToRanking({
      nickname: nicknameInput ? nicknameInput.value.trim() : "",
      goal: studyGoal,
      focusedTime,
      totalElapsedTime,
      drowsyTime,
      noFaceTime,
      sensitivityLevel
    });
    await loadRanking();
  } catch (err) {
    console.error("세션 저장/랭킹 갱신 중 오류:", err);
  }

  // 웹캠 스트림 정리(선택)
  if (video.srcObject) {
    const tracks = video.srcObject.getTracks();
    tracks.forEach((t) => t.stop());
    video.srcObject = null;
  }
}

// === 9. 영상 처리 루프 ===
async function predictWebcam() {
  if (!isStudying) return;

  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;

  const now = performance.now();
  const deltaTime = (now - lastFrameTime) / 1000; // 초
  lastFrameTime = now;

  if (!isPaused) {
    totalElapsedTime += deltaTime;

    if (faceLandmarker && lastVideoTime !== video.currentTime) {
      lastVideoTime = video.currentTime;

      const startTimeMsLocal = performance.now();
      const results = faceLandmarker.detectForVideo(video, startTimeMsLocal);

      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        const landmarks = results.faceLandmarks[0];
        noFaceTime = 0;

        const leftEAR = calcEAR(
          landmarks,
          LEFT_EYE_IDX,
          video.videoWidth,
          video.videoHeight
        );
        const rightEAR = calcEAR(
          landmarks,
          RIGHT_EYE_IDX,
          video.videoWidth,
          video.videoHeight
        );
        const avgEAR = (leftEAR + rightEAR) / 2.0;

        let currentThreshold = EAR_THRESHOLD;
        if (eyeClosedThresholdSec <= 1.0) currentThreshold = 0.25;

        let state = "";
        let color = "white";

        if (avgEAR > currentThreshold) {
          // 눈 뜬 상태
          state = "FOCUS";
          color = "#00FF00";
          eyesClosedTime = 0;
          focusedTime += deltaTime;
          stopAlarm();
        } else {
          // 눈 감은 상태
          eyesClosedTime += deltaTime;
          if (eyesClosedTime >= eyeClosedThresholdSec) {
            state = "DROWSY";
            color = "red";
            drowsyTime += deltaTime;
            if (alarmAudio.paused) {
              alarmAudio.play().catch((e) => console.log(e));
            }
          } else {
            state = "BLINK / WARNING";
            color = "yellow";
            stopAlarm();
          }
        }

        // 눈 부분 점 찍기
        canvasCtx.fillStyle = "#00FF00";
        [...LEFT_EYE_IDX, ...RIGHT_EYE_IDX].forEach((idx) => {
          const p = landmarks[idx];
          canvasCtx.beginPath();
          canvasCtx.arc(
            p.x * canvasElement.width,
            p.y * canvasElement.height,
            2,
            0,
            2 * Math.PI
          );
          canvasCtx.fill();
        });

        const statusEl = document.getElementById("statusText");
        statusEl.innerText = `State: ${state} | EAR: ${avgEAR.toFixed(
          3
        )} | Closed: ${eyesClosedTime.toFixed(1)}s`;
        statusEl.style.color = color;
      } else {
        // 얼굴 없음
        noFaceTime += deltaTime;
        stopAlarm();
        const statusEl = document.getElementById("statusText");
        if (noFaceTime > 5.0) {
          statusEl.innerText = "자리 비움 (AWAY)";
          statusEl.style.color = "orange";
        } else {
          statusEl.innerText = "얼굴 찾는 중...";
          statusEl.style.color = "white";
        }
      }
    }
  }

  // 상단 타이머 / 통계 갱신
  document.getElementById("mainTimer").innerText =
    formatTime(totalElapsedTime);
  document.getElementById(
    "statsDisplay"
  ).innerText = `집중: ${formatTime(
    focusedTime
  )} | 졸음/이석: ${formatTime(drowsyTime + noFaceTime)}`;

  window.requestAnimationFrame(predictWebcam);
}

// === 10. 전역에서 쓸 수 있게 등록 ===
window.startStudy = startStudy;
window.togglePause = togglePause;
window.endStudy = endStudy;

// 페이지 로드 시 기존 랭킹 한 번 불러오기 (결과 화면에서 바로 보이도록)
loadRanking().catch((e) => console.error("초기 랭킹 로딩 오류:", e));