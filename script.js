import { FilesetResolver, FaceLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

// === 1. 변수 선언 ===
let faceLandmarker;
let video = document.getElementById("webcam");
let canvasElement = document.getElementById("output_canvas");
let canvasCtx = canvasElement.getContext("2d");
let lastVideoTime = -1;

// 알람 설정
const alarmAudio = new Audio('alarm.wav');
alarmAudio.loop = true; 

// 상태 변수
let isStudying = false;
let isPaused = false;
let totalElapsedTime = 0;
let focusedTime = 0;
let drowsyTime = 0;
let noFaceTime = 0;
let eyesClosedTime = 0;
let lastFrameTime = 0;

// 설정 변수
let eyeClosedThresholdSec = 3.0;
const EAR_THRESHOLD = 0.21;

// Mediapipe 인덱스
const LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380];

// === 2. 초기화 ===
async function createFaceLandmarker() {
    const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU"
        },
        outputFaceBlendshapes: true,
        runningMode: "VIDEO",
        numFaces: 1
    });
}
createFaceLandmarker();

// D-Day 오늘 날짜로 초기화
document.getElementById('dDayInput').valueAsDate = new Date();

// === 3. 유틸리티 함수 ===
function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function calcEAR(landmarks, indices, width, height) {
    const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
    const p = indices.map(idx => ({
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

// === 4. 메인 로직 함수 (HTML에서 호출 가능하도록 window에 등록) ===

async function startStudy() {
    const goal = document.getElementById('studyGoal').value;
    if (!goal) { alert("공부 목표를 입력해주세요!"); return; }
    
    // D-Day 계산
    const dDayTarget = new Date(document.getElementById('dDayInput').value);
    const today = new Date();
    today.setHours(0,0,0,0);
    dDayTarget.setHours(0,0,0,0);
    
    const diffTime = dDayTarget - today;
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    let dDayText = diffDays > 0 ? `D-${diffDays}` : (diffDays === 0 ? "D-Day" : `D+${Math.abs(diffDays)}`);
    let dDayColor = diffDays > 0 ? "#FF5722" : (diffDays === 0 ? "#F44336" : "#9E9E9E");
    
    document.getElementById('dDayDisplay').innerText = dDayText;
    document.getElementById('dDayDisplay').style.backgroundColor = dDayColor;
    document.getElementById('goalDisplay').innerText = goal;

    // 설정 적용
    const radios = document.getElementsByName('sensitivity');
    for (let radio of radios) {
        if (radio.checked) eyeClosedThresholdSec = parseFloat(radio.value);
    }

    // 화면 전환
    document.getElementById('setupScreen').classList.remove('active');
    document.getElementById('studyScreen').classList.add('active');

    // 웹캠 시작
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    } catch (err) {
        alert("웹캠을 찾을 수 없습니다.");
        console.error(err);
    }

    isStudying = true;
    isPaused = false;
    lastFrameTime = performance.now();
}

function togglePause() {
    isPaused = !isPaused;
    const overlay = document.getElementById('pauseOverlay');
    overlay.style.display = isPaused ? 'flex' : 'none';
    
    if(isPaused) {
        stopAlarm();
    } else {
        lastFrameTime = performance.now();
    }
}

function endStudy() {
    isStudying = false;
    stopAlarm();
    
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }

    document.getElementById('studyScreen').classList.remove('active');
    document.getElementById('resultScreen').classList.add('active');
    
    document.getElementById('resTotal').innerText = formatTime(totalElapsedTime);
    document.getElementById('resFocus').innerText = formatTime(focusedTime);
    document.getElementById('resDrowsy').innerText = formatTime(drowsyTime + noFaceTime);
    
    const focusRatio = totalElapsedTime > 0 ? (focusedTime / totalElapsedTime) * 100 : 0;
    document.getElementById('resProgress').style.width = `${focusRatio}%`;
    document.getElementById('resScore').innerText = `집중도: ${focusRatio.toFixed(1)}%`;
}

// HTML의 onclick 속성에서 이 함수들을 찾을 수 있게 window 객체에 연결
window.startStudy = startStudy;
window.togglePause = togglePause;
window.endStudy = endStudy;


// === 5. 영상 처리 루프 ===
async function predictWebcam() {
    if (!isStudying) return;

    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;
    
    const now = performance.now();
    const deltaTime = (now - lastFrameTime) / 1000;
    lastFrameTime = now;

    if (!isPaused) {
        totalElapsedTime += deltaTime;
        
        let startTimeMs = performance.now();
        if (lastVideoTime !== video.currentTime) {
            lastVideoTime = video.currentTime;
            
            if (faceLandmarker) {
                const results = faceLandmarker.detectForVideo(video, startTimeMs);
                canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
                
                if (results.faceLandmarks.length > 0) {
                    const landmarks = results.faceLandmarks[0];
                    noFaceTime = 0;

                    const leftEAR = calcEAR(landmarks, LEFT_EYE_IDX, video.videoWidth, video.videoHeight);
                    const rightEAR = calcEAR(landmarks, RIGHT_EYE_IDX, video.videoWidth, video.videoHeight);
                    const avgEAR = (leftEAR + rightEAR) / 2.0;

                    let currentThreshold = EAR_THRESHOLD;
                    if (eyeClosedThresholdSec <= 1.0) currentThreshold = 0.25;

                    let state = "";
                    let color = "white";

                    if (avgEAR > currentThreshold) {
                        state = "FOCUS";
                        color = "#00FF00";
                        eyesClosedTime = 0;
                        focusedTime += deltaTime;
                        stopAlarm();
                    } else {
                        eyesClosedTime += deltaTime;
                        if (eyesClosedTime >= eyeClosedThresholdSec) {
                            state = "DROWSY";
                            color = "red";
                            drowsyTime += deltaTime;
                            if (alarmAudio.paused) {
                                alarmAudio.play().catch(e => console.log(e));
                            }
                        } else {
                            state = "BLINK / WARNING";
                            color = "yellow";
                            stopAlarm();
                        }
                    }

                    canvasCtx.fillStyle = "#00FF00";
                    [...LEFT_EYE_IDX, ...RIGHT_EYE_IDX].forEach(idx => {
                        const p = landmarks[idx];
                        canvasCtx.beginPath();
                        canvasCtx.arc(p.x * canvasElement.width, p.y * canvasElement.height, 2, 0, 2 * Math.PI);
                        canvasCtx.fill();
                    });

                    document.getElementById("statusText").innerText = 
                        `State: ${state} | EAR: ${avgEAR.toFixed(3)} | Closed: ${eyesClosedTime.toFixed(1)}s`;
                    document.getElementById("statusText").style.color = color;

                } else {
                    noFaceTime += deltaTime;
                    stopAlarm();
                    
                    if (noFaceTime > 5.0) {
                            document.getElementById("statusText").innerText = "자리 비움 (AWAY)";
                            document.getElementById("statusText").style.color = "orange";
                    } else {
                            document.getElementById("statusText").innerText = "얼굴 찾는 중...";
                    }
                }
            }
        }
    }

    document.getElementById('mainTimer').innerText = formatTime(totalElapsedTime);
    document.getElementById('statsDisplay').innerText = 
        `집중: ${formatTime(focusedTime)} | 졸음/이석: ${formatTime(drowsyTime + noFaceTime)}`;

    window.requestAnimationFrame(predictWebcam);
}