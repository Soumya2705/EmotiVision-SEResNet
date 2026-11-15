
import os
import threading
import time
from datetime import datetime

import av
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from streamlit_webrtc import WebRtcMode, webrtc_streamer


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FER Live Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "FER_Final_Model.pth")
CAPTURE_DIR = os.path.join(PROJECT_DIR, "Captures")
os.makedirs(CAPTURE_DIR, exist_ok=True)

CLASS_NAMES = [
    "Angry", "Disgust", "Fear", "Happy",
    "Neutral", "Sad", "Surprise"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VIDEO_MAX_SECONDS = 60
VIDEO_FPS = 20.0
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# Shared state: accessed by the Streamlit main thread and WebRTC callback.
@st.cache_resource
def get_capture_state():
    # This object survives Streamlit script reruns caused by button clicks.
    return {
        "lock": threading.RLock(),
        "latest_frame": None,
        "latest_results": [],
        "photo_path": None,
        "photo_results": [],
        "photo_time": None,
        "photo_preview_until": None,
        "camera_mode": "photo",
        "recording": False,
        "recording_started": None,
        "writer": None,
        "video_path": None,
        "video_frames_written": 0,
        "video_emotion_counts": {name: 0 for name in CLASS_NAMES},
        "saved_video_path": None,
        "saved_video_frames": 0,
        "saved_video_emotion_counts": {},
        "recording_finished_reason": None,
    }


CAPTURE_STATE = get_capture_state()


# ============================================================
# 2. MODEL ARCHITECTURE
# Exact SE-ResNet architecture used for EXP05.
# ============================================================

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class SEResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels, reduction=8)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.drop(out)
        return self.act(out + identity)


class FER_SEResNetScratch(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )

        self.stage1 = nn.Sequential(
            SEResidualBlock(32, 32, stride=1, dropout=0.02),
            SEResidualBlock(32, 32, stride=1, dropout=0.02)
        )
        self.stage2 = nn.Sequential(
            SEResidualBlock(32, 64, stride=2, dropout=0.04),
            SEResidualBlock(64, 64, stride=1, dropout=0.04)
        )
        self.stage3 = nn.Sequential(
            SEResidualBlock(64, 128, stride=2, dropout=0.06),
            SEResidualBlock(128, 128, stride=1, dropout=0.06)
        )
        self.stage4 = nn.Sequential(
            SEResidualBlock(128, 192, stride=2, dropout=0.08),
            SEResidualBlock(192, 192, stride=1, dropout=0.08)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=5**0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for module in self.modules():
            if isinstance(module, SEResidualBlock):
                nn.init.zeros_(module.bn2.weight)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return self.head(x)


# ============================================================
# 3. LOAD MODEL AND CASCADE DETECTORS
# ============================================================

@st.cache_resource
def load_fer_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    model = FER_SEResNetScratch(num_classes=7).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


@st.cache_resource
def load_face_detectors():
    base = cv2.data.haarcascades

    face_path = os.path.join(
        base, "haarcascade_frontalface_default.xml"
    )
    eye_path = os.path.join(
        base, "haarcascade_eye_tree_eyeglasses.xml"
    )

    face_detector = cv2.CascadeClassifier(face_path)
    eye_detector = cv2.CascadeClassifier(eye_path)

    if face_detector.empty():
        raise RuntimeError("Could not load OpenCV frontal-face cascade.")
    if eye_detector.empty():
        raise RuntimeError("Could not load OpenCV eye cascade.")

    return face_detector, eye_detector


# ============================================================
# 4. PREPROCESSING AND MODEL INFERENCE
# ============================================================

inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

MODEL_LOCK = threading.Lock()


def predict_face(face_gray, model):
    face_pil = Image.fromarray(face_gray)
    face_tensor = (
        inference_transform(face_pil)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with MODEL_LOCK:
        with torch.inference_mode():
            logits = model(face_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

    predicted_id = int(probabilities.argmax().item())
    confidence = float(probabilities[predicted_id].item())

    return (
        CLASS_NAMES[predicted_id],
        confidence,
        probabilities.detach().cpu().numpy()
    )


# ============================================================
# 5. STRICT FACE DETECTION
# - Higher neighbour threshold
# - Face aspect-ratio check
# - Verify a pair of eyes in the upper face region
# - Suppress overlapping duplicate boxes
# ============================================================

def boxes_overlap_enough(a, b, threshold=0.40):
    ax, ay, aw, ah = map(int, a)
    bx, by, bw, bh = map(int, b)

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    if x2 <= x1 or y2 <= y1:
        return False

    intersection = (x2 - x1) * (y2 - y1)
    smaller_area = min(aw * ah, bw * bh)

    return intersection / max(smaller_area, 1) >= threshold


def has_two_eye_candidates(face_gray, eye_detector):
    h, w = face_gray.shape[:2]

    # Eyes should be in the upper part of a frontal face.
    upper_face = face_gray[:int(h * 0.68), :]

    min_eye = max(12, int(min(w, h) * 0.10))

    eyes = eye_detector.detectMultiScale(
        upper_face,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(min_eye, min_eye)
    )

    centers = []
    for ex, ey, ew, eh in eyes:
        cx = ex + ew / 2
        cy = ey + eh / 2

        # Ignore detections too near the edge or low in the face.
        if 0.12 * w <= cx <= 0.88 * w and cy <= 0.68 * h:
            centers.append((cx, cy))

    # Verify at least one plausible left/right eye pair.
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            x1, y1 = centers[i]
            x2, y2 = centers[j]

            horizontal_gap = abs(x1 - x2)
            vertical_gap = abs(y1 - y2)

            if (
                horizontal_gap >= 0.22 * w
                and vertical_gap <= 0.22 * h
            ):
                return True

    return False


def detect_valid_faces(gray, face_detector, eye_detector, min_neighbors=8):
    min_dim = min(gray.shape[:2])
    min_face_size = max(60, int(min_dim * 0.12))

    try:
        rectangles, detection_votes = face_detector.detectMultiScale2(
            gray,
            scaleFactor=1.10,
            minNeighbors=int(min_neighbors),
            minSize=(min_face_size, min_face_size)
        )
    except AttributeError:
        rectangles = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.10,
            minNeighbors=int(min_neighbors),
            minSize=(min_face_size, min_face_size)
        )
        detection_votes = np.full(len(rectangles), min_neighbors)

    candidates = []

    for rect, votes in zip(rectangles, detection_votes):
        x, y, w, h = map(int, rect)

        if w <= 0 or h <= 0:
            continue

        aspect_ratio = w / float(h)
        if not 0.78 <= aspect_ratio <= 1.28:
            continue

        # A candidate must contain a plausible pair of eyes.
        face_roi = gray[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue

        if not has_two_eye_candidates(face_roi, eye_detector):
            continue

        candidates.append((x, y, w, h, int(votes)))

    # Keep stronger candidate when detections substantially overlap.
    candidates.sort(key=lambda item: item[4], reverse=True)

    selected = []
    for candidate in candidates:
        rect = candidate[:4]

        if any(
            boxes_overlap_enough(rect, kept[:4])
            for kept in selected
        ):
            continue

        selected.append(candidate)

    return [item[:4] for item in selected]


# ============================================================
# 6. FRAME PROCESSING
# ============================================================

def process_frame(
    frame,
    model,
    face_detector,
    eye_detector,
    min_neighbors=8,
    mirror=False
):
    if mirror:
        frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detect_valid_faces(
        gray,
        face_detector,
        eye_detector,
        min_neighbors=min_neighbors
    )

    results = []

    for (x, y, w, h) in faces:
        face_crop = gray[y:y+h, x:x+w]

        if face_crop.size == 0:
            continue

        emotion, confidence, probabilities = predict_face(
            face_crop, model
        )

        color = (40, 190, 125)
        label = f"{emotion}  {confidence * 100:.1f}%"

        cv2.rectangle(
            frame, (x, y), (x+w, y+h), color, 2
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.48, min(0.65, w / 650))
        thickness = 2

        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        label_y = max(y, text_h + baseline + 8)
        label_top = label_y - text_h - baseline - 8

        cv2.rectangle(
            frame,
            (x, label_top),
            (min(x + text_w + 14, frame.shape[1] - 1), label_y),
            color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (x + 7, label_y - baseline - 4),
            font,
            font_scale,
            (15, 20, 20),
            thickness,
            cv2.LINE_AA
        )

        results.append({
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": probabilities
        })

    if not faces:
        cv2.putText(
            frame,
            "No clear frontal face detected",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 165, 255),
            2,
            cv2.LINE_AA
        )

    return frame, results


# ============================================================
# 7. PHOTO / SHORT VIDEO RECORDING
# Video is written directly to disk and capped at 60 seconds.
# ============================================================

def finish_recording(reason="stopped by user"):
    """Stop the active writer and return a verified recording summary."""
    with CAPTURE_STATE["lock"]:
        writer = CAPTURE_STATE["writer"]
        CAPTURE_STATE["writer"] = None
        CAPTURE_STATE["recording"] = False
        CAPTURE_STATE["recording_started"] = None
        CAPTURE_STATE["recording_finished_reason"] = reason

        if writer is not None:
            writer.release()

        video_path = CAPTURE_STATE["video_path"]
        frames_written = int(CAPTURE_STATE["video_frames_written"])
        emotion_counts = dict(CAPTURE_STATE["video_emotion_counts"])

        valid_file = (
            bool(video_path)
            and os.path.isfile(video_path)
            and os.path.getsize(video_path) > 1000
            and frames_written > 0
        )

        if valid_file:
            CAPTURE_STATE["saved_video_path"] = video_path
            CAPTURE_STATE["saved_video_frames"] = frames_written
            CAPTURE_STATE["saved_video_emotion_counts"] = emotion_counts
        else:
            CAPTURE_STATE["saved_video_path"] = None
            CAPTURE_STATE["saved_video_frames"] = frames_written
            CAPTURE_STATE["saved_video_emotion_counts"] = emotion_counts

        return {
            "path": video_path if valid_file else None,
            "frames": frames_written,
            "emotion_counts": emotion_counts,
            "reason": reason,
        }


def start_recording():
    with CAPTURE_STATE["lock"]:
        if CAPTURE_STATE["recording"]:
            return False, "Recording is already running."

        latest = CAPTURE_STATE["latest_frame"]
        if latest is None:
            return False, "Start the webcam first and wait for a live frame."

        height, width = latest.shape[:2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(
            CAPTURE_DIR, f"FER_ShortVideo_{timestamp}.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            video_path, fourcc, VIDEO_FPS, (width, height)
        )

        if not writer.isOpened():
            writer.release()
            return False, (
                "MP4 writer could not be opened. Check the Captures folder "
                "and available disk space."
            )

        CAPTURE_STATE["writer"] = writer
        CAPTURE_STATE["video_path"] = video_path
        CAPTURE_STATE["video_frames_written"] = 0
        CAPTURE_STATE["video_emotion_counts"] = {
            name: 0 for name in CLASS_NAMES
        }
        CAPTURE_STATE["recording_started"] = time.monotonic()
        CAPTURE_STATE["recording"] = True
        CAPTURE_STATE["recording_finished_reason"] = None
        CAPTURE_STATE["saved_video_path"] = None

        return True, video_path


def save_photo_from_latest_frame():
    with CAPTURE_STATE["lock"]:
        latest = (
            CAPTURE_STATE["latest_frame"].copy()
            if CAPTURE_STATE["latest_frame"] is not None
            else None
        )
        # Copy prediction data before returning to the Streamlit thread.
        latest_results = [
            {
                "emotion": item["emotion"],
                "confidence": float(item["confidence"]),
                "probabilities": np.asarray(
                    item["probabilities"], dtype=float
                ).copy()
            }
            for item in CAPTURE_STATE["latest_results"]
        ]

    if latest is None:
        return None, []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    photo_path = os.path.join(
        CAPTURE_DIR, f"FER_Photo_{timestamp}.jpg"
    )

    success = cv2.imwrite(photo_path, latest)
    return (photo_path, latest_results) if success else (None, [])


# ============================================================
# 8. WEBCAM CALLBACK
# ============================================================

def make_video_frame_callback(model, face_detector, eye_detector, sensitivity):
    def video_frame_callback(frame):
        image = frame.to_ndarray(format="bgr24")

        # Constrain the webcam stream to a moderate 640x480 preview.
        h, w = image.shape[:2]
        scale = min(VIDEO_WIDTH / w, VIDEO_HEIGHT / h, 1.0)
        if scale < 1.0:
            image = cv2.resize(
                image,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA
            )

        processed, results = process_frame(
            image,
            model,
            face_detector,
            eye_detector,
            min_neighbors=sensitivity,
            mirror=True
        )

        with CAPTURE_STATE["lock"]:
            CAPTURE_STATE["latest_frame"] = processed.copy()
            CAPTURE_STATE["latest_results"] = [
                {
                    "emotion": item["emotion"],
                    "confidence": float(item["confidence"]),
                    "probabilities": np.asarray(
                        item["probabilities"], dtype=float
                    ).copy()
                }
                for item in results
            ]

            if CAPTURE_STATE["recording"]:
                elapsed = (
                    time.monotonic()
                    - CAPTURE_STATE["recording_started"]
                )

                if elapsed >= VIDEO_MAX_SECONDS:
                    writer = CAPTURE_STATE["writer"]
                    CAPTURE_STATE["writer"] = None
                    CAPTURE_STATE["recording"] = False
                    CAPTURE_STATE["recording_started"] = None
                    CAPTURE_STATE["recording_finished_reason"] = (
                        "60-second limit reached"
                    )

                    if writer is not None:
                        writer.release()

                    video_path = CAPTURE_STATE["video_path"]
                    frames_written = int(
                        CAPTURE_STATE["video_frames_written"]
                    )

                    if (
                        video_path
                        and os.path.isfile(video_path)
                        and os.path.getsize(video_path) > 1000
                        and frames_written > 0
                    ):
                        CAPTURE_STATE["saved_video_path"] = video_path
                        CAPTURE_STATE["saved_video_frames"] = frames_written
                        CAPTURE_STATE["saved_video_emotion_counts"] = dict(
                            CAPTURE_STATE["video_emotion_counts"]
                        )

                else:
                    writer = CAPTURE_STATE["writer"]
                    if writer is not None:
                        writer.write(processed)
                        CAPTURE_STATE["video_frames_written"] += 1

                        # Count model predictions per detected face per frame.
                        for item in results:
                            emotion = item["emotion"]
                            if emotion in CAPTURE_STATE["video_emotion_counts"]:
                                CAPTURE_STATE["video_emotion_counts"][emotion] += 1

        return av.VideoFrame.from_ndarray(
            processed, format="bgr24"
        )

    return video_frame_callback


# ============================================================
# WEBCAM ACTION PANEL
# Fragment reruns keep the WebRTC component alive when buttons are clicked.
# It also refreshes the detected emotion and 60-second timer every second.
# ============================================================

@st.fragment(run_every="0.5s")
def render_webcam_action_panel(selected_mode):

    with CAPTURE_STATE["lock"]:
        current_results = list(CAPTURE_STATE["latest_results"])
        recording_now = CAPTURE_STATE["recording"]
        started_at = CAPTURE_STATE["recording_started"]
        finish_reason = CAPTURE_STATE["recording_finished_reason"]
        saved_video_path = CAPTURE_STATE["saved_video_path"]
        saved_video_frames = CAPTURE_STATE["saved_video_frames"]
        saved_video_counts = dict(CAPTURE_STATE["saved_video_emotion_counts"])
        saved_photo_path = CAPTURE_STATE["photo_path"]
        saved_photo_results = list(CAPTURE_STATE["photo_results"])
        saved_photo_time = CAPTURE_STATE["photo_time"]
        photo_preview_until = CAPTURE_STATE["photo_preview_until"]
        current_video_path = CAPTURE_STATE["video_path"]

    # Current live prediction summary.
    st.markdown("#### Live Emotion")
    if current_results:
        metric_cols = st.columns(min(3, len(current_results)))
        for index, item in enumerate(current_results):
            with metric_cols[index % len(metric_cols)]:
                st.metric(
                    f"Face {index + 1} · {item['emotion']}",
                    f"{item['confidence'] * 100:.1f}%"
                )
    else:
        st.caption(
            "Your live prediction will appear here after a clear face is detected."
        )

    st.divider()

    # --------------------------------------------------------
    # LIVE DETECTION MODE
    # --------------------------------------------------------
    if selected_mode == "Live Detection":
        st.success(
            "LIVE DETECTION ACTIVE · Your camera stays on and predictions update continuously."
        )
        st.caption("Nothing is saved in this mode.")

    # --------------------------------------------------------
    # PHOTO MODE: phone-style shutter + 5-second result preview
    # --------------------------------------------------------
    elif selected_mode == "Capture Photo":
        st.markdown("### Capture a Moment")
        st.markdown(
            "Keep your face inside the green box, then tap the round shutter button."
        )

        shutter_left, shutter_center, shutter_right = st.columns([1, 1, 1])
        with shutter_center:
            pressed = st.button(
                "●",
                type="primary",
                use_container_width=True,
                key="shutter_button",
                help="Take and save a photo with the detected facial expression."
            )
            st.markdown(
                "<div style='text-align:center;font-weight:700;letter-spacing:1px;"
                "font-size:0.78rem;margin-top:-0.4rem;'>TAP TO CAPTURE</div>",
                unsafe_allow_html=True
            )

        if pressed:
            photo_path, photo_results = save_photo_from_latest_frame()

            if photo_path:
                with CAPTURE_STATE["lock"]:
                    CAPTURE_STATE["photo_path"] = photo_path
                    CAPTURE_STATE["photo_results"] = photo_results
                    CAPTURE_STATE["photo_time"] = datetime.now().strftime(
                        "%d %b %Y · %I:%M:%S %p"
                    )
                    CAPTURE_STATE["photo_preview_until"] = (
                        time.monotonic() + 5.0
                    )
                st.toast("Photo captured and saved!", icon="📸")
            else:
                st.warning(
                    "Camera is not sending a frame yet. Start the webcam and wait a moment."
                )

        # Read current saved photo/preview state after possible button click.
        with CAPTURE_STATE["lock"]:
            saved_photo_path = CAPTURE_STATE["photo_path"]
            saved_photo_results = list(CAPTURE_STATE["photo_results"])
            saved_photo_time = CAPTURE_STATE["photo_time"]
            photo_preview_until = CAPTURE_STATE["photo_preview_until"]

        preview_active = (
            photo_preview_until is not None
            and time.monotonic() < photo_preview_until
            and saved_photo_path
            and os.path.isfile(saved_photo_path)
        )

        if preview_active:
            seconds_left = max(
                0.0, photo_preview_until - time.monotonic()
            )
            st.markdown("### 📸 Captured Photo")
            st.caption(
                f"Saved successfully · {saved_photo_time} · "
                f"Returning to live camera in {seconds_left:.1f}s"
            )

            preview_col, emotion_col = st.columns([1.35, 1])

            with preview_col:
                st.image(
                    saved_photo_path,
                    caption="Photo with face box and predicted expression",
                    use_container_width=True
                )

            with emotion_col:
                st.markdown("#### Expression at Capture")
                if saved_photo_results:
                    for face_index, item in enumerate(saved_photo_results, start=1):
                        st.metric(
                            f"Face {face_index}: {item['emotion']}",
                            f"{item['confidence'] * 100:.2f}%"
                        )
                        st.progress(
                            max(0.0, min(1.0, float(item["confidence"])))
                        )
                else:
                    st.info(
                        "The image was saved, but no verified face was found in that frame."
                    )

                st.caption(
                    "This is the model's prediction for the exact captured frame."
                )

            # Download is available during the 5-second preview.
            with open(saved_photo_path, "rb") as photo_file:
                st.download_button(
                    "DOWNLOAD PHOTO (JPG)",
                    data=photo_file.read(),
                    file_name=os.path.basename(saved_photo_path),
                    mime="image/jpeg",
                    key="download_photo_button"
                )

        else:
            # Expire the display preview, but keep the JPG on disk.
            if photo_preview_until is not None:
                with CAPTURE_STATE["lock"]:
                    CAPTURE_STATE["photo_preview_until"] = None

            if saved_photo_path and os.path.isfile(saved_photo_path):
                st.caption(
                    f"Last photo saved in Captures: {os.path.basename(saved_photo_path)}"
                )

            st.success(
                "Camera is in live mode. The captured photo preview automatically clears "
                "after 5 seconds; the JPG remains saved in the Captures folder."
            )

    # --------------------------------------------------------
    # SHORT VIDEO MODE: START + PAUSE & SAVE only
    # --------------------------------------------------------
    elif selected_mode == "Record Short Video (60 sec)":
        st.markdown("### Short Video Studio")
        st.markdown(
            "Press **START** to record your live face and emotion label. "
            "Press **PAUSE & SAVE** to finish and save the clip. "
            "The recording also stops automatically at 60 seconds."
        )

        with CAPTURE_STATE["lock"]:
            recording_now = CAPTURE_STATE["recording"]
            started_at = CAPTURE_STATE["recording_started"]

        start_col, pause_col = st.columns([1, 1])

        with start_col:
            if st.button(
                "▶ START",
                type="primary",
                use_container_width=True,
                key="start_recording_button",
                disabled=recording_now
            ):
                ok, message = start_recording()
                if ok:
                    st.toast(
                        "Recording started. Emotion labels are being added to the video.",
                        icon="🔴"
                    )
                else:
                    st.error(message)

        with pause_col:
            if st.button(
                "Ⅱ PAUSE & SAVE",
                use_container_width=True,
                key="pause_save_recording_button",
                disabled=not recording_now
            ):
                result = finish_recording("paused and saved by user")

                if result["path"]:
                    st.toast(
                        f"Video saved: {result['frames']} frames.",
                        icon="✅"
                    )
                    st.success(
                        "Recording paused and the short video has been saved. "
                        "The webcam remains live."
                    )
                else:
                    st.error(
                        f"Video could not be finalised. Frames written: {result['frames']}. "
                        "Try recording for at least 1–2 seconds before pausing."
                    )

        # Refresh the state after a possible action.
        with CAPTURE_STATE["lock"]:
            recording_now = CAPTURE_STATE["recording"]
            started_at = CAPTURE_STATE["recording_started"]
            finish_reason = CAPTURE_STATE["recording_finished_reason"]
            saved_video_path = CAPTURE_STATE["saved_video_path"]
            saved_video_frames = CAPTURE_STATE["saved_video_frames"]
            saved_video_counts = dict(
                CAPTURE_STATE["saved_video_emotion_counts"]
            )
            current_results = list(CAPTURE_STATE["latest_results"])
            current_video_path = CAPTURE_STATE["video_path"]

        if recording_now and started_at is not None:
            elapsed = min(
                time.monotonic() - started_at,
                VIDEO_MAX_SECONDS
            )
            remaining = max(0, int(VIDEO_MAX_SECONDS - elapsed))

            st.error(
                f"● RECORDING · {remaining:02d}s remaining"
            )
            st.progress(
                min(elapsed / VIDEO_MAX_SECONDS, 1.0)
            )

            if current_results:
                item = current_results[0]
                st.info(
                    f"Current prediction: **{item['emotion']}** "
                    f"({item['confidence'] * 100:.1f}%)"
                )

        elif finish_reason == "60-second limit reached":
            st.success(
                "60-second limit reached. The video was automatically saved."
            )

        elif finish_reason == "paused and saved by user":
            st.success(
                "Recording paused and saved. You can press START to create another clip."
            )

        if saved_video_path and os.path.isfile(saved_video_path):
            file_size_kb = os.path.getsize(saved_video_path) / 1024

            st.markdown("### Your Saved Short Video")
            st.caption(
                f"{os.path.basename(saved_video_path)} · "
                f"{saved_video_frames} frames · {file_size_kb:.1f} KB"
            )

            st.video(saved_video_path)

            positive_counts = {
                key: value
                for key, value in saved_video_counts.items()
                if value > 0
            }

            if positive_counts:
                dominant_emotion = max(
                    positive_counts, key=positive_counts.get
                )
                total_predictions = sum(positive_counts.values())

                st.markdown("#### Most Frequent Predicted Expression")
                st.metric(
                    dominant_emotion,
                    f"{positive_counts[dominant_emotion] / total_predictions * 100:.1f}%",
                    help=(
                        "Share of detected face-frame predictions in this clip. "
                        "This is a model output, not a statement about internal emotion."
                    )
                )
                st.bar_chart(positive_counts)
            else:
                st.info(
                    "The video was saved, but no verified face was detected in its frames."
                )

            with open(saved_video_path, "rb") as video_file:
                st.download_button(
                    "DOWNLOAD VIDEO (MP4)",
                    data=video_file.read(),
                    file_name=os.path.basename(saved_video_path),
                    mime="video/mp4",
                    key="download_video_button"
                )

        elif current_video_path and finish_reason and not saved_video_path:
            st.warning(
                "No playable video was created. Start the webcam, wait for a frame, "
                "record for at least 1–2 seconds, then press PAUSE & SAVE."
            )


# ============================================================
# 9. LOAD RESOURCES AND PAGE HEADER
# ============================================================

st.markdown(
    """
    <style>
    /* Camera preview stays large; adjacent toolbar contains all capture actions. */
    .st-key-toolbar_photo_shutter button {
        min-height: 76px !important;
        height: 76px !important;
        width: 76px !important;
        border-radius: 50% !important;
        margin: 0 auto !important;
        border: 6px solid rgba(255,255,255,0.95) !important;
        box-shadow: 0 0 0 3px rgba(215,55,75,0.60) !important;
        font-size: 2rem !important;
    }
    .st-key-toolbar_photo_shutter {
        display: flex;
        justify-content: center;
    }
    .block-container {
        max-width: 1120px;
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stVideo"] video {
        max-height: 430px;
        border-radius: 16px;
    }
    .st-key-shutter_button button {
        width: 82px !important;
        height: 82px !important;
        min-height: 82px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        font-size: 2.0rem !important;
        line-height: 1 !important;
        border: 6px solid rgba(255,255,255,0.92) !important;
        box-shadow: 0 0 0 3px rgba(220,55,75,0.75),
                    0 8px 24px rgba(0,0,0,0.16) !important;
    }
    .st-key-shutter_button button p {
        font-size: 2.1rem !important;
        margin: 0 !important;
    }
    div[data-testid="stMetric"] {
        background: rgba(120, 140, 160, 0.08);
        padding: 0.7rem 0.9rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Facial Emotion Recognition")
st.markdown(
    "Real-time facial expression prediction using your trained **SE-ResNet** model."
)

try:
    model = load_fer_model()
    face_detector, eye_detector = load_face_detectors()
except Exception as error:
    st.error(f"Could not load the model or face detector: {error}")
    st.stop()

with st.sidebar:
    st.header("Model Information")
    st.success("Final Model Loaded")
    st.write("Architecture: SE-ResNet")
    st.write("Input: 48 × 48 Grayscale")
    st.write("Classes: 7")
    st.write("Device:", str(DEVICE))
    st.write("Validation Macro-F1: 66.20%")
    st.divider()

    min_neighbors = st.slider(
        "Face detection strictness",
        min_value=6,
        max_value=12,
        value=8,
        help="Higher values reduce false detections but may miss faces in poor lighting."
    )

    st.caption(
        "A face box is shown only when a plausible pair of eyes is also found."
    )


# ============================================================
# 10. MAIN TABS
# ============================================================

@st.fragment(run_every="0.5s")
def render_camera_side_toolbar():

    with CAPTURE_STATE["lock"]:
        current_mode = CAPTURE_STATE.get("camera_mode", "photo")
        current_results = list(CAPTURE_STATE["latest_results"])
        recording_now = CAPTURE_STATE["recording"]
        started_at = CAPTURE_STATE["recording_started"]
        finish_reason = CAPTURE_STATE["recording_finished_reason"]
        saved_video_path = CAPTURE_STATE["saved_video_path"]
        saved_video_frames = CAPTURE_STATE["saved_video_frames"]
        saved_video_counts = dict(CAPTURE_STATE["saved_video_emotion_counts"])
        saved_photo_path = CAPTURE_STATE["photo_path"]
        saved_photo_results = list(CAPTURE_STATE["photo_results"])
        saved_photo_time = CAPTURE_STATE["photo_time"]
        photo_preview_until = CAPTURE_STATE["photo_preview_until"]
        current_video_path = CAPTURE_STATE["video_path"]

    st.markdown("### Camera Controls")

    # Keep mode options grouped in the camera toolbar.
    selected_label = st.radio(
        "Choose camera mode",
        ["Live", "Photo", "Video"],
        index={"live": 0, "photo": 1, "video": 2}.get(current_mode, 1),
        horizontal=False,
        key="camera_mode_selector",
        label_visibility="collapsed"
    )
    selected_mode = {
        "Live": "live",
        "Photo": "photo",
        "Video": "video"
    }[selected_label]

    if selected_mode != current_mode:
        with CAPTURE_STATE["lock"]:
            CAPTURE_STATE["camera_mode"] = selected_mode
        st.rerun(scope="fragment")

    st.divider()

    # Live prediction status, regardless of selected action.
    st.markdown("**Live Expression**")
    with CAPTURE_STATE["lock"]:
        current_results = list(CAPTURE_STATE["latest_results"])

    if current_results:
        for idx, item in enumerate(current_results[:2], start=1):
            st.metric(
                f"Face {idx} · {item['emotion']}",
                f"{item['confidence'] * 100:.1f}%"
            )
    else:
        st.caption("Waiting for a verified face...")

    st.divider()

    # --------------------------------------------------------
    # LIVE MODE
    # --------------------------------------------------------
    if selected_mode == "live":
        st.markdown("**Live Detection**")
        st.success("Live detection is running. No file will be saved.")
        st.caption(
            "Keep the webcam connected. Emotion labels remain visible on the preview."
        )

    # --------------------------------------------------------
    # PHOTO MODE
    # --------------------------------------------------------
    elif selected_mode == "photo":
        st.markdown("**Photo Capture**")
        st.caption("Keep the camera on. Tap the shutter once to capture.")

        if st.button(
            "●",
            type="primary",
            use_container_width=True,
            key="camera_shutter",
            help="Capture a photo with the face box and expression."
        ):
            photo_path, photo_results = save_photo_from_latest_frame()

            if photo_path:
                with CAPTURE_STATE["lock"]:
                    CAPTURE_STATE["photo_path"] = photo_path
                    CAPTURE_STATE["photo_results"] = photo_results
                    CAPTURE_STATE["photo_time"] = datetime.now().strftime(
                        "%d %b %Y · %I:%M:%S %p"
                    )
                    CAPTURE_STATE["photo_preview_until"] = time.monotonic() + 5.0
                st.toast("Photo captured and saved!", icon="📸")
            else:
                st.warning(
                    "No camera frame is available yet. Start the webcam and wait for a frame."
                )

        # Read photo state after possible capture.
        with CAPTURE_STATE["lock"]:
            saved_photo_path = CAPTURE_STATE["photo_path"]
            saved_photo_results = list(CAPTURE_STATE["photo_results"])
            saved_photo_time = CAPTURE_STATE["photo_time"]
            photo_preview_until = CAPTURE_STATE["photo_preview_until"]

        preview_active = (
            photo_preview_until is not None
            and time.monotonic() < photo_preview_until
            and saved_photo_path
            and os.path.isfile(saved_photo_path)
        )

        if preview_active:
            remaining = max(
                0.0, photo_preview_until - time.monotonic()
            )
            st.success("Captured & saved")
            st.image(
                saved_photo_path,
                caption="Captured frame",
                use_container_width=True
            )
            st.caption(
                f"{saved_photo_time or ''} · Returning to camera in {remaining:.1f}s"
            )

            if saved_photo_results:
                for idx, item in enumerate(saved_photo_results[:2], start=1):
                    st.metric(
                        f"Face {idx}: {item['emotion']}",
                        f"{item['confidence'] * 100:.1f}%"
                    )
            else:
                st.caption("No verified face was present in the captured frame.")

            with open(saved_photo_path, "rb") as photo_file:
                st.download_button(
                    "DOWNLOAD JPG",
                    data=photo_file.read(),
                    file_name=os.path.basename(saved_photo_path),
                    mime="image/jpeg",
                    key="camera_download_photo"
                )
        else:
            if photo_preview_until is not None:
                with CAPTURE_STATE["lock"]:
                    CAPTURE_STATE["photo_preview_until"] = None

            if saved_photo_path and os.path.isfile(saved_photo_path):
                st.caption(f"Last saved: {os.path.basename(saved_photo_path)}")
                with open(saved_photo_path, "rb") as photo_file:
                    st.download_button(
                        "DOWNLOAD LAST JPG",
                        data=photo_file.read(),
                        file_name=os.path.basename(saved_photo_path),
                        mime="image/jpeg",
                        key="camera_download_last_photo"
                    )

    # --------------------------------------------------------
    # VIDEO MODE
    # --------------------------------------------------------
    elif selected_mode == "video":
        st.markdown("**Short Video · 60 sec max**")
        st.caption("START to record. PAUSE & SAVE to finish the clip.")

        if st.button(
            "▶ START",
            type="primary",
            use_container_width=True,
            key="camera_video_start",
            disabled=recording_now
        ):
            ok, message = start_recording()
            if ok:
                st.toast("Video recording started", icon="🔴")
            else:
                st.error(message)

        if st.button(
            "Ⅱ PAUSE & SAVE",
            type="secondary",
            use_container_width=True,
            key="camera_video_pause",
            disabled=not recording_now
        ):
            result = finish_recording("paused and saved by user")

            if result["path"]:
                st.success(
                    f"Saved · {result['frames']} frames"
                )
            else:
                st.error(
                    f"Video was not finalized. {result['frames']} frames written. "
                    "Record for at least 1–2 seconds before saving."
                )

        with CAPTURE_STATE["lock"]:
            recording_now = CAPTURE_STATE["recording"]
            started_at = CAPTURE_STATE["recording_started"]
            finish_reason = CAPTURE_STATE["recording_finished_reason"]
            saved_video_path = CAPTURE_STATE["saved_video_path"]
            saved_video_frames = CAPTURE_STATE["saved_video_frames"]
            saved_video_counts = dict(
                CAPTURE_STATE["saved_video_emotion_counts"]
            )
            current_results = list(CAPTURE_STATE["latest_results"])
            current_video_path = CAPTURE_STATE["video_path"]

        if recording_now and started_at is not None:
            elapsed = min(
                time.monotonic() - started_at,
                VIDEO_MAX_SECONDS
            )
            remaining = max(0, int(VIDEO_MAX_SECONDS - elapsed))
            st.error(f"● REC · {remaining:02d}s remaining")
            st.progress(min(elapsed / VIDEO_MAX_SECONDS, 1.0))

            if current_results:
                item = current_results[0]
                st.caption(
                    f"Current: {item['emotion']} · {item['confidence'] * 100:.1f}%"
                )

        elif finish_reason == "60-second limit reached":
            st.success("60 seconds reached · Video saved automatically.")
        elif finish_reason == "paused and saved by user":
            st.success("Paused · Video saved.")

        if saved_video_path and os.path.isfile(saved_video_path):
            size_kb = os.path.getsize(saved_video_path) / 1024
            st.caption(
                f"{os.path.basename(saved_video_path)} · "
                f"{saved_video_frames} frames · {size_kb:.0f} KB"
            )

            positive_counts = {
                key: value for key, value in saved_video_counts.items()
                if value > 0
            }
            if positive_counts:
                dominant = max(positive_counts, key=positive_counts.get)
                total = sum(positive_counts.values())
                st.metric(
                    "Most frequent prediction",
                    dominant,
                    f"{positive_counts[dominant] / total * 100:.1f}%"
                )

            with open(saved_video_path, "rb") as video_file:
                st.download_button(
                    "DOWNLOAD MP4",
                    data=video_file.read(),
                    file_name=os.path.basename(saved_video_path),
                    mime="video/mp4",
                    key="camera_download_video"
                )

        elif current_video_path and finish_reason and not saved_video_path:
            st.warning(
                "No playable video was created. Keep the webcam running and record "
                "for at least 1–2 seconds before pausing."
            )


webcam_tab, upload_tab = st.tabs([
    "Webcam Studio",
    "Upload Image"
])


# ============================================================
# 11. CAMERA STUDIO
# Camera preview remains prominent at the top.
# Controls are a compact camera-style toolbar on its right.
# ============================================================

with webcam_tab:
    st.subheader("Webcam Studio")
    st.caption(
        "The camera preview stays in place. Choose Live, Photo or Video "
        "from the control panel beside it."
    )

    preview_col, controls_col = st.columns(
        [4.5, 1.35],
        gap="medium",
        vertical_alignment="top"
    )

    rtc_configuration = {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }

    callback = make_video_frame_callback(
        model, face_detector, eye_detector, min_neighbors
    )

    with preview_col:
        st.markdown("#### Live Camera Preview")
        webrtc_ctx = webrtc_streamer(
            key="fer-camera-fixed-preview",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": VIDEO_WIDTH},
                    "height": {"ideal": VIDEO_HEIGHT},
                    "frameRate": {"ideal": VIDEO_FPS, "max": VIDEO_FPS}
                },
                "audio": False
            },
            video_frame_callback=callback,
            video_html_attrs={
                "style": {
                    "width": "100%",
                    "max-width": "100%",
                    "height": "auto",
                    "object-fit": "contain",
                    "margin": "0 auto",
                    "border-radius": "14px"
                },
                "controls": False,
                "autoPlay": True,
                "muted": True
            },
            async_processing=True
        )
        st.caption(
            "Use the webcam START/STOP controls below the preview only to connect "
            "or disconnect the camera."
        )

    with controls_col:
        render_camera_side_toolbar()


# ============================================================
# 12. IMAGE UPLOAD MODE
# ============================================================

with upload_tab:
    st.subheader("Upload an Image for Emotion Detection")

    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG, PNG or BMP image",
        type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(
            image_array, cv2.COLOR_RGB2BGR
        )

        processed_image, results = process_frame(
            image_bgr,
            model,
            face_detector,
            eye_detector,
            min_neighbors=min_neighbors,
            mirror=False
        )

        processed_rgb = cv2.cvtColor(
            processed_image, cv2.COLOR_BGR2RGB
        )

        st.image(
            processed_rgb,
            caption="Detected face and emotion",
            width=720
        )

        if results:
            st.markdown("### Prediction Results")

            for index, result in enumerate(results, start=1):
                with st.container(border=True):
                    st.write(f"**Face {index}: {result['emotion']}**")
                    st.metric(
                        "Model score",
                        f"{result['confidence'] * 100:.2f}%"
                    )

                    probability_data = {
                        CLASS_NAMES[i]: float(result["probabilities"][i])
                        for i in range(7)
                    }
                    st.bar_chart(probability_data)
        else:
            st.warning(
                "No clear frontal face was detected. Try a well-lit, front-facing image."
            )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()
st.caption(
    "Facial Emotion Recognition | SE-ResNet Scratch Model | "
    "OpenCV Face + Eye Verification | PyTorch | Streamlit | WebRTC"
)
st.caption(
    "Predictions are estimates from the model and can be inaccurate. "
    "Lighting, camera angle, occlusion and ambiguous expressions can affect results."
)
