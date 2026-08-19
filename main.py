from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd


# ============================================================
# Configuration
# ============================================================

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path("models/hand_landmarker.task")

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

SAMPLE_RATE = 44_100
BLOCK_SIZE = 256
MASTER_VOLUME = 0.30

MAX_HANDS = 2
FINGER_TIPS = (4, 8, 12, 16, 20)

# Keyboard layout
STATUS_BAR_HEIGHT = 42
KEYBOARD_TOP_RATIO = 0.56
KEYBOARD_BOTTOM = DISPLAY_HEIGHT - STATUS_BAR_HEIGHT

# Keep the original, more permissive strike logic.
DOWNWARD_RETRIGGER_SPEED = 0.018
RETRIGGER_COOLDOWN = 0.12

WHITE_KEYS = [
    ("C4", 261.63),
    ("D4", 293.66),
    ("E4", 329.63),
    ("F4", 349.23),
    ("G4", 392.00),
    ("A4", 440.00),
    ("B4", 493.88),
    ("C5", 523.25),
]

BLACK_KEYS = [
    ("C#4", 277.18, 1),
    ("D#4", 311.13, 2),
    ("F#4", 369.99, 4),
    ("G#4", 415.30, 5),
    ("A#4", 466.16, 6),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ============================================================
# MediaPipe model setup
# ============================================================

def ensure_model() -> Path:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100_000:
        return MODEL_PATH

    print("[AirPiano] Downloading MediaPipe Hand Landmarker model...")
    temp_path = MODEL_PATH.with_suffix(".part")

    try:
        with urlopen(MODEL_URL, timeout=120) as response, temp_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        temp_path.replace(MODEL_PATH)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return MODEL_PATH


# ============================================================
# Audio
# ============================================================

@dataclass
class Voice:
    frequency: float
    phase: float = 0.0
    age: float = 0.0
    velocity: float = 1.0


class PianoSynth:
    def __init__(self):
        self.lock = threading.Lock()
        self.voices: list[Voice] = []
        self.muted = False

        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="float32",
            latency="low",
            callback=self._callback,
        )

    def start(self):
        self.stream.start()

    def trigger(self, frequency: float, velocity: float = 1.0):
        with self.lock:
            if self.muted:
                return

            self.voices.append(
                Voice(
                    frequency=float(frequency),
                    velocity=max(0.15, min(1.0, float(velocity))),
                )
            )

            if len(self.voices) > 24:
                self.voices = self.voices[-24:]

    def toggle_mute(self):
        with self.lock:
            self.muted = not self.muted
            if self.muted:
                self.voices.clear()

    def _callback(self, outdata, frames, time_info, status):
        del time_info

        t = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
        mixed = np.zeros(frames, dtype=np.float64)

        with self.lock:
            voices = list(self.voices)
            muted = self.muted

        if not muted:
            surviving: list[Voice] = []

            for voice in voices:
                ages = voice.age + t

                attack = np.minimum(1.0, ages / 0.008)
                decay = np.exp(-2.8 * ages)
                envelope = attack * decay * voice.velocity

                omega = 2.0 * np.pi * voice.frequency
                phase = voice.phase + omega * t

                wave = (
                    0.72 * np.sin(phase)
                    + 0.18 * np.sin(2.0 * phase)
                    + 0.07 * np.sin(3.0 * phase)
                    + 0.03 * np.sin(4.0 * phase)
                )

                mixed += wave * envelope

                voice.age += frames / SAMPLE_RATE
                voice.phase = float(
                    (voice.phase + omega * frames / SAMPLE_RATE) % (2.0 * np.pi)
                )

                if voice.age < 2.2:
                    surviving.append(voice)

            with self.lock:
                if not self.muted:
                    self.voices = surviving

        mixed = np.tanh(mixed * 0.9) * MASTER_VOLUME
        outdata[:, 0] = mixed.astype(np.float32)

    def close(self):
        with self.lock:
            self.voices.clear()

        self.stream.stop()
        self.stream.close()


# ============================================================
# Frame / UI helpers
# ============================================================

def resize_cover(frame, target_width: int, target_height: int):
    """
    Fill the whole output canvas with the webcam image.
    Extra pixels are center-cropped so no gray side bars remain.
    """
    h, w = frame.shape[:2]

    scale = max(target_width / w, target_height / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        frame,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    x1 = max(0, (new_w - target_width) // 2)
    y1 = max(0, (new_h - target_height) // 2)

    return resized[
        y1:y1 + target_height,
        x1:x1 + target_width,
    ].copy()


def build_keyboard():
    keyboard_top = int(DISPLAY_HEIGHT * KEYBOARD_TOP_RATIO)
    keyboard_bottom = KEYBOARD_BOTTOM
    keyboard_height = keyboard_bottom - keyboard_top
    white_width = DISPLAY_WIDTH / len(WHITE_KEYS)

    white_rects = []

    for i, (name, frequency) in enumerate(WHITE_KEYS):
        x1 = int(round(i * white_width))
        x2 = int(round((i + 1) * white_width))

        white_rects.append(
            {
                "name": name,
                "frequency": frequency,
                "rect": (
                    x1,
                    keyboard_top,
                    x2,
                    keyboard_bottom,
                ),
            }
        )

    black_width = int(white_width * 0.58)
    black_height = int(keyboard_height * 0.60)

    black_rects = []

    for name, frequency, gap_index in BLACK_KEYS:
        center_x = int(round(gap_index * white_width))
        x1 = center_x - black_width // 2
        x2 = center_x + black_width // 2

        black_rects.append(
            {
                "name": name,
                "frequency": frequency,
                "rect": (
                    x1,
                    keyboard_top,
                    x2,
                    keyboard_top + black_height,
                ),
            }
        )

    return keyboard_top, white_rects, black_rects


def point_in_rect(x: int, y: int, rect) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= x < x2 and y1 <= y < y2


def hit_test_key(x: int, y: int, white_rects, black_rects):
    for key in black_rects:
        if point_in_rect(x, y, key["rect"]):
            return key

    for key in white_rects:
        if point_in_rect(x, y, key["rect"]):
            return key

    return None


def draw_keyboard(frame, white_rects, black_rects, active_notes, now):
    overlay = frame.copy()

    for key in white_rects:
        x1, y1, x2, y2 = key["rect"]
        active = now - active_notes.get(key["name"], -999.0) < 0.16

        fill = (205, 205, 205) if active else (245, 245, 245)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), 2)

        text = key["name"]
        tw = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            1,
        )[0][0]

        cv2.putText(
            overlay,
            text,
            (x1 + ((x2 - x1) - tw) // 2, y2 - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )

    for key in black_rects:
        x1, y1, x2, y2 = key["rect"]
        active = now - active_notes.get(key["name"], -999.0) < 0.16

        fill = (85, 85, 85) if active else (20, 20, 20)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), 2)

        text = key["name"]
        tw = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            1,
        )[0][0]

        cv2.putText(
            overlay,
            text,
            (x1 + ((x2 - x1) - tw) // 2, y2 - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

    top = white_rects[0]["rect"][1]
    bottom = white_rects[0]["rect"][3]

    camera_roi = frame[top:bottom].copy()
    keyboard_roi = overlay[top:bottom]

    frame[top:bottom] = cv2.addWeighted(
        keyboard_roi,
        0.84,
        camera_roi,
        0.16,
        0,
    )


def draw_hand(frame, landmarks):
    points = []

    for landmark in landmarks:
        points.append(
            (
                int(landmark.x * DISPLAY_WIDTH),
                int(landmark.y * DISPLAY_HEIGHT),
            )
        )

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (215, 215, 215),
            2,
            cv2.LINE_AA,
        )

    for index, point in enumerate(points):
        is_tip = index in FINGER_TIPS
        radius = 8 if is_tip else 4

        cv2.circle(
            frame,
            point,
            radius,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            frame,
            point,
            radius,
            (25, 25, 25),
            2 if is_tip else 1,
            cv2.LINE_AA,
        )

    return points


def draw_header(frame, hands_count, muted, last_note, fps):
    cv2.rectangle(
        frame,
        (18, 18),
        (410, 145),
        (18, 18, 18),
        -1,
    )

    cv2.putText(
        frame,
        "AIR PIANO",
        (38, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.84,
        (250, 250, 250),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Hands: {hands_count}/{MAX_HANDS}",
        (38, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Last note: {last_note}",
        (38, 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Audio: {'MUTED' if muted else 'ON'}",
        (220, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (220, 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Tap fingertips down into the keys",
        (38, 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )


def draw_status_bar(frame):
    y1 = DISPLAY_HEIGHT - STATUS_BAR_HEIGHT

    cv2.rectangle(
        frame,
        (0, y1),
        (DISPLAY_WIDTH, DISPLAY_HEIGHT),
        (18, 18, 18),
        -1,
    )

    cv2.putText(
        frame,
        "SPACE = mute/unmute    |    Q / ESC = quit",
        (26, y1 + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# Camera / detection helpers
# ============================================================

def open_webcam():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def get_hand_label(result, hand_index):
    try:
        categories = result.handedness[hand_index]

        if categories:
            name = categories[0].category_name
            if name:
                return str(name)
    except Exception:
        pass

    return f"Hand{hand_index}"


# ============================================================
# Main
# ============================================================

def main():
    model_path = ensure_model()

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    cap = open_webcam()

    synth = PianoSynth()
    synth.start()

    _, white_rects, black_rects = build_keyboard()

    start_time = time.perf_counter()
    previous_time = start_time
    last_timestamp_ms = -1
    smoothed_fps = 0.0

    finger_state = {}
    last_trigger_time = {}

    active_notes = {}
    last_note = "-"
    last_note_time = -999.0

    window_name = "Air Piano - Finger Controlled Piano"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        window_name,
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT,
    )

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, raw_frame = cap.read()

                if not ok:
                    break

                raw_frame = cv2.flip(raw_frame, 1)

                frame = resize_cover(
                    raw_frame,
                    DISPLAY_WIDTH,
                    DISPLAY_HEIGHT,
                )

                now = time.perf_counter()

                timestamp_ms = max(
                    int((now - start_time) * 1000),
                    last_timestamp_ms + 1,
                )
                last_timestamp_ms = timestamp_ms

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                draw_keyboard(
                    frame,
                    white_rects,
                    black_rects,
                    active_notes,
                    now,
                )

                hands_count = len(result.hand_landmarks)
                currently_seen_fingers = set()

                for hand_index, landmarks in enumerate(result.hand_landmarks):
                    hand_label = get_hand_label(
                        result,
                        hand_index,
                    )

                    points = draw_hand(
                        frame,
                        landmarks,
                    )

                    for tip_id in FINGER_TIPS:
                        finger_id = (
                            hand_label,
                            tip_id,
                        )

                        currently_seen_fingers.add(
                            finger_id
                        )

                        tip = landmarks[tip_id]

                        x_px, y_px = points[tip_id]
                        y_norm = float(tip.y)

                        key = hit_test_key(
                            x_px,
                            y_px,
                            white_rects,
                            black_rects,
                        )

                        current_note = (
                            key["name"]
                            if key
                            else None
                        )

                        previous = finger_state.get(
                            finger_id,
                            {
                                "y": y_norm,
                                "note": None,
                            },
                        )

                        dy = (
                            y_norm
                            - previous["y"]
                        )

                        previous_note = (
                            previous["note"]
                        )

                        # ORIGINAL ACCURATE BEHAVIOR:
                        # entering a new key can trigger immediately
                        entered_new_key = (
                            current_note is not None
                            and current_note != previous_note
                        )

                        # If staying over the same key, a clear downward tap
                        # retriggers it after a short cooldown.
                        strong_down_press = (
                            current_note is not None
                            and current_note == previous_note
                            and dy >= DOWNWARD_RETRIGGER_SPEED
                            and now
                            - last_trigger_time.get(
                                finger_id,
                                -999.0,
                            )
                            >= RETRIGGER_COOLDOWN
                        )

                        should_trigger = False

                        if entered_new_key:
                            # Same permissive threshold as the earlier version.
                            should_trigger = (
                                dy > -0.006
                            )

                        if strong_down_press:
                            should_trigger = True

                        if (
                            should_trigger
                            and key is not None
                        ):
                            velocity = max(
                                0.35,
                                min(
                                    1.0,
                                    0.55
                                    + max(0.0, dy)
                                    * 12.0,
                                ),
                            )

                            synth.trigger(
                                key["frequency"],
                                velocity=velocity,
                            )

                            active_notes[
                                key["name"]
                            ] = now

                            last_trigger_time[
                                finger_id
                            ] = now

                            last_note = key["name"]
                            last_note_time = now

                        finger_state[
                            finger_id
                        ] = {
                            "y": y_norm,
                            "note": current_note,
                        }

                        if key is not None:
                            cv2.circle(
                                frame,
                                (x_px, y_px),
                                12,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )

                stale = [
                    finger_id
                    for finger_id in finger_state
                    if finger_id
                    not in currently_seen_fingers
                ]

                for finger_id in stale:
                    finger_state.pop(
                        finger_id,
                        None,
                    )

                if now - last_note_time > 1.4:
                    last_note = "-"

                dt = max(
                    now - previous_time,
                    1e-6,
                )
                previous_time = now

                instant_fps = 1.0 / dt

                smoothed_fps = (
                    instant_fps
                    if smoothed_fps == 0.0
                    else 0.90 * smoothed_fps
                    + 0.10 * instant_fps
                )

                draw_header(
                    frame,
                    hands_count=hands_count,
                    muted=synth.muted,
                    last_note=last_note,
                    fps=smoothed_fps,
                )

                draw_status_bar(
                    frame
                )

                cv2.imshow(
                    window_name,
                    frame,
                )

                key_code = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key_code in (
                    ord("q"),
                    ord("Q"),
                    27,
                ):
                    break

                if key_code == 32:
                    synth.toggle_mute()

    finally:
        synth.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
