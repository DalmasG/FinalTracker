import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OpenMP runtime error

import cv2
import time
import numpy as np
import warnings
from collections import deque
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from sort import Sort  # assumes sort.py is in the same directory

# ── CONFIG ─────────────────────────────────────────────────────────
VIDEO_PATH   = os.path.join("footage", "test_1.mp4")
OUTPUT_PATH  = "test_1_tracked.mp4"
YOLO_MODEL   = "yolov8s.pt"
DET_CONF_TH  = 0.3
PROC_W, PROC_H = 640, 640
SHOW_FPS     = True

# ── Suppress Warnings & Logs ───────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
LOGGER.setLevel("ERROR")

# ── Dummy Box for Fallback Splits ─────────────────────────────────
class DummyBox:
    def __init__(self, xyxy_list, conf, is_fake=True):
        self.xyxy = [xyxy_list]
        self.conf = [conf]
        self.is_fake = is_fake

# ── Drawing Boxes with ID Labels ──────────────────────────────────
def draw_boxes(frame, tracks, fps_text=""):
    out = frame.copy()
    for track in tracks:
        x1, y1, x2, y2, track_id, is_fake = track
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        track_id = int(track_id)

        if is_fake:
            color = (0, 165, 255)  # orange for inferred
            label = f"Inferred ID {track_id}"
        else:
            color = (255, 0, 0)    # blue for real detection
            label = f"ID {track_id}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if fps_text:
        cv2.putText(out, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out

# ── Main Run Function ─────────────────────────────────────────────
def run():
    print("🎬 Opening video:", os.path.abspath(VIDEO_PATH))
    yolo = YOLO(YOLO_MODEL)
    tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.3)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video file: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (PROC_W, PROC_H)
    )
    print(f"✅ {total} frames @ {fps:.1f} FPS → output: {OUTPUT_PATH}")

    buf = deque(maxlen=30)
    prev = time.time()
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1

        resized = cv2.resize(frame, (PROC_W, PROC_H))
        result = yolo.predict(
            source=resized,
            conf=DET_CONF_TH,
            classes=[0],
            verbose=False
        )[0]

        boxes = result.boxes
        processed_boxes = []

        if len(boxes) > 2:
            processed_boxes = sorted(boxes, key=lambda b: b.conf[0].item(), reverse=True)[:2]
        elif len(boxes) == 1:
            b = boxes[0]
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf = b.conf[0].item()
            mid_x = (x1 + x2) // 2
            processed_boxes = [DummyBox([x1, y1, mid_x, y2], conf), DummyBox([mid_x, y1, x2, y2], conf)]
        elif len(boxes) == 0:
            w, h, y = PROC_W // 4, PROC_H // 2, PROC_H // 4
            processed_boxes = [
                DummyBox([PROC_W//4 - w//2, y, PROC_W//4 + w//2, y + h], 0.1),
                DummyBox([3*PROC_W//4 - w//2, y, 3*PROC_W//4 + w//2, y + h], 0.1)
            ]
        else:
            processed_boxes = boxes

        dets = []
        fake_flags = []
        for box in processed_boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            dets.append([x1, y1, x2, y2, conf])
            fake_flags.append(1 if hasattr(box, "is_fake") and box.is_fake else 0)

        dets_np = np.array(dets) if len(dets) > 0 else np.empty((0, 5))
        tracks_np = tracker.update(dets_np)

        tracks = []
        for i, t in enumerate(tracks_np):
            x1, y1, x2, y2, track_id = t
            is_fake = fake_flags[i] if i < len(fake_flags) else 0
            tracks.append([x1, y1, x2, y2, track_id, is_fake])

        now = time.time()
        inst_fps = 1.0 / (now - prev + 1e-6)
        prev = now
        buf.append(inst_fps)
        fps_text = f"FPS: {inst_fps:.1f} (avg {sum(buf)/len(buf):.1f})" if SHOW_FPS else ""

        out = draw_boxes(resized, tracks, fps_text)
        writer.write(out)

        if idx % 50 == 0 or idx == total:
            print(f" … {idx}/{total} frames processed")

    cap.release()
    writer.release()
    print("🎉 Tracked video saved to:", OUTPUT_PATH)

    # Playback
    pv = cv2.VideoCapture(OUTPUT_PATH)
    delay = int(1000 / fps)
    while True:
        ok, f = pv.read()
        if not ok:
            break
        cv2.imshow("YOLO + SORT Tracking with Inferred Boxes", f)
        if cv2.waitKey(delay) & 0xFF == 27:
            break
    pv.release()
    cv2.destroyAllWindows()

# ── Entry Point ────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
