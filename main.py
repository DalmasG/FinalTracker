import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import warnings
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from sort import Sort
from filterpy.kalman import KalmanFilter

#imports for classification
import torch
from model import PositionClassifier
#load class names and instantiate classifier
DATA = np.load("dataset_grappling/processed/features_labels.npz", allow_pickle=True)
CLASSES = DATA["classes"].tolist()
clf = PositionClassifier(num_classes=len(CLASSES))
clf.load_state_dict(torch.load("checkpoints/position_classifier.pt", map_location="cpu"))
clf.eval()

#CONFIG
VIDEO_PATH    = os.path.join("footage", "test_1.mp4")
OUTPUT_PATH   = "test_1_yolopose_tracked_kf_classified_long.mp4"
YOLO_MODEL    = "yolov8x-pose.pt"
DET_CONF_TH   = 0.1
POSE_CONF_TH  = 0.3
PROC_W, PROC_H = 1280, 1280

warnings.filterwarnings("ignore", category=UserWarning)
LOGGER.setLevel("ERROR")

#COCO skeleton pose
POSE_CONNECTIONS = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (11,13),(13,15),(12,14),(14,16),(11,12),
    (5,11),(6,12),
]
COLOR_BY_ID = [(255,0,0),(0,165,255),(0,255,0),(255,255,0)]

def make_2d_kf():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
    kf.H = np.array([[1,0,0,0],[0,1,0,0]])
    kf.R *= 5.
    kf.P *= 1000.
    kf.Q  = np.eye(4)*0.01
    return kf

#helpers to normalize poses
PAD = 5
def compute_bbox(pts):
    xs = pts[:,0]; ys = pts[:,1]
    xmin, xmax = xs.min()-PAD, xs.max()+PAD
    ymin, ymax = ys.min()-PAD, ys.max()+PAD
    return xmin, ymin, xmax, ymax

def normalize_pose(pts, bbox):
    xmin,ymin,xmax,ymax = bbox
    w = xmax - xmin + 1e-6
    h = ymax - ymin + 1e-6
    flat = []
    for x,y,c in pts:
        nx = (x - xmin) / w
        ny = (y - ymin) / h
        flat.extend([nx, ny, c])
    return flat

def draw_pose(frame, tracks):
    out = frame.copy()
    for x1,y1,x2,y2,tid,kps in tracks:
        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
        c = COLOR_BY_ID[int(tid) % len(COLOR_BY_ID)]
        cv2.rectangle(out, (x1,y1), (x2,y2), c, 2)
        cv2.putText(out, f"ID{int(tid)}", (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        if isinstance(kps, np.ndarray) and kps.shape[1]==3:
            for x,y,conf in kps:
                if conf >= POSE_CONF_TH and x1 < x < x2 and y1 < y < y2:
                    cv2.circle(out, (int(x),int(y)), 3, c, -1)
            for a,b in POSE_CONNECTIONS:
                xa,ya,ca = kps[a]; xb,yb,cb = kps[b]
                if (ca>=POSE_CONF_TH and cb>=POSE_CONF_TH and
                    x1<xa<x2 and y1<ya<y2 and
                    x1<xb<x2 and y1<yb<y2):
                    cv2.line(out, (int(xa),int(ya)), (int(xb),int(yb)), c, 2)
    return out

def run():
    print("▶ Loading YOLOv8-Pose model…")
    model = YOLO(YOLO_MODEL)
    tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.3)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Could not open {VIDEO_PATH}")
        return

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (PROC_W, PROC_H)
    )
    print(f"✅ {total} frames @ {fps:.1f}FPS → {OUTPUT_PATH}")

    idx = 0
    kf_store = {}  #tid → list of 17 KalmanFilters

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1

        im = cv2.resize(frame, (PROC_W, PROC_H))
        res = model.predict(source=im, conf=DET_CONF_TH, verbose=False)[0]

        #extract detections + keypoints
        dets, kps_list = [], []
        if res.boxes is not None and res.keypoints is not None:
            boxes  = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            allkps = res.keypoints.data.cpu().numpy()
            for b,sc,kp in zip(boxes, scores, allkps):
                dets.append([*b, float(sc)])
                kps_list.append(kp)

        #enforce two athletes
        if len(dets)>2:
            dets, kps_list = dets[:2], kps_list[:2]
        elif len(dets)==1:
            x1,y1,x2,y2,sc = dets[0]
            mx = (x1+x2)/2
            dets    = [[x1,y1,mx,y2,sc], [mx,y1,x2,y2,sc]]
            kps_list = [kps_list[0], None]
        elif not dets:
            w,h = PROC_W//4, PROC_H//2; y = PROC_H//4
            dets = [
              [PROC_W//4-w//2, y, PROC_W//4+w//2, y+h, 0.1],
              [3*PROC_W//4-w//2, y, 3*PROC_W//4+w//2, y+h, 0.1]
            ]
            kps_list = [None, None]

        dets_np = np.array(dets)
        tracks_np = tracker.update(dets_np)

        #smooth poses per-track with KF
        tracks = []
        for x1,y1,x2,y2,tid in tracks_np:
            best_iou, best_j = 0, -1
            for j,d in enumerate(dets):
                xi1,yi1,xi2,yi2,_ = d
                ia = max(0, min(x2,xi2)-max(x1,xi1)) * max(0, min(y2,yi2)-max(y1,yi1))
                ua = (x2-x1)*(y2-y1) + (xi2-xi1)*(yi2-yi1) - ia + 1e-6
                iou = ia/ua
                if iou > best_iou:
                    best_iou, best_j = iou, j

            kps = kps_list[best_j]
            if isinstance(kps, np.ndarray):
                if tid not in kf_store:
                    kf_store[tid] = [make_2d_kf() for _ in range(kps.shape[0])]
                sm = []
                for i,(x,y,c) in enumerate(kps):
                    kf = kf_store[tid][i]
                    kf.predict()
                    if c>=POSE_CONF_TH and x1<x<x2 and y1<y<y2:
                        kf.x[:2,0] = np.array([x,y])
                        kf.update(np.array([x,y]))
                    sm.append([float(kf.x[0,0]), float(kf.x[1,0]), float(c)])
                kps = np.array(sm, dtype=np.float32)
            tracks.append([x1,y1,x2,y2,tid,kps])

        #classify frame if both poses exist
        label = None
        if len(tracks)==2 and all(isinstance(t[5], np.ndarray) for t in tracks):
            tracks_sorted = sorted(tracks, key=lambda x: x[4])
            kps1 = tracks_sorted[0][5]
            kps2 = tracks_sorted[1][5]
            vec1 = normalize_pose(kps1, compute_bbox(kps1))
            vec2 = normalize_pose(kps2, compute_bbox(kps2))
            feat = np.array(vec1 + vec2, dtype=np.float32)
            t = torch.from_numpy(feat).unsqueeze(0)
            with torch.no_grad():
                logits = clf(t).cpu().numpy()[0]
            pred = int(np.argmax(logits))
            label = CLASSES[pred]

        out = draw_pose(im, tracks)

        #overlay classification label
        if label is not None:
            cv2.putText(
                out,
                f"Position: {label}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0,255,255),  #yellow text 
                2
            )

        writer.write(out)
        if idx%50==0 or idx==total:
            print(f" … {idx}/{total} frames")

    cap.release()
    writer.release()
    print("🎉 Saved →", OUTPUT_PATH)

    #playback
    pv = cv2.VideoCapture(OUTPUT_PATH)
    delay = int(1000/fps)
    while True:
        ok, f = pv.read()
        if not ok: break
        cv2.imshow("Pose+Position Tracking", f)
        if cv2.waitKey(delay) & 0xFF == 27:
            break
    pv.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    run()
