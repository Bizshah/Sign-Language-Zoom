"""Extract MediaPipe landmarks per video -> (frames, 171) float32 .npy files.

Feature layout per frame (171 values, all x,y,z):
  pose  : 7 pts  (nose, L/R shoulder, L/R elbow, L/R wrist)       -> 21
  left hand : 21 pts                                              -> 63
  right hand: 21 pts                                              -> 63
  face  : 8 pts  (nose tip, mouth corners, upper/lower lip,
                  both inner brows, chin)                         -> 24
Coordinates are normalized: origin = shoulder midpoint, scale = shoulder width,
so features don't depend on where the person sits or how far from the camera.
Missing body parts (hand not detected) are zero-filled.

Usage:
  python ml/extract_landmarks.py --signs ml/signs.json --videos data/videos \
      --out data/landmarks --limit 5          # quick test on 5 videos
  python ml/extract_landmarks.py --signs ml/signs.json --videos data/videos \
      --out data/landmarks                    # full run
"""
import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

POSE_IDS = [0, 11, 12, 13, 14, 15, 16]
FACE_IDS = [1, 61, 291, 13, 14, 105, 334, 152]
N_FEATURES = (len(POSE_IDS) + 21 + 21 + len(FACE_IDS)) * 3  # 171


def _pts(landmarks, ids=None):
    """landmark list -> (n,3) array, or None if not detected."""
    if landmarks is None:
        return None
    lm = landmarks.landmark
    ids = ids if ids is not None else range(len(lm))
    return np.array([[lm[i].x, lm[i].y, lm[i].z] for i in ids], dtype=np.float32)


def frame_features(res):
    pose = _pts(res.pose_landmarks, POSE_IDS)
    if pose is None:  # no body -> nothing reliable to normalize against
        return np.zeros(N_FEATURES, dtype=np.float32)

    lh = _pts(res.left_hand_landmarks)
    rh = _pts(res.right_hand_landmarks)
    face = _pts(res.face_landmarks, FACE_IDS)

    # pose[1], pose[2] = left/right shoulder
    origin = (pose[1] + pose[2]) / 2
    scale = np.linalg.norm(pose[1][:2] - pose[2][:2])
    scale = scale if scale > 1e-4 else 1.0

    def norm(a, n):
        return np.zeros((n, 3), np.float32) if a is None else (a - origin) / scale

    parts = [norm(pose, len(POSE_IDS)), norm(lh, 21), norm(rh, 21), norm(face, len(FACE_IDS))]
    return np.concatenate([p.ravel() for p in parts]).astype(np.float32)


def extract_video(path, holistic):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    feats = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        feats.append(frame_features(holistic.process(rgb)))
    cap.release()
    return np.stack(feats) if feats else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signs", default="ml/signs.json")
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default="data/landmarks")
    ap.add_argument("--limit", type=int, default=None, help="process only N videos (testing)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    signs = json.load(open(args.signs))
    jobs = [(g, i["video_id"], i["split"]) for g, items in signs.items() for i in items]
    if args.limit:
        jobs = jobs[: args.limit]

    out_root, video_dir = Path(args.out), Path(args.videos)
    manifest = []

    with mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as holistic:
        for n, (gloss, vid, split) in enumerate(jobs, 1):
            dst = out_root / gloss.replace(" ", "_") / f"{vid}.npy"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and not args.overwrite:
                arr = np.load(dst)
            else:
                arr = extract_video(video_dir / f"{vid}.mp4", holistic)
                holistic.reset()  # don't leak tracking state between videos
                if arr is None:
                    print(f"[{n}/{len(jobs)}] FAILED {vid}")
                    continue
                np.save(dst, arr)

            hand_rate = float((np.abs(arr[:, 21:147]).sum(1) > 0).mean())
            print(f"[{n}/{len(jobs)}] {gloss:<12} {vid}  shape={arr.shape}  hands_seen={hand_rate:.0%}")
            manifest.append({"gloss": gloss, "video_id": vid, "split": split,
                             "path": str(dst), "frames": int(arr.shape[0])})

    json.dump(manifest, open(out_root / "manifest.json", "w"), indent=2)
    print(f"\nSaved {len(manifest)} samples. Feature dim = {N_FEATURES}")


if __name__ == "__main__":
    main()