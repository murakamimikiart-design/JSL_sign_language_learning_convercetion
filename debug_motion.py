#!/usr/bin/env python3
"""動作検出デバッグ: フレーム500〜2000を処理して速度と状態を表示"""
import cv2, mediapipe as mp, numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

BASE  = "/Users/murakamimiki/Desktop/JSL_app"
VIDEO = "/Users/murakamimiki/Downloads/全国手話検定試験　５級　手話を覚えてみよう！.mp4"
POSE_INDICES = [11,12,13,14,15,16,19,20]

def r3(v): return round(v*1000)/1000
def norm_hand(lm):
    if not lm or len(lm)<10: return [[0.0,0.0]]*21
    w=lm[0]; ref=((lm[9][0]-w[0])**2+(lm[9][1]-w[1])**2)**0.5 or 1
    return [[r3((p[0]-w[0])/ref),r3((p[1]-w[1])/ref)] for p in lm]

def build_vec(lh, rh, pl):
    v = []
    for pair in norm_hand(lh)+norm_hand(rh): v += pair
    if pl and len(pl)>16:
        ls,rs = pl[11],pl[12]
        cx=(ls[0]+rs[0])/2; cy=(ls[1]+rs[1])/2
        sc=((ls[0]-rs[0])**2+(ls[1]-rs[1])**2)**0.5 or 1
        for i in POSE_INDICES:
            p = pl[i] if i<len(pl) else None
            v += [r3((p[0]-cx)/sc),r3((p[1]-cy)/sc)] if p else [0,0]
    else:
        v += [0]*16
    v += [0]*13
    return v

print("モデル読み込み中...")
hand_det = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=f"{BASE}/hand_landmarker.task"),
        running_mode=mp_vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.3, min_tracking_confidence=0.3
    ))
pose_det = mp_vision.PoseLandmarker.create_from_options(
    mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=f"{BASE}/pose_landmarker.task"),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.3, min_tracking_confidence=0.3
    ))

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
START = 500
cap.set(cv2.CAP_PROP_POS_FRAMES, START)

prev = None
state = 'idle'; mc = 0; sc_ = 0; buf = []
THRESH = 0.003
episodes = 0

print(f"フレーム{START}〜から1500フレーム処理\n")
print("フレーム | 速度    | 状態   | 手 | ポーズ")
print("-"*50)

for i in range(1500):
    ret, frame = cap.read()
    if not ret: break
    fidx = START + i
    ts = int(fidx * 1000 / fps)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    hr = hand_det.detect_for_video(mp_img, ts)
    pr = pose_det.detect_for_video(mp_img, ts)

    lh = rh = None
    for j, h in enumerate(hr.handedness):
        lms = [(lm.x,lm.y) for lm in hr.hand_landmarks[j]]
        if h[0].category_name == "Left": lh = lms
        else: rh = lms

    pl = [(lm.x,lm.y) for lm in pr.pose_landmarks[0]] if pr.pose_landmarks else None
    vec = build_vec(lh, rh, pl)

    vel = 0
    if prev:
        start_d, end_d = 84, min(100, len(vec), len(prev))
        n = end_d - start_d
        if n > 0:
            vel = (sum((vec[k]-prev[k])**2 for k in range(start_d, end_d)) / n) ** 0.5

    if state == 'idle':
        if vel > THRESH: mc += 1
        else: mc = 0
        if mc >= 3: state = 'active'; buf = []; sc_ = 0
    elif state == 'active':
        buf.append(vec)
        if vel < THRESH: sc_ += 1
        else: sc_ = 0
        if sc_ >= 20:
            n = len(buf)
            if 6 <= n <= 150:
                episodes += 1
                print(f"\n  ✅ エピソード{episodes}: {n}フレーム (フレーム{fidx-n}〜{fidx})\n")
            buf = []; state = 'idle'; mc = 0

    if i % 30 == 0:
        print(f"  {fidx:6d} | {vel:.5f} | {state:6s} | {len(hr.hand_landmarks)} | {'○' if pl else '×'}")

    prev = vec

cap.release(); hand_det.close(); pose_det.close()
print(f"\n合計 {episodes} エピソード検出")
print(f"速度閾値: {THRESH} → 閾値を下げる必要がある場合は THRESH を小さくしてください")
