#!/usr/bin/env python3
"""
JSL5級動画 → 学習データJSON 変換スクリプト

字幕が変わるタイミング = 単語の区切りとして使い、
その間のフレームをまるごと1サンプルとして保存します。

pip3 install opencv-python mediapipe numpy
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import json, datetime, sys, os

# ── 設定 ───────────────────────────────────────
BASE_DIR   = "/Users/murakamimiki/Desktop/JSL_app"
VIDEO_PATH = "/Users/murakamimiki/Downloads/全国手話検定試験　５級　手話を覚えてみよう！.mp4"
OUTPUT     = f"{BASE_DIR}/training_from_video.json"

HAND_MODEL = f"{BASE_DIR}/hand_landmarker.task"
POSE_MODEL = f"{BASE_DIR}/pose_landmarker.task"
FACE_MODEL = f"{BASE_DIR}/face_landmarker.task"

# 字幕エリア（左上）
SUB_T, SUB_B = 0.0,  0.14
SUB_L, SUB_R = 0.0,  0.40
SUB_THRESH   = 8.0   # 変化感度
SUB_STABLE   = 8     # 安定フレーム数

SKIP = 2  # MediaPipe を何フレームおきに実行するか

# ── JS と完全一致の定数・関数 ──────────────────
POSE_INDICES     = [11,12,13,14,15,16,19,20]
FACE_KEY_INDICES = [1, 4, 33, 263, 13, 14]

def r3(v): return round(v*1000)/1000

def norm_hand(lm):
    if not lm or len(lm)<10: return [[0.0,0.0]]*21
    w = lm[0]
    ref = ((lm[9][0]-w[0])**2+(lm[9][1]-w[1])**2)**0.5 or 1.0
    return [[r3((p[0]-w[0])/ref), r3((p[1]-w[1])/ref)] for p in lm]

def build_feature_vector(lh, rh, pl, fl):
    v = [x for pair in norm_hand(lh)+norm_hand(rh) for x in pair]

    if pl and len(pl)>16:
        ls,rs = pl[11],pl[12]
        cx=(ls[0]+rs[0])/2; cy=(ls[1]+rs[1])/2
        sc=((ls[0]-rs[0])**2+(ls[1]-rs[1])**2)**0.5 or 1.0
        for i in POSE_INDICES:
            p = pl[i] if i<len(pl) else None
            v += [r3((p[0]-cx)/sc), r3((p[1]-cy)/sc)] if p else [0.0,0.0]
    else:
        v += [0.0]*16

    if fl and len(fl)>263:
        nose = fl[1]
        ex = fl[33][0]-fl[263][0]; ey = fl[33][1]-fl[263][1]
        ed = (ex**2+ey**2)**0.5 or 1.0
        for i in FACE_KEY_INDICES:
            p = fl[i] if i<len(fl) else None
            v += [r3((p[0]-nose[0])/ed), r3((p[1]-nose[1])/ed)] if p else [0.0,0.0]
        v += [r3(abs(fl[14][1]-fl[13][1])/ed)] if len(fl)>14 else [0.0]
    else:
        v += [0.0]*13

    return v

def get_subtitle_img(frame):
    h, w = frame.shape[:2]
    crop = frame[int(h*SUB_T):int(h*SUB_B), int(w*SUB_L):int(w*SUB_R)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (160, 20))

# ── 5級単語リスト ──────────────────────────────
WORD_LIST = [
    "愛","挨拶","間","会う","合う","青","赤","赤ちゃん","秋","開ける",
    "あげる","朝","浅い","明日","遊ぶ","新しい","あなた","兄","姉","雨",
    "謝る","ありがとう","有る","歩く","安心","井","言うA","言うB","家","行くA",
    "行くB","いくつ","池","囲碁","石A","石B","医者","椅子","忙しいA","忙しいB",
    "位置","いつ","一致","一緒","犬","今","妹","居る","色","いろいろ",
    "岩","上A","上B","受ける","兎","歌","歌う","内","うどん","生まれる",
    "海","うらやましい","売る","映画","選ぶ","円","美味しいA","美味しいB","多い","大きい",
    "オートバイ","おかしい","お金","怒るA","怒るB","おしゃべり","遅い","教わる","夫","弟",
    "男","おとな","驚く","同じ","おにぎり","おはようございます","覚える","おめでとう","思う","面白い",
    "おやすみなさい","おやつ","終わる","音楽","女",
    "会","会社","会話","買う","帰るA","帰るB","書く","数","家族","学校",
    "悲しい","かまわない","紙","カメラ","通う","から","川","考え","簡単","木",
    "黄","聞く","聞こえないA","聞こえないB","北","きのう","休憩","今日","教育","競技",
    "兄弟","嫌い","きれい","金","草","曇り","くらい（程度）","暗い","暮らし","来るA",
    "来るB","苦しい","車いす","黒","携帯電話","ケーキ","ゲートボール","結婚","月曜日","けれども",
    "けんかA","けんかB","元気","健聴","検定",
    "剣道","高校","紅茶","交流","口話","声","コーヒー","国語","ご苦労さま","午後",
    "午前","答え","子どもA","子どもB","困る","ゴルフ","今度","こんにちは","こんばんは","サークル",
    "最初","坂","魚","酒","サッカー","寂しい","さようなら","残念","時","時間",
    "式","試験","自己紹介","次女","下A","下B","した","肢体不自由","質問","自転車",
    "自動車","次男","島","姉妹","事務","社会","ジュース","集会","柔道","主婦",
    "趣味","手話","手話通訳","小","障害者","小学校","将棋","上下A","上下B","正午",
    "上手","昭和","職業","親戚","心配","新聞","水泳","スキー","スケート","少し",
    "スポーツ","すみません","相撲","する","生徒","世話","先生","洗濯","全部","双生児",
    "祖父","ソフトボール","祖母",
    "田","体育","大学","大工","対象","大丈夫","大切","太陽","高い（金額）","高い（高さ）",
    "助ける","立場","建つ","卓球","谷","楽しい","頼む","食べる","駄目","誰",
    "小さい","近い","違う","力","父","茶A","茶B","中学校","聴覚","長女",
    "長男","通学","通訳","使う","疲れる","次","作る","土","妻","強い",
    "釣り","テーマ","テニス","出る","テレビ","店員","点字","電車","電話","どう",
    "動物","遠い","得意","独身","どこ","どちら","とても","隣り（右）","隣り（左）","友達",
    "鳥",
    "無い","中","長い","なかなか","仲間","仲良し","泣く","夏","何か","名前A",
    "名前B","なるほど","難聴","苦手","西","日曜日A","日曜日B","庭","猫","鼠",
    "寝る","年","年齢","飲む",
    "入る","白杖","橋","はじめまして","場所","走るA","走るB","バス","恥ずかしい","バスケットボール",
    "パソコン","バトミントン","花","バナナ","母","浜","早い","林","原","春",
    "晴れ","バレーボール","番号","日","火","ピアノ","ビール","東","低い","左の方",
    "筆談","人","人々","表現","表情","ファクシミリ","夫婦","深い","冬","古い",
    "文","平成","下手","便所A","便所B","保育所","方向","方法","星","欲しい",
    "ポスター","補聴器","殆ど","本",
    "孫","まずい","また","まだまだ","町","待つ","松","まで","真似る","マラソン",
    "蜜柑","右の方","短い","水A","水B","店","道","身振り","見る","ミルク",
    "みんな","虫","難しい","息子","娘","村","メール","眼鏡",
    "盲","もう一度","持つ","貰う","森","問題","野球A","野球B","安い","休み",
    "山","山登り","郵便","ゆっくり","指文字","良い","幼稚園","読む","よろしく","弱い",
    "ラーメン","ラジオ","理科","離婚","両親","料理","りんご","列","練習","連絡",
    "聾唖A","聾唖B","老人","若い","分からない","分かる","別れる","忘れる","わたしA","わたしB",
    "わたしC","笑う","悪い",
]

# ── メイン ────────────────────────────────────
def process_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"動画: {fps:.1f}fps × {total}フレーム ({total/fps/60:.1f}分)")
    print(f"5級単語: {len(WORD_LIST)}語")
    print("モデル読み込み中...\n")

    RunMode = mp_vision.RunningMode
    hand_det = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=RunMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.3, min_tracking_confidence=0.3))
    pose_det = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL),
            running_mode=RunMode.VIDEO,
            min_pose_detection_confidence=0.3, min_tracking_confidence=0.3))
    face_det = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL),
            running_mode=RunMode.VIDEO,
            min_face_detection_confidence=0.3, min_tracking_confidence=0.3))

    print("処理開始！(Ctrl+C で中断→途中保存)\n")

    training  = {}
    word_idx  = 0
    frame_idx = 0
    prev_sub  = None
    sub_change_flag = False
    stable_cnt = 0
    word_buf  = []   # 現在単語のフレームバッファ
    mp_frame_idx = 0  # MediaPipeに渡す連番（スキップしても単調増加が必要）

    try:
        while word_idx < len(WORD_LIST):
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            ts_ms = int(frame_idx * 1000 / fps)

            # ── 字幕変化検出（毎フレーム） ──
            curr_sub = get_subtitle_img(frame)
            if prev_sub is not None:
                diff = np.abs(curr_sub.astype(float)-prev_sub.astype(float)).mean()
                if diff > SUB_THRESH:
                    sub_change_flag = True
                    stable_cnt = 0
                elif sub_change_flag:
                    stable_cnt += 1
                    if stable_cnt >= SUB_STABLE:
                        # 字幕が安定 → 前の単語を保存して次へ
                        _save_word(training, word_idx, word_buf)
                        word_idx += 1
                        word_buf = []
                        sub_change_flag = False
                        stable_cnt = 0
                        if word_idx < len(WORD_LIST):
                            print(f"\n  → [{word_idx+1}]「{WORD_LIST[word_idx]}」")
            prev_sub = curr_sub

            if word_idx >= len(WORD_LIST): break

            # ── MediaPipe（フレームスキップ） ──
            if frame_idx % SKIP != 0:
                continue

            mp_frame_idx += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            try:
                hr = hand_det.detect_for_video(mp_img, mp_frame_idx)
                pr = pose_det.detect_for_video(mp_img, mp_frame_idx)
                fr = face_det.detect_for_video(mp_img, mp_frame_idx)
            except Exception:
                continue

            lh = rh = None
            for j, h in enumerate(hr.handedness):
                lms = [(lm.x,lm.y) for lm in hr.hand_landmarks[j]]
                if h[0].category_name == "Left": lh = lms
                else: rh = lms

            pl = [(lm.x,lm.y) for lm in pr.pose_landmarks[0]] if pr.pose_landmarks else None
            fl = [(lm.x,lm.y) for lm in fr.face_landmarks[0]]  if fr.face_landmarks  else None

            vec = build_feature_vector(lh, rh, pl, fl)

            # ポーズが検出できたフレームのみバッファに追加
            if pl is not None:
                word_buf.append(vec)

            if frame_idx % 150 == 0:
                pct  = frame_idx/total*100
                word = WORD_LIST[word_idx]
                tots = sum(len(v) for v in training.values())
                print(f"  {pct:.1f}% | [{word_idx+1}]「{word}」buf={len(word_buf)}f | 保存済:{tots}サンプル   ", end='\r')

    except KeyboardInterrupt:
        print("\n\n中断。途中保存します...")
        _save_word(training, word_idx, word_buf)

    cap.release()
    hand_det.close(); pose_det.close(); face_det.close()

    tots = sum(len(v) for v in training.values())
    print(f"\n完了: {len(training)}単語 / {tots}サンプル")

    out = {
        "version": 4, "type": "jsl-holistic-dtw", "featureDim": 111,
        "timestamp": datetime.datetime.now().isoformat(),
        "totalSamples": tots,
        "wordCounts": {w: len(s) for w,s in training.items()},
        "data": training,
    }
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"保存: {OUTPUT}")
    print("\n── インポート方法 ──────────────────────")
    print("1. JSL起動.command をダブルクリック")
    print("2. 学習モード → 「JSONを読み込む」")
    print("3. training_from_video.json を選択")

def _save_word(training, word_idx, buf):
    if word_idx >= len(WORD_LIST) or len(buf) < 6:
        return
    word = WORD_LIST[word_idx]
    if word not in training: training[word] = []
    training[word].append(buf[:])
    cnt = len(training[word])
    print(f"\n    ✅「{word}」{len(buf)}フレーム → {cnt}サンプル")

if __name__ == '__main__':
    for p in [HAND_MODEL, POSE_MODEL, FACE_MODEL]:
        if not os.path.exists(p):
            print(f"❌ モデルなし: {p}"); sys.exit(1)
    process_video()
