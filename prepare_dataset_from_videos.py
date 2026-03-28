import os
import cv2
import random
import shutil
from pathlib import Path

# ==========================
# 你只需要改这里
# ==========================
RAW_ROOT = r"G:\cnn\Real Life Violence Dataset"  # 数据集根目录
OUT_ROOT = r"G:\cnn\data"                       # 输出到你的项目 data/
VAL_RATIO = 0.2                                 # 20% 验证集
FRAME_EVERY_N = 10                              # 每隔N帧抽1帧（10=抽得多，20=抽得少）
MAX_FRAMES_PER_VIDEO = 80                       # 每个视频最多抽多少帧（防止爆炸）
IMG_SIZE = 224                                  # 输出图片尺寸（224适配ResNet）


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def clear_dir(p):
    if os.path.exists(p):
        shutil.rmtree(p)
    os.makedirs(p, exist_ok=True)


def extract_frames(video_path, save_dir, prefix):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[跳过] 无法打开: {video_path}")
        return 0

    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔 N 帧抽一帧
        if count % FRAME_EVERY_N == 0:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            out_path = os.path.join(save_dir, f"{prefix}_{saved:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1

            if saved >= MAX_FRAMES_PER_VIDEO:
                break

        count += 1

    cap.release()
    return saved


def split_videos(video_list, val_ratio=0.2):
    random.shuffle(video_list)
    n_val = int(len(video_list) * val_ratio)
    val_videos = video_list[:n_val]
    train_videos = video_list[n_val:]
    return train_videos, val_videos


def main():
    random.seed(42)

    violence_dir = os.path.join(RAW_ROOT, "Violence")
    nonviolence_dir = os.path.join(RAW_ROOT, "NonViolence")

    violence_videos = sorted(list(Path(violence_dir).glob("*.mp4")))
    nonviolence_videos = sorted(list(Path(nonviolence_dir).glob("*.mp4")))

    print("Violence videos:", len(violence_videos))
    print("NonViolence videos:", len(nonviolence_videos))

    # 输出目录（严格匹配你的项目结构）
    train_normal = os.path.join(OUT_ROOT, "train", "normal")
    train_abnormal = os.path.join(OUT_ROOT, "train", "abnormal")
    val_normal = os.path.join(OUT_ROOT, "val", "normal")
    val_abnormal = os.path.join(OUT_ROOT, "val", "abnormal")

    # 清空并重建目录（避免重复生成）
    clear_dir(os.path.join(OUT_ROOT, "train"))
    clear_dir(os.path.join(OUT_ROOT, "val"))
    ensure_dir(train_normal)
    ensure_dir(train_abnormal)
    ensure_dir(val_normal)
    ensure_dir(val_abnormal)

    # 划分 train / val（按视频划分，防止同一视频帧泄漏到train和val）
    v_train, v_val = split_videos(violence_videos, VAL_RATIO)
    nv_train, nv_val = split_videos(nonviolence_videos, VAL_RATIO)

    # ========== 抽帧 ==========
    total = 0

    print("\n[1/4] 抽取 train/abnormal (Violence)")
    for vp in v_train:
        saved = extract_frames(vp, train_abnormal, prefix=vp.stem)
        total += saved

    print("\n[2/4] 抽取 val/abnormal (Violence)")
    for vp in v_val:
        saved = extract_frames(vp, val_abnormal, prefix=vp.stem)
        total += saved

    print("\n[3/4] 抽取 train/normal (NonViolence)")
    for vp in nv_train:
        saved = extract_frames(vp, train_normal, prefix=vp.stem)
        total += saved

    print("\n[4/4] 抽取 val/normal (NonViolence)")
    for vp in nv_val:
        saved = extract_frames(vp, val_normal, prefix=vp.stem)
        total += saved

    print("\n完成，总共生成帧图片:", total)
    print("输出目录:", OUT_ROOT)
    print("\n目录结构：")
    print("data/train/normal")
    print("data/train/abnormal")
    print("data/val/normal")
    print("data/val/abnormal")


if __name__ == "__main__":
    main()
