import os
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.behavior_model import BehaviorModel
from utils import get_device, load_model
from predict import predict_video  # 复用你 predict.py 的函数


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[OK] Confusion matrix saved: {save_path}")


def classification_report_from_cm(cm, class_names):
    # cm: [[TP_abnormal??]] 这里按顺序 [abnormal, normal]
    report_lines = []
    report_lines.append("Classification Report\n")

    total = cm.sum()
    acc = (cm[0, 0] + cm[1, 1]) / total if total > 0 else 0

    for idx, cls in enumerate(class_names):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        report_lines.append(
            f"{cls:10s}  precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}"
        )

    report_lines.append(f"\nAccuracy: {acc:.4f}")
    return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="test",
                        help="test directory containing Violence and NonViolence")
    parser.add_argument("--weights", type=str, default="best_model.pth")
    parser.add_argument("--frames", type=int, default=15,
                        help="sampled frames per video")
    parser.add_argument("--threshold", type=float, default=None,
                        help="optional threshold for abnormal probability")
    args = parser.parse_args()

    # 你训练时的类别顺序：['abnormal', 'normal']
    class_names = ["abnormal", "normal"]

    # test 目录结构：Violence / NonViolence
    violence_dir = os.path.join(args.test_dir, "Violence")
    nonviolence_dir = os.path.join(args.test_dir, "NonViolence")

    if not os.path.exists(violence_dir) or not os.path.exists(nonviolence_dir):
        raise FileNotFoundError(
            f"test_dir should contain Violence and NonViolence folders.\n"
            f"Found: {violence_dir}, {nonviolence_dir}"
        )

    device = get_device()

    model = BehaviorModel(num_classes=2)
    model = load_model(model, args.weights, device)

    # 收集所有测试视频
    test_samples = []
    for fname in os.listdir(violence_dir):
        if fname.lower().endswith(".mp4"):
            test_samples.append((os.path.join(violence_dir, fname), 0))  # 0=abnormal
    for fname in os.listdir(nonviolence_dir):
        if fname.lower().endswith(".mp4"):
            test_samples.append((os.path.join(nonviolence_dir, fname), 1))  # 1=normal

    test_samples.sort(key=lambda x: x[0])
    print(f"[INFO] Total test videos: {len(test_samples)}")

    # confusion matrix: rows true, cols pred
    cm = np.zeros((2, 2), dtype=int)
    wrong_list = []

    for video_path, true_label in test_samples:
        pred_label, probs = predict_video(
            model=model,
            video_path=video_path,
            device=device,
            num_frames=args.frames,
            return_probs=True
        )

        # predict_video 返回的是 class index：0/1
        pred_idx = pred_label

        # 可选：阈值判决（降低误报）
        if args.threshold is not None:
            abnormal_prob = float(probs[0])
            if abnormal_prob >= args.threshold:
                pred_idx = 0
            else:
                pred_idx = 1

        cm[true_label, pred_idx] += 1

        if pred_idx != true_label:
            wrong_list.append({
                "video": video_path,
                "true": class_names[true_label],
                "pred": class_names[pred_idx],
                "prob_abnormal": float(probs[0]),
                "prob_normal": float(probs[1]),
            })

    # 输出指标
    total = cm.sum()
    acc = (cm[0, 0] + cm[1, 1]) / total if total > 0 else 0

    print("\n========== Evaluation Summary ==========")
    print("Confusion Matrix:\n", cm)
    print(f"Total videos: {total}")
    print(f"Accuracy: {acc:.4f}")
    print("========================================\n")

    # 保存输出
    os.makedirs("outputs", exist_ok=True)

    plot_confusion_matrix(cm, class_names, "outputs/confusion_matrix_test.png")

    report = classification_report_from_cm(cm, class_names)
    with open("outputs/classification_report_test.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("[OK] Classification report saved: outputs/classification_report_test.txt")

    with open("outputs/wrong_samples_test.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "true", "pred", "prob_abnormal", "prob_normal"])
        writer.writeheader()
        writer.writerows(wrong_list)
    print("[OK] Wrong samples saved: outputs/wrong_samples_test.csv")
    print(f"[INFO] Wrong count: {len(wrong_list)}")


if __name__ == "__main__":
    main()
