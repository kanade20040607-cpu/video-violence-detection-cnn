import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import get_device
from models.behavior_model import BehaviorModel
from predict import predict_video


def list_videos(test_dir):
    """
    读取 test_dir 下两类视频
    test/
      ├── NonViolence/*.mp4   -> normal
      └── Violence/*.mp4      -> abnormal
    """
    nonv_dir = os.path.join(test_dir, "NonViolence")
    vio_dir = os.path.join(test_dir, "Violence")

    nonv_videos = sorted(glob.glob(os.path.join(nonv_dir, "*.mp4")))
    vio_videos = sorted(glob.glob(os.path.join(vio_dir, "*.mp4")))

    samples = []
    for p in nonv_videos:
        samples.append((p, "normal"))
    for p in vio_videos:
        samples.append((p, "abnormal"))

    return samples


def evaluate_at_threshold(model, device, samples, num_frames, threshold):
    """
    用阈值 threshold 做二分类判定：
    prob_abnormal >= threshold -> abnormal
    else -> normal
    """
    TP = FP = TN = FN = 0

    for video_path, true_label in samples:
        # probs: [p_abnormal, p_normal]  (按你 predict.py 的输出习惯)
        _, probs = predict_video(
            model=model,
            device=device,
            video_path=video_path,
            frames=num_frames,
            threshold=threshold,   # 这里传不传都行，最终判定我们自己做
            show=False
        )

        p_abnormal = float(probs[0])
        pred_label = "abnormal" if p_abnormal >= threshold else "normal"

        if true_label == "abnormal" and pred_label == "abnormal":
            TP += 1
        elif true_label == "normal" and pred_label == "abnormal":
            FP += 1
        elif true_label == "normal" and pred_label == "normal":
            TN += 1
        elif true_label == "abnormal" and pred_label == "normal":
            FN += 1

    total = TP + TN + FP + FN
    acc = (TP + TN) / total if total > 0 else 0

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    fp_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
    fn_rate = FN / (FN + TP) if (FN + TP) > 0 else 0

    return {
        "threshold": threshold,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="test", help="测试集目录")
    parser.add_argument("--model_path", type=str, default="best_model.pth", help="模型权重路径")
    parser.add_argument("--frames", type=int, default=15, help="每个视频抽帧数")
    parser.add_argument("--min_t", type=float, default=0.1, help="最小阈值")
    parser.add_argument("--max_t", type=float, default=0.95, help="最大阈值")
    parser.add_argument("--step", type=float, default=0.05, help="阈值步长")
    parser.add_argument("--recall_min", type=float, default=0.95, help="约束：召回率最低要求")
    parser.add_argument("--out_dir", type=str, default="outputs", help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) device
    device = get_device()

    # 2) model
    model = BehaviorModel(num_classes=2)
    model.load_state_dict(
        __import__("torch").load(args.model_path, map_location=device)
    )
    model.to(device)
    model.eval()
    print(f"[OK] Loaded model: {args.model_path}")

    # 3) samples
    samples = list_videos(args.test_dir)
    print(f"[INFO] Total test videos: {len(samples)}")

    # 4) thresholds
    thresholds = np.arange(args.min_t, args.max_t + 1e-9, args.step).round(4).tolist()

    results = []
    best = None

    print("\n========== Threshold Search ==========")
    for t in thresholds:
        r = evaluate_at_threshold(
            model=model,
            device=device,
            samples=samples,
            num_frames=args.frames,
            threshold=t
        )
        results.append(r)

        print(
            f"t={t:.2f} | acc={r['acc']:.4f} "
            f"P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} "
            f"FP_rate={r['fp_rate']:.4f} FN_rate={r['fn_rate']:.4f}"
        )

        # 选择策略：在 Recall>=recall_min 下，F1 最大
        if r["recall"] >= args.recall_min:
            if best is None or r["f1"] > best["f1"]:
                best = r

    print("=====================================\n")

    # 5) best threshold output
    if best is None:
        print(f"[WARN] No threshold meets recall >= {args.recall_min}")
        best = max(results, key=lambda x: x["f1"])
        print("[INFO] fallback: use global best F1")

    print("========== Best Threshold ==========")
    print(f"Best threshold: {best['threshold']:.2f}")
    print(f"Accuracy: {best['acc']:.4f}")
    print(f"Precision(abnormal): {best['precision']:.4f}")
    print(f"Recall(abnormal): {best['recall']:.4f}")
    print(f"F1(abnormal): {best['f1']:.4f}")
    print(f"TP={best['TP']} FP={best['FP']} TN={best['TN']} FN={best['FN']}")
    print("===================================\n")

    # 6) save curves
    th = [x["threshold"] for x in results]
    acc = [x["acc"] for x in results]
    p = [x["precision"] for x in results]
    r = [x["recall"] for x in results]
    f1 = [x["f1"] for x in results]
    fp_rate = [x["fp_rate"] for x in results]
    fn_rate = [x["fn_rate"] for x in results]

    plt.figure(figsize=(10, 6))
    plt.plot(th, acc, label="Accuracy")
    plt.plot(th, p, label="Precision (abnormal)")
    plt.plot(th, r, label="Recall (abnormal)")
    plt.plot(th, f1, label="F1 (abnormal)")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Search Curves")
    plt.legend()
    curve_path = os.path.join(args.out_dir, "threshold_curves.png")
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"[OK] Saved: {curve_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(th, fp_rate, label="FP rate (normal -> abnormal)")
    plt.plot(th, fn_rate, label="FN rate (abnormal -> normal)")
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title("FP/FN Rate vs Threshold")
    plt.legend()
    rate_path = os.path.join(args.out_dir, "threshold_fp_fn.png")
    plt.savefig(rate_path, dpi=200)
    plt.close()
    print(f"[OK] Saved: {rate_path}")

    # 7) save txt summary
    txt_path = os.path.join(args.out_dir, "threshold_best.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Best threshold search result\n")
        f.write(f"best_threshold={best['threshold']}\n")
        f.write(f"accuracy={best['acc']}\n")
        f.write(f"precision_abnormal={best['precision']}\n")
        f.write(f"recall_abnormal={best['recall']}\n")
        f.write(f"f1_abnormal={best['f1']}\n")
        f.write(f"TP={best['TP']} FP={best['FP']} TN={best['TN']} FN={best['FN']}\n")
    print(f"[OK] Saved: {txt_path}")


if __name__ == "__main__":
    main()
