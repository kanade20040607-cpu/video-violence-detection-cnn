import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from models.behavior_model import BehaviorModel
from utils import get_device, load_model


# 和训练时一致：224 + normalize
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def sample_frames(video_path, num_frames=15):
    """
    从视频中均匀采样 num_frames 帧
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has 0 frames: {video_path}")

    # 均匀采样索引
    idxs = np.linspace(0, total_frames - 1, num_frames).astype(int)

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames


def preprocess_frames(frames):
    """
    frames: list[np.ndarray RGB]
    return: tensor [N, 3, 224, 224]
    """
    imgs = []
    for f in frames:
        pil = Image.fromarray(f)
        imgs.append(_transform(pil))
    return torch.stack(imgs, dim=0)


@torch.no_grad()
def predict_video(
    model,
    video_path,
    device,
    num_frames=15,
    return_probs=False,
    show=False
):
    """
    预测一个视频属于 abnormal / normal

    返回：
      - 默认：pred_idx
      - return_probs=True：pred_idx, probs(np.array shape [2])
    """
    frames = sample_frames(video_path, num_frames=num_frames)

    if len(frames) == 0:
        raise RuntimeError(f"No frames sampled from: {video_path}")

    # 可视化（可选）
    if show:
        for f in frames[:min(10, len(frames))]:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            cv2.imshow("sampled frame", bgr)
            if cv2.waitKey(120) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

    x = preprocess_frames(frames).to(device)  # [N, 3, 224, 224]
    outputs = model(x)  # [N, 2]

    # 帧级 softmax -> 视频级平均
    probs = F.softmax(outputs, dim=1)  # [N, 2]
    video_probs = probs.mean(dim=0)    # [2]

    pred_idx = int(torch.argmax(video_probs).item())

    if return_probs:
        return pred_idx, video_probs.detach().cpu().numpy()
    return pred_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="video path")
    parser.add_argument("--weights", type=str, default="best_model.pth")
    parser.add_argument("--frames", type=int, default=15)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    device = get_device()

    model = BehaviorModel(num_classes=2)
    model = load_model(model, args.weights, device)

    class_names = ["abnormal", "normal"]

    pred_idx, probs = predict_video(
        model=model,
        video_path=args.video,
        device=device,
        num_frames=args.frames,
        return_probs=True,
        show=args.show
    )

    confidence = float(np.max(probs))

    print("\n========== Predict Result ==========")
    print(f"Video: {args.video}")
    print(f"Sampled Frames: {args.frames}")
    print(f"Prediction: {class_names[pred_idx]}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Probabilities: {probs.tolist()}")
    print("===================================\n")


if __name__ == "__main__":
    main()
