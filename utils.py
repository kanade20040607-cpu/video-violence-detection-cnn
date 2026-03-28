import torch
import os


def get_device():
    """
    自动选择运行设备（GPU / CPU）
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def save_model(model, path):
    """
    保存模型权重
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path, device):
    """
    加载模型权重
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model


def accuracy(outputs, labels):
    """
    计算分类准确率
    """
    with torch.no_grad():
        _, preds = torch.max(outputs, 1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)
    return correct / total
