import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloader
from models.behavior_model import BehaviorModel
from utils import get_device, save_model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def main():
    # ========= 1. 超参数 =========
    num_epochs = 10
    batch_size = 16
    learning_rate = 1e-4
    num_classes = 2

    # ========= 2. 设备 =========
    device = get_device()

    # ========= 3. 数据 =========
    train_loader, class_names = get_dataloader(
        data_dir="data/train",
        batch_size=batch_size,
        is_train=True
    )

    val_loader, _ = get_dataloader(
        data_dir="data/val",
        batch_size=batch_size,
        is_train=False
    )

    print("Classes:", class_names)

    # ========= 4. 模型 =========
    model = BehaviorModel(num_classes=num_classes)
    model.to(device)

    # ========= 5. 损失 & 优化器 =========
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ========= 6. 训练循环 =========
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, "best_model.pth")

    print("Training finished. Best Val Acc:", best_val_acc)


if __name__ == "__main__":
    main()
