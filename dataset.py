import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloader(data_dir, batch_size=16, is_train=True):
    """
    data_dir:
        data/train
        data/val
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=2
    )

    return dataloader, dataset.classes
