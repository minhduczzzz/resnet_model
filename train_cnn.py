import os
import shutil
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import (
    CenterCrop, Compose, RandomResizedCrop, RandomHorizontalFlip,
    RandomRotation, ColorJitter, ToTensor, Normalize, RandomErasing, Resize
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import DogBreedTrainValDataset
from model import DogBreedResNet


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 50
    batch_size = 32
    patience = 10

    labels_path = "labels.csv"
    train_dir = "data/train"


    # 🔥 Strong augmentation
    train_transform = Compose([
        RandomResizedCrop(224, scale=(0.6, 1.0)),
        RandomHorizontalFlip(),
        RandomRotation(15),
        ColorJitter(0.3, 0.3, 0.3),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406],
                  [0.229, 0.224, 0.225]),
        RandomErasing(p=0.25)
    ])

    val_transform = Compose([
        Resize(256), 
        CenterCrop(224),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])

    df = pd.read_csv(labels_path)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["breed"]
    )

    train_dataset = DogBreedTrainValDataset(
        image_dir=train_dir,
        dataframe=train_df,
        transform=train_transform
    )

    val_dataset = DogBreedTrainValDataset(
        image_dir=train_dir,
        dataframe=val_df,
        transform=val_transform,
        class_to_idx=train_dataset.class_to_idx
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Reset tensorboard
    if os.path.isdir("tensorboard"):
        shutil.rmtree("tensorboard")

    os.makedirs("training_models", exist_ok=True)

    writer = SummaryWriter("tensorboard")

    num_classes = len(train_dataset.class_to_idx)

    # 🔥 Freeze backbone initially
    model = DogBreedResNet(num_classes=num_classes, pretrained=True, freeze=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = Adam([
        {"params": model.backbone.fc.parameters(), "lr": 1e-3},
        {"params": model.backbone.layer4.parameters(), "lr": 1e-4},
        {"params": model.backbone.layer3.parameters(), "lr": 5e-5},
        {"params": model.backbone.layer2.parameters(), "lr": 1e-5},
        {"params": model.backbone.layer1.parameters(), "lr": 1e-5},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_loader, colour="green")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            images, y_a, y_b, lam = mixup_data(images, labels)

            outputs = model(images)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            acc = (preds == labels).float().mean()

            progress_bar.set_description(
                f"Epoch {epoch+1} | Loss {loss.item():.4f} | Acc {acc:.4f}"
            )

        # 🔍 Validation
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)

        writer.add_scalar("Val/Accuracy", val_acc, epoch)
        scheduler.step()

        print(f"Epoch {epoch+1}: Val Acc = {val_acc:.4f}")

        # Save model
        torch.save(model.state_dict(), "training_models/last.pth")

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), "training_models/best.pth")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print("Early stopping!")
            break

    writer.close()