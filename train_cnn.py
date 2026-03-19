import os
import shutil
import torch
import torch.nn as nn
import pandas as pd

from sklearn.metrics import accuracy_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import (
    Compose, RandomResizedCrop, RandomHorizontalFlip,
    RandomRotation, ColorJitter, ToTensor, Normalize,
    Resize, CenterCrop, RandomErasing
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import DogBreedTrainValDataset
from model import DogBreedResNet


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 50
    batch_size = 32
    patience = 5
    best_acc = 0
    no_improve_epochs = 0

    labels_path = "labels.csv"
    train_dir = "data/train"

    # ================= AUGMENTATION =================
    train_transform = Compose([
        RandomResizedCrop(224, scale=(0.7, 1.0)),
        RandomHorizontalFlip(),
        RandomRotation(25),
        ColorJitter(0.4, 0.4, 0.4, 0.1),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406],
                  [0.229, 0.224, 0.225]),
        RandomErasing(p=0.5)
    ])

    val_transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406],
                  [0.229, 0.224, 0.225])
    ])

    # ================= DATA =================
    df = pd.read_csv(labels_path)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["breed"],
        random_state=42
    )

    train_dataset = DogBreedTrainValDataset(train_dir, train_df, train_transform)

    val_dataset = DogBreedTrainValDataset(
        train_dir,
        val_df,
        val_transform,
        class_to_idx=train_dataset.class_to_idx
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ================= LOG =================
    if os.path.isdir("tensorboard"):
        shutil.rmtree("tensorboard")

    os.makedirs("training_models", exist_ok=True)
    writer = SummaryWriter("tensorboard")

    # ================= MODEL =================
    num_classes = len(train_dataset.class_to_idx)
    model = DogBreedResNet(num_classes=num_classes, pretrained=True).to(device)

    # 🔥 Freeze backbone
    for param in model.resnet.parameters():
        param.requires_grad = False

    optimizer = Adam(model.resnet.fc.parameters(), lr=1e-4, weight_decay=5e-4)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=2,
        factor=0.3
    )

    criterion = nn.CrossEntropyLoss()

    # ================= TRAIN =================
    for epoch in range(num_epochs):

        # 🔥 Unfreeze layer4 sau 5 epoch
        if epoch == 5:
            for param in model.resnet.layer4.parameters():
                param.requires_grad = True

            optimizer = Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-5,
                weight_decay=5e-4
            )
            print("🔥 Unfroze layer4")

        model.train()
        loop = tqdm(train_loader, colour="green")

        for i, (images, labels) in enumerate(loop):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            acc = (preds == labels).float().mean()

            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item(), acc=acc.item())

            step = epoch * len(train_loader) + i
            writer.add_scalar("Train/Loss", loss.item(), step)
            writer.add_scalar("Train/Acc", acc.item(), step)

        # ================= VALID =================
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
        writer.add_scalar("Val/Acc", val_acc, epoch)

        scheduler.step(val_acc)

        print(f"Epoch {epoch+1} - Val Acc: {val_acc:.4f}")

        # ================= SAVE =================
        torch.save(model.state_dict(), "training_models/last.pth")

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), "training_models/best.pth")
        else:
            no_improve_epochs += 1

        # ================= EARLY STOP =================
        if no_improve_epochs >= patience:
            print("⛔ Early stopping")
            break

    writer.close()