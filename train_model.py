import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import time

from dataset import DogBreedTrainValDataset
from resnet import ResNet18


def accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    return (preds == labels).float().mean()


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========================
    # DATA PREPROCESSING
    # ========================

    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    # ========================
    # LOAD CSV
    # ========================

    labels_df = pd.read_csv("labels.csv")

    train_df, val_df = train_test_split(
        labels_df,
        test_size=0.2,
        stratify=labels_df["breed"],
        random_state=42
    )

    # ========================
    # DATASET
    # ========================

    train_dataset = DogBreedTrainValDataset(
        image_dir="data/train",
        dataframe=train_df,
        transform=train_transform
    )

    val_dataset = DogBreedTrainValDataset(
        image_dir="data/train",
        dataframe=val_df,
        transform=val_transform,
        class_to_idx=train_dataset.class_to_idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ========================
    # MODEL
    # ========================

    num_classes = len(train_dataset.class_to_idx)

    model = ResNet18(num_classes=num_classes)

    model = model.to(device)

    # ========================
    # LOSS + OPTIMIZER
    # ========================

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0003
    )

    # ========================
    # TRAIN LOOP
    # ========================

    epochs = 50

    for epoch in range(epochs):

        model.train()

        total_loss = 0
        train_acc = 0

        start_time = time.time()

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            train_acc += accuracy(outputs, labels).item()

        # ========================
        # VALIDATION
        # ========================

        model.eval()

        val_acc = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                val_acc += accuracy(outputs, labels).item()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss {total_loss/len(train_loader):.4f} | "
            f"Train Acc {train_acc/len(train_loader):.4f} | "
            f"Val Acc {val_acc/len(val_loader):.4f} | "
            f"Time {time.time()-start_time:.1f}s"
        )

    print("Training Finished")


if __name__ == "__main__":
    main()