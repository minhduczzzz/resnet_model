import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class DogBreedTrainValDataset(Dataset):
    def __init__(self, image_dir, dataframe, transform=None, class_to_idx=None):
        self.image_dir = image_dir
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

        if class_to_idx is None:
            classes = sorted(self.df["breed"].unique())
            self.class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_id = row["id"]
        breed = row["breed"]

        image_path = os.path.join(self.image_dir, image_id + ".jpg")
        image = Image.open(image_path).convert("RGB")
        label = self.class_to_idx[breed]

        if self.transform:
            image = self.transform(image)

        return image, label


class DogBreedTestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_file)

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        image_id = os.path.splitext(image_file)[0]
        return image, image_id