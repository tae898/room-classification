"""
Inspired by https://www.kaggle.com/shivanandmn/efficientnet-pytorch-lightning-train-inference
"""
import argparse
import json
import os
import random
from glob import glob

import albumentations as A
import numpy as np
# pytorch-lightning on top of PyTorch framework
import pytorch_lightning as pl
# PyTorch - deep learning framework
import torch
import torch.nn.functional as F
# from pytorch_lightning.metrics.functional import accuracy
import torchmetrics
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor
# for efficient model transfer learning
from efficientnet_pytorch import EfficientNet
# for albumentations uses cv2 where as torchvision transforms uses PIL
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils import crop_center_square


def get_splits(
    data_dir: str,
    limit_data: float,
    ratios: dict = {"train": 0.9, "val": 0.05, "test": 0.05},
    save_at: str = "splits.json",
) -> None:
    """Split data into train, val, and test splits.

    Args
    ----
    data_dir: data directory
    limit_data: e.g.,, 0.1 will only use 10% of the entire data.
    ratios: e.g., {"train": 0.9, "val": 0.05, "test": 0.05}
    save_at: save the split paths

    """
    paths_ = glob(f"{data_dir}/*/*.jpg") + glob(f"{data_dir}/*/*/*.jpg")

    random.shuffle(paths_)
    paths_ = paths_[: int(len(paths_) * limit_data)]

    assert sum(ratios.values()) == 1

    paths = {}
    paths["train"] = paths_[: int(len(paths_) * ratios["train"])]
    paths["val"] = paths_[
        int(len(paths_) * ratios["train"]) : int(
            len(paths_) * (ratios["train"] + ratios["val"])
        )
    ]
    paths["test"] = paths_[int(len(paths_) * (ratios["train"] + ratios["val"])) :]

    assert sum([len(foo) for foo in paths.values()]) == len(paths_)

    with open(save_at, "w") as stream:
        json.dump(paths, stream, indent=4)


class RoomEfficientNet(pl.LightningModule):
    """EfficientNet class for training and inference."""

    def __init__(
        self, num_classes: int, efficientnet: str, weights_path: str = None
    ) -> None:
        """
        Args
        ----
        num_classes: number of classes
        efficientnet: EfficientNet type (e.g., efficientnet-b3)

        """
        super().__init__()
        self.efficient_net = EfficientNet.from_pretrained(
            efficientnet, num_classes=num_classes, weights_path=weights_path
        )
        in_features = self.efficient_net._fc.in_features
        self.efficient_net._fc = nn.Linear(in_features, num_classes)
        self.calc_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        out = self.efficient_net(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        acc = self.calc_accuracy(y_hat, y)
        self.log(
            "train_acc",
            acc,
            prog_bar=True,
            logger=True,
        ),
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.calc_accuracy(y_hat, y)

        self.log("val_acc", acc, prog_bar=True, logger=True),
        self.log("val_loss", loss, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx) -> None:
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.calc_accuracy(y_hat, y)

        self.log("test_acc", acc, prog_bar=True, logger=True),
        self.log("test_loss", loss, prog_bar=True, logger=True)


class RoomDataset(Dataset):
    """Pytorch-compatible room type dataset"""

    def __init__(
        self, paths: list, transform, label2idx: dict, image_size: int
    ) -> None:
        """
        Args
        ----
        paths: image paths to load.
        transform:
        label2idx:
        image_size:

        """
        super().__init__()
        self.paths = paths
        self.transform = transform
        self.label2idx = label2idx
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        """Get x (data) and y (label)."""
        path = self.paths[item]
        img = Image.open(path)
        img = img.convert("RGB")

        img = crop_center_square(img)
        img = img.resize(size=(self.image_size, self.image_size))
        img = np.array(img)
        img = self.transform(image=img)

        # albumentations transform return a dictionary with "image" as key
        image = img["image"]
        label = path.split("/")[1].lower()
        return {
            "x": image,
            "y": self.label2idx[label],
        }


class RoomDataModule(pl.LightningDataModule):
    """Pytorch-lightning data module for the room dataset.

    This does initial setup, image augmentation, and loading dataloaser classes.

    """

    def __init__(self, image_size: int, batch_size: int) -> None:
        """
        Args
        ----
        image_size: image size (width and height)
        batch_size: batch size

        """
        super().__init__()
        self.image_size = image_size
        self.train_transform = A.Compose(
            [
                # A.Resize(int(image_size * 1.1), int(image_size * 1.1)),
                # A.RandomCrop(image_size, image_size),
                A.HorizontalFlip(),
                # A.VerticalFlip(),
                A.ShiftScaleRotate(),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensor(),
            ]
        )
        self.test_transform = A.Compose(
            [
                # A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensor(),
            ]
        )
        self.batch_size = batch_size

    def setup(
        self,
        stage=None,
        splits: str = "splits.json",
        label2idx: dict = {
            "interior": 0,
            "bathroom": 1,
            "bedroom": 2,
            "exterior": 3,
            "living_room": 4,
            "kitchen": 5,
            "dining_room": 6,
        },
    ) -> None:
        """
        Args
        ----
        stage: ?
        splits: path to the splits.json
        label2idx: label to index.

        """
        with open(splits, "r") as stream:
            splits = json.load(stream)

        self.label2idx = label2idx

        self.train_dataset = RoomDataset(
            paths=splits["train"],
            transform=self.train_transform,
            label2idx=label2idx,
            image_size=self.image_size,
        )

        self.val_dataset = RoomDataset(
            paths=splits["val"],
            transform=self.test_transform,
            label2idx=label2idx,
            image_size=self.image_size,
        )

        self.test_dataset = RoomDataset(
            paths=splits["test"],
            transform=self.test_transform,
            label2idx=label2idx,
            image_size=self.image_size,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)


def main(
    seed: int,
    batch_size: int,
    image_size: int,
    limit_data: float,
    num_classes: int,
    data_dir: str,
    efficientnet: str,
    epochs: int,
    use_gpu: bool,
    precision: int,
    patience: int,
) -> None:
    """Run training with the given arguments."""

    seed_everything(seed)

    get_splits(data_dir, limit_data)

    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        verbose=True,
        filename="{epoch}_{val_loss:.4f}_{val_acc:02f}",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=patience, verbose=True, mode="min"
    )

    dm = RoomDataModule(image_size=image_size, batch_size=batch_size)
    model = RoomEfficientNet(num_classes=num_classes, efficientnet=efficientnet)

    if use_gpu:
        gpus = -1
    else:
        gpus = 0
        precision = 32

    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=epochs,
        callbacks=[model_checkpoint, early_stop_callback],
        flush_logs_every_n_steps=1,
        log_every_n_steps=1,
        precision=precision,
    )
    trainer.fit(model=model, datamodule=dm)
    trainer.test(dataloaders=dm.test_dataloader())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training wiht pytorch-lightning.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Set the random seed for reproducibility",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="batch size",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=300,
        help="height and width of target image.",
    )

    parser.add_argument(
        "--limit-data",
        type=float,
        default=1.0,
        help="limit data. For example, 0.1 will only use 10 percent of the entire data.",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=7,
        help="number of classes.",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="data directory, inside which bathroom, bedroom, dining_room, "
        "Exterior, Interior, kitchen, and living_room should be located as "
        "sub-directories",
    )

    parser.add_argument(
        "--efficientnet",
        type=str,
        default="efficientnet-b3",
        help="efficientnet-b0, efficientnet-b1, efficientnet-b2, ...",
    )

    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train"
    )

    parser.add_argument(
        "--use-gpu", action="store_true", help="whether to use GPU or not"
    )

    parser.add_argument(
        "--precision", type=int, default=16, help="GPU floating point precision"
    )

    parser.add_argument(
        "--patience", type=int, default=3, help="Early stopping patience epochs."
    )

    args = vars(parser.parse_args())

    main(**args)
