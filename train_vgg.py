from load_data import get_dataloader
import argparse
import models
from test import test

import os
from utils import mylogger
import logging

from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")

mylogger.setup("configs/logger.yaml")
logger = logging.getLogger("root")

import torch
import torch.nn as nn
import torch.optim as optim
import tensorboardX

from ruamel.yaml import YAML


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    num_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    use_cbam: bool,
    use_residual: bool,
    save_dir: str,
    test_after_train: bool = False,
    name: str = "model",
):
    save_dir = f"{save_dir}/{name}_{TIMESTAMP}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    writer_path = f"runs/{name}_{TIMESTAMP}"
    writer = tensorboardX.SummaryWriter(writer_path)

    # Get dataloaders
    train_loader, valid_loader, test_loader = get_dataloader(batch_size)

    # Input: 3 * 224 * 224 Output: 12
    model = models.VGG16(in_channels=3, num_classes=12, use_cbam=use_cbam, use_residual=use_residual)
    model = model.to(device)

    # Optimizer: Adam
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Loss function: CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    # Learning rate scheduler: StepLR
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.4)

    logger.info("------------ Training started ------------")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Weight decay: {weight_decay}")
    logger.info(f"Use CBAM: {use_cbam}")
    logger.info(f"Save directory: {save_dir}")
    logger.info(f"Name: {name}")

    best_acc = 0
    best_epoch = 0
    no_improve = 0

    for epoch in range(num_epochs):
        # Training >>>>>
        model.train()
        for batch_idx, data in enumerate(train_loader):
            img, label = data
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            writer.add_scalar("train/loss", loss.item(), epoch * len(train_loader) + batch_idx)
            logger.info(
                f"Epoch {epoch}/{num_epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item()}"
            )
        # <<<<< Training

        # Validation
        logger.info("------------ Validation ------------")

        model.eval()
        with torch.no_grad():
            val_acc = []
            for batch_idx, data in enumerate(valid_loader):
                img, label = data
                img, label = img.to(device), label.to(device)

                pred = model(img)
                loss = criterion(pred, label)

                # Accuracy
                acc = (pred.argmax(1) == label).float().mean()
                val_acc.append(acc)
                writer.add_scalar("valid/loss", loss.item(), epoch * len(valid_loader) + batch_idx)
                writer.add_scalar("valid/acc_batch", acc.item(), epoch * len(valid_loader) + batch_idx)
                logger.info(
                    f"Epoch {epoch}/{num_epochs} - Batch {batch_idx}/{len(valid_loader)} - Loss: {loss.item()}"
                )

            val_acc = torch.tensor(val_acc).mean()
            writer.add_scalar("valid/accuracy", val_acc.item(), epoch)
            logger.info(
                f"Epoch {epoch}/{num_epochs} - Accuracy: {val_acc}"
            )

            # If the model has the best accuracy, save it and update the best accuracy
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), f"{save_dir}/model.pth")
                logger.info(f"Model saved at {save_dir}/model.pth")
                no_improve = 0
            else:
                no_improve += 1

                if no_improve >= 5:
                    logger.info("Training stopped due to no improvement in 5 epochs.")
                    break
        # <<<<< Validation

        # Update learning rate
        scheduler.step()
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
        logger.info(
            f"Epoch {epoch}/{num_epochs} - Learning rate: {scheduler.get_last_lr()[0]}"
        )

    logger.info("------------ Training finished ------------")
    logger.info(f"Best accuracy: {best_acc} at epoch {best_epoch}")

    # Test
    if test_after_train:
        model.load_state_dict(torch.load(f"{save_dir}/model.pth"))
        test(model, test_loader)


if __name__ == "__main__":
    # torch.manual_seed(1337)

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--name", type=str, default='model')
    args = argparser.parse_args()

    with open("configs/train.yaml") as f:
        config = YAML().load(f)["train"]

    train(**config, name=args.name)
