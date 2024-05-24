from load_data import get_dataloader
from models import VGG16

import torch
from utils.plot_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix

from utils import mylogger
import logging

mylogger.setup("configs/logger.yaml")
logger = logging.getLogger("root")

from ruamel.yaml import YAML

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, test_loader, plot_cm: bool):
    logger.info("------------ Test ------------")

    labels = []
    preds = []

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            img, label = data
            img, label = img.to(device), label.to(device)

            pred = model(img)
            pred = pred.argmax(dim=1)

            labels.append(label)
            preds.append(pred)

    labels = torch.cat(labels).cpu().numpy()
    preds = torch.cat(preds).cpu().numpy()

    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    logger.info(f"Accuracy on test set: {acc}")
    logger.info(f"Confusion matrix:\n{cm}")

    if plot_cm:
        plot_confusion_matrix(cm, class_names=[str(i) for i in range(12)])


if __name__ == "__main__":
    with open("configs/train.yaml") as f:
        config = YAML().load(f)["test"]

    model = VGG16(
        in_channels=3,
        num_classes=12,
        use_cbam=config["use_cbam"],
        use_residual=config["use_residual"],
    )
    model.load_state_dict(torch.load(config["model_path"]))
    model = model.to(device)

    _, _, test_loader = get_dataloader(config["batch_size"])

    test(model, test_loader, config["plot_cm"])
