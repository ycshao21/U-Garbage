import pandas as pd
from matplotlib import pyplot as plt, rcParams
from pathlib import Path
from pathlib import Path


LABELS = [
    "VGG16",
    "VGG16 + CBAM",
    "VGG16 + Residual",
    "VGG16 + Residual + CBAM",
]

figsize = (9, 6)

colors = [
    "#4B0082",
    "#FF1493",
    "#FFD700",
    "#808000",
]


def plot_valid_acc():
    """Plot the accuracy curve on validation set
    [NOTE] Please download the data from tensorboard and save them in the `data/result/valid_acc` folder.
    """
    data_dir = Path("data/result/valid_acc")
    paths = [
        data_dir / "vgg16.csv",
        data_dir / "vgg16_cbam.csv",
        data_dir / "vgg16_res.csv",
        data_dir / "vgg16_res_cbam.csv",
    ]

    rcParams.update(
        {
            "font.size": 18,
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "axes.labelweight": "bold",
        }
    )

    plt.figure(figsize=figsize)
    for path, label in zip(paths, LABELS):
        df = pd.read_csv(path)
        plt.plot(
            df["Step"],
            df["Value"],
            label=label,
            linewidth=1.2,
            color=colors[LABELS.index(label)],
        )

    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.tight_layout()

    figure_dir = Path("figures")
    if not figure_dir.exists():
        figure_dir.mkdir(parents=True)
    plt.savefig(str(figure_dir / "val_acc.png"))


def plot_valid_loss():
    """Plot the validation loss curve
    [NOTE] Please download the data from tensorboard and save them in the `data/result/valid_loss` folder.
    """
    data_dir = Path("data/result/valid_loss")
    paths = [
        data_dir / "vgg16.csv",
        data_dir / "vgg16_cbam.csv",
        data_dir / "vgg16_res.csv",
        data_dir / "vgg16_res_cbam.csv",
    ]

    rcParams.update(
        {
            "font.size": 18,
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "axes.labelweight": "bold",
        }
    )

    plt.figure(figsize=figsize)
    for path, label in zip(paths, LABELS):
        df = pd.read_csv(path)
        plt.plot(
            df["Step"],
            df["Value"],
            label=label,
            linewidth=1.2,
            color=colors[LABELS.index(label)],
        )

    plt.xlabel("Step")
    plt.ylabel("Validation Loss")
    plt.legend(loc="upper right")
    plt.tight_layout()

    figure_dir = Path("figures")
    if not figure_dir.exists():
        figure_dir.mkdir(parents=True)
    plt.savefig(str(figure_dir / "val_loss.png"))


if __name__ == "__main__":
    plot_valid_acc()
    plot_valid_loss()
