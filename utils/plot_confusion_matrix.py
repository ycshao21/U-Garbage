import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from pathlib import Path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    camp: str = "Blues",
) -> None:
    rcParams.update(
        {
            "font.size": 18,
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "axes.labelweight": "bold",
        }
    )

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        data=cm,
        cmap=camp,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xticks()
    plt.yticks()
    plt.tight_layout()

    figure_dir = Path("figures")
    if not figure_dir.exists():
        figure_dir.mkdir(parents=True)
    
    save_path = figure_dir / "confusion_matrix.png"
    plt.savefig(str(save_path))