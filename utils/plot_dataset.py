from matplotlib import pyplot as plt, rcParams
from pathlib import Path
from preprocess import CATEGORIES
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/garbage_classification")


NUM_COLORS = 12
base_color = np.array([135/255, 206/255, 235/255])
colors = [(base_color * (i / (NUM_COLORS - 1))) for i in range(NUM_COLORS)]
colors = [(base_color * (0.3 + 0.7 * i / (NUM_COLORS - 1))) for i in range(NUM_COLORS)]


def plot_dataset(data_dir: Path, save_name: str):
    """Plot the number of images per category in the dataset

    Args:
        dataset (Path): Path to the dataset
        save_name (str): Name of the saved figure
    """
    rcParams.update(
        {
            "font.size": 18,
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "axes.labelweight": "bold",
        }
    )

    categories = [data_dir / category for category, _ in CATEGORIES.items()]
    num_per_category = [len(list(category.iterdir())) for category in categories]

    # Sort by number of images
    num_per_category, categories = zip(
        *sorted(zip(num_per_category, categories), key=lambda x: x[0], reverse=True)
    )

    plt.figure(figsize=(12, 5))
    plt.bar([category.name for category in categories], num_per_category, color=colors)
    plt.ylabel("Number of Images")
    plt.tight_layout()

    figure_dir = Path("figures")
    if not figure_dir.exists():
        figure_dir.mkdir(parents=True)
    plt.savefig(str(figure_dir / save_name))


if __name__ == "__main__":
    plot_dataset(DATA_DIR, save_name="number_of_original_images_per_category.png")
