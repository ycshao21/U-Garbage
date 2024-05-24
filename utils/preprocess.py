import shutil
import pandas as pd
from pathlib import Path
from torchvision import transforms
from PIL import Image
import random

# Paths to the dataset
DATA_DIR = Path("data/garbage_classification")
SAVE_DIR = Path("data/garbage_classification_preprocessed")

TARGET_NUM_PER_CATEGORY = 2600

CATEGORIES = {
    "battery": 0,
    "biological": 1,
    "brown-glass": 2,
    "cardboard": 3,
    "clothes": 4,
    "green-glass": 5,
    "metal": 6,
    "paper": 7,
    "plastic": 8,
    "shoes": 9,
    "trash": 10,
    "white-glass": 11,
}

img_size = (224, 224)

transform = transforms.Compose(
    [
        transforms.Resize(img_size, interpolation=Image.BILINEAR),
        transforms.Pad(padding=40, fill=5, padding_mode="reflect"),
        transforms.GaussianBlur(kernel_size=3),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomVerticalFlip(p=0.4),
        transforms.RandomRotation(degrees=20),
        transforms.CenterCrop(img_size),
    ]
)


def _preprocess_images() -> None:
    """Preprocess images and save them to the `SAVE_DIR` directory"""

    print("Preprocessing images...")

    for category_dir in DATA_DIR.iterdir():
        if not category_dir.is_dir():
            continue

        print(f"Processing {category_dir.name}...")

        save_category_dir = SAVE_DIR / category_dir.name
        if not save_category_dir.exists():
            save_category_dir.mkdir(exist_ok=True)

        # Deal with data imbalance >>>>>
        cnt = 1
        original_num = len(list(category_dir.iterdir()))
        current_num = len(list(save_category_dir.iterdir()))

        while current_num < TARGET_NUM_PER_CATEGORY:
            path_list = list(category_dir.iterdir())
            if current_num + original_num > TARGET_NUM_PER_CATEGORY:
                num = TARGET_NUM_PER_CATEGORY - current_num
                path_list = random.sample(path_list, num)

            for img_path in path_list:
                img_path_new = save_category_dir / f"{img_path.stem}_{cnt}.jpg"

                img = Image.open(img_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = transform(img)
                img.save(img_path_new, "JPEG")

            current_num = len(list(save_category_dir.iterdir()))
            cnt += 1
        # <<<<< Deal with data imbalance

    print("Preprocessing images finished.")


def _generate_csv() -> None:
    """Generate a CSV file containing the paths and labels of the images"""

    print("Generating CSV file...")

    df_data = []
    for category, label in CATEGORIES.items():
        for path in (SAVE_DIR / category).glob("*.jpg"):
            df_data.append((str(path), label))

    df = pd.DataFrame(df_data, columns=["path", "category"])

    csv_path = str(SAVE_DIR / "data.csv")
    df.to_csv(csv_path, index=False)

    print(f"CSV file saved to {csv_path}")


def preprocess_garbage() -> None:
    """Preprocess the garbage classification dataset and generate a CSV file

    Raises:
        FileNotFoundError: If the `DATA_DIR` directory is not found
    """

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"{DATA_DIR} not found.")

    # Remove the existing `SAVE_DIR` directory
    if SAVE_DIR.exists():
        shutil.rmtree(SAVE_DIR)

    SAVE_DIR.mkdir(exist_ok=True)

    try:
        _preprocess_images()
        _generate_csv()
        print("Preprocessing finished.")
    except Exception as e:
        print("Preprocessing failed.")
        print(e)


if __name__ == "__main__":
    preprocess_garbage()
