# U-Garbage
A garbage sorting model using VGG16, Residual Blocks, and Convolutional Block Attention Module, implemented in PyTorch.

## How to get started
Clone the repository:
```bash
git clone git@github.com:ycshao21/U-Garbage.git
cd U-Garbage
```

Set up the conda environment:
```bash
conda create -n garbage python=3.12
conda activate garbage

conda install numpy pandas matplotlib
conda install pillow
conda install ruamel.yaml
pip install seaborn=0.13.1

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

conda install tensorboardX
```

Download the dataset [Garbage Classification (12 classes)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification/data) by Mostafa Mohamed from Kaggle, and unzip it under `data/`.

Preprocess the dataset:
```bash
python utils/preprocess.py
```

To train the model, modify `configs/train.yaml` and run:
```bash
python train_vgg.py --name <name>
```

To test the model, modify `configs/train.yaml` and run:
```bash
python test.py
```