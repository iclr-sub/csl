# Cost-Sensitive Learning (CSL) Loss Function

This repository contains the code for training machine learning models using a novel Cost-Sensitive Learning (CSL) loss function designed to handle class imbalance effectively. The code is modular and has been tested on multiple datasets, including ImageNet-LT, CIFAR-10, CIFAR-100, Tiny ImageNet and iNaturalist2018.

## Datasets
The following datasets are supported and can be used for training and evaluation:
- ImageNet-LT
- iNaturalist2018
- CIFAR-10
- CIFAR-100
- Tiny ImageNet

## Getting Started

Follow these instructions to set up and run the project.

### 1. Clone the Repository

Clone the repository using the following command:

```bash
git clone https://github.com/iclr-sub/csl.git
cd csl
```

### 2. Create and Activate Virtual Environment

To ensure package compatibility, create a virtual environment:

```bash
python -m venv csl_env
```

Activate the virtual environment:

- On Windows:
    ```bash
    csl_env\Scripts\activate
    ```
- On Unix/macOS:
    ```bash
    source csl_env/bin/activate
    ```

### 3. Install Dependencies

Install the required dependencies specified in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Prepare Datasets

Place the required datasets in the `datasets/` directory. The structure should follow this pattern:

```
datasets/
    ImageNet_LT/
    INaturalist18/
    CIFAR-10/
    CIFAR-100/
    TinyImageNet/
```

Ensure the appropriate images and labels are placed correctly for each dataset. You can refer to the dataset loader files for more details on the format required.

### 5. Training the Model on ImageNet-LT or iNaturalist2018

To train models using the ImageNet-LT or iNaturalist datasets, use the `main.py` script. You can customize the dataset, model, batch size, and other hyperparameters as needed.

```bash
python main.py --dataset_name imagenet --model_name resnet50 
```

Example for training on iNaturalist with ResNet-50:

```bash
python main.py --dataset_name inaturalist --model_name resnet50
```

### 6. Running Experiments on CIFAR-10, CIFAR-100, and Tiny ImageNet

We provide Jupyter notebooks for training and evaluating models on CIFAR-10, CIFAR-100, and Tiny ImageNet. You can execute these notebooks directly by opening them in Jupyter:

- **`cifar_10.ipynb`**
- **`cifar_100.ipynb`**
- **`tiny_imagenet.ipynb`**

To start the Jupyter environment:

```bash
jupyter notebook
```

Open the relevant notebook and run the cells to execute the experiments.

### Code Structure

The repository is organized as follows:

```
├── datasets/                   # Datasets directory (not  included in the repo)
├── dataloaders/
│   ├── __init__.py
│   ├── ImageNet_LT/
│   ├── Inaturalist18/
│   ├── imagenet_lt_loader.py    # ImageNet-LT dataset loader
│   └── inaturalist_loader.py    # iNaturalist dataset loader
├── models/
│   ├── __init__.py
│   ├── resnet.py                # ResNet model implementations
│   └── resnext.py               # ResNeXt model implementations
├── notebooks/
│   ├── cifar_10.ipynb           # Experiment notebook for CIFAR-10
│   ├── cifar_100.ipynb          # Experiment notebook for CIFAR-100
│   └── tiny_imagenet.ipynb      # Experiment notebook for Tiny ImageNet
├── utils/
│   ├── __init__.py
│   ├── csl_loss.py              # Implementation of the CSL loss function
│   └── plot_utils.py            # Utility functions for plotting results
├── main.py                      # Main training script
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

### Arguments for Training

When running `main.py`, you can pass in various arguments to customize the training process:

- `--dataset_name`: The dataset to use (`imagenet`, `inaturalist`).
- `--model_name`: The model to use (`resnet32`, `resnet50`, `resnext50`, `resnext101`).
- `--batch_size`: Batch size for training (default: 256).
- `--num_epochs`: Number of training epochs (default: 200).
- `--learning_rate`: Initial learning rate (default: 0.01).
- `--data_path`: Path to the dataset directory (default: `datasets/`).
