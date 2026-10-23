"""Configuration for BiCS-MVC."""
import os
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
# datasets folder is at parent directory
DATA_DIR = os.path.join(os.path.dirname(BASE), 'datasets')
RESULTS_DIR = os.path.join(BASE, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
FEATURE_DIM = 512
HIGH_DIM = 128
BATCH_SIZE = 64
TEMPERATURE = 0.5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 50
LAMBDA_BI = 1.0
LAMBDA_SEM = 0.3

# Evaluation
N_RUNS = 10
SEED_START = 42

# Datasets (available in datasets folder)
TARGETS = ['MNIST_USPS', 'NUSWIDE_5000', 'Hdigit', 'Fashion_MV', 'Digit_Product']
# TARGETS = ['Cifar100','YouTubeVideo']
METHOD = 'BiCS-MVC'
