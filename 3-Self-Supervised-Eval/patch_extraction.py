### Dependencies
# Base Dependencies
import os
import pickle
import sys

# LinAlg / Stats / Plotting Dependencies
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import umap
import umap.plot
from tqdm import tqdm

# Torch Dependencies
import torch
import torch.multiprocessing
import torchvision
import torch.utils.data.dataset as Dataset
from torchvision import transforms
from pl_bolts.models.self_supervised import resnets
from pl_bolts.utils.semi_supervised import Identity
device = torch.device('cuda:0')
torch.multiprocessing.set_sharing_strategy('file_system')

# Model Architectures
from nn_encoder_arch.vision_transformer import vit_small
from nn_encoder_arch.resnet_trunc import resnet50_trunc_baseline

### Extracting Patch Features
patch_datasets = 'path/to/patch/datasets'
library_path = './embeddings_patch_library/'
os.makedirs(library_path, exist_ok=True)

models = ['resnet50_trunc', 'resnet50_tcga_brca_simclr', 'vits_tcga_brca_dino']

for enc_name in models:
    create_embeddings(patch_datasets=patch_datasets, embeddings_dir=library_path, enc_name=enc_name, dataset='crc100knonorm')
    create_embeddings(patch_datasets=patch_datasets, embeddings_dir=library_path, enc_name=enc_name, dataset='crc100k')
    create_embeddings(patch_datasets=patch_datasets, embeddings_dir=library_path, enc_name=enc_name, dataset='bcss')
    create_embeddings(patch_datasets=patch_datasets, embeddings_dir=library_path, enc_name=enc_name, dataset='breastpathq')