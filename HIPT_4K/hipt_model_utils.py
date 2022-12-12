### Dependencies
# Base Dependencies
import argparse
import colorsys
from io import BytesIO
import os
import random
import requests
import sys

# LinAlg / Stats / Plotting Dependencies
import cv2
import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from scipy.stats import rankdata
import skimage.io
from skimage.measure import find_contours
from tqdm import tqdm
import webdataset as wds

# Torch Dependencies
import torch
import torch.multiprocessing
import torchvision
from torchvision import transforms
from einops import rearrange, repeat
torch.multiprocessing.set_sharing_strategy('file_system')

# Local Dependencies
import vision_transformer as vits
import vision_transformer4k as vits4k

def get_vit256(pretrained_weights, arch='vit_small', device=torch.device('cuda:0')):
    r"""
    Builds ViT-256 Model.
    
    Args:
    - pretrained_weights (str): Path to ViT-256 Model Checkpoint.
    - arch (str): Which model architecture.
    - device (torch): Torch device to save model.
    
    Returns:
    - model256 (torch.nn): Initialized model.
    """
    
    checkpoint_key = 'teacher'
    device = torch.device("cpu")
    model256 = vits.__dict__[arch](patch_size=16, num_classes=0)
    for p in model256.parameters():
        p.requires_grad = False
    model256.eval()
    model256.to(device)

    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model256.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        
    return model256


def get_vit4k(pretrained_weights, arch='vit4k_xs', device=torch.device('cuda:1')):
    r"""
    Builds ViT-4K Model.
    
    Args:
    - pretrained_weights (str): Path to ViT-4K Model Checkpoint.
    - arch (str): Which model architecture.
    - device (torch): Torch device to save model.
    
    Returns:
    - model256 (torch.nn): Initialized model.
    """
    
    checkpoint_key = 'teacher'
    device = torch.device("cpu")
    model4k = vits4k.__dict__[arch](num_classes=0)
    for p in model4k.parameters():
        p.requires_grad = False
    model4k.eval()
    model4k.to(device)

    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model4k.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        
    return model4k


def eval_transforms():
	"""
	"""
	mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
	eval_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)])
	return eval_t


def roll_batch2img(batch: torch.Tensor, w: int, h: int, patch_size=256):
	"""
	Rolls an image tensor batch (batch of [256 x 256] images) into a [W x H] Pil.Image object.
	
	Args:
		batch (torch.Tensor): [B x 3 x 256 x 256] image tensor batch.
		
	Return:
		Image.PIL: [W x H X 3] Image.
	"""
	batch = batch.reshape(w, h, 3, patch_size, patch_size)
	img = rearrange(batch, 'p1 p2 c w h-> c (p1 w) (p2 h)').unsqueeze(dim=0)
	return Image.fromarray(tensorbatch2im(img)[0])


def tensorbatch2im(input_image, imtype=np.uint8):
    r""""
    Converts a Tensor array into a numpy image array.
    
    Args:
        - input_image (torch.Tensor): (B, C, W, H) Torch Tensor.
        - imtype (type): the desired type of the converted numpy array
        
    Returns:
        - image_numpy (np.array): (B, W, H, C) Numpy Array.
    """
    if not isinstance(input_image, np.ndarray):
        image_numpy = input_image.cpu().float().numpy()  # convert it into a numpy array
        #if image_numpy.shape[0] == 1:  # grayscale to RGB
        #    image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)
