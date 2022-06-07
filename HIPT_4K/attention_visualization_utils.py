### Dependencies
import argparse
import colorsys
from io import BytesIO
import os
import random
import requests
import sys

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

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import torchvision.transforms as transforms
from einops import rearrange, repeat

sys.path.append('../')
sys.path.append('../Hierarchical-Pretraining/')
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
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
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
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
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


def cmap_map(function, cmap):
    r""" 
    Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    
    Args:
    - function (function)
    - cmap (matplotlib.colormap)
    
    Returns:
    - matplotlib.colormap
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)


def identity(x):
    r"""
    Identity Function.
    
    Args:
    - x:
    
    Returns:
    - x
    """
    return x

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

def getConcatImage(imgs, how='horizontal', gap=0):
    r"""
    Function to concatenate list of images (vertical or horizontal).

    Args:
        - imgs (list of PIL.Image): List of PIL Images to concatenate.
        - how (str): How the images are concatenated (either 'horizontal' or 'vertical')
        - gap (int): Gap (in px) between images

    Return:
        - dst (PIL.Image): Concatenated image result.
    """
    gap_dist = (len(imgs)-1)*gap
    
    if how == 'vertical':
        w, h = np.max([img.width for img in imgs]), np.sum([img.height for img in imgs])
        h += gap_dist
        curr_h = 0
        dst = Image.new('RGBA', (w, h), color=(255, 255, 255, 0))
        for img in imgs:
            dst.paste(img, (0, curr_h))
            curr_h += img.height + gap

    elif how == 'horizontal':
        w, h = np.sum([img.width for img in imgs]), np.min([img.height for img in imgs])
        w += gap_dist
        curr_w = 0
        dst = Image.new('RGBA', (w, h), color=(255, 255, 255, 0))

        for idx, img in enumerate(imgs):
            dst.paste(img, (curr_w, 0))
            curr_w += img.width + gap

    return dst


def add_margin(pil_img, top, right, bottom, left, color):
    r"""
    Adds custom margin to PIL.Image.
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def concat_scores256(attns, size=(256,256)):
    r"""
    """
    rank = lambda v: rankdata(v)*100/len(v)
    color_block = [rank(attn.flatten()).reshape(size) for attn in attns]
    color_hm = np.concatenate([
        np.concatenate(color_block[i:(i+16)], axis=1)
        for i in range(0,256,16)
    ])
    return color_hm


def concat_scores4k(attn, size=(4096, 4096)):
    r"""
    """
    rank = lambda v: rankdata(v)*100/len(v)
    color_hm = rank(attn.flatten()).reshape(size)
    return color_hm



def get_scores256(attns, size=(256,256)):
    r"""
    """
    rank = lambda v: rankdata(v)*100/len(v)
    color_block = [rank(attn.flatten()).reshape(size) for attn in attns][0]
    return color_block


def get_patch_attention_scores(patch, model256, scale=1, device256=torch.device('cuda:0')):
    r"""
    Forward pass in ViT-256 model with attention scores saved.
    
    Args:
    - region (PIL.Image):       4096 x 4096 Image 
    - model256 (torch.nn):      256-Level ViT 
    - scale (int):              How much to scale the output image by (e.g. - scale=4 will resize images to be 1024 x 1024.)
    
    Returns:
    - attention_256 (torch.Tensor): [1, 256/scale, 256/scale, 3] torch.Tensor of attention maps for 256-sized patches.
    """
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        )
    ])

    with torch.no_grad():   
        batch_256 = t(patch).unsqueeze(0)
        batch_256 = batch_256.to(device256, non_blocking=True)
        features_256 = model256(batch_256)

        attention_256 = model256.get_last_selfattention(batch_256)
        nh = attention_256.shape[1] # number of head
        attention_256 = attention_256[:, :, 0, 1:].reshape(256, nh, -1)
        attention_256 = attention_256.reshape(1, nh, 16, 16)
        attention_256 = nn.functional.interpolate(attention_256, scale_factor=int(16/scale), mode="nearest").cpu().numpy()

        if scale != 1:
            batch_256 = nn.functional.interpolate(batch_256, scale_factor=(1/scale), mode="nearest")
            
    return tensorbatch2im(batch_256), attention_256


def create_patch_heatmaps_indiv(patch, model256, output_dir, fname, threshold=0.5,
                             offset=16, alpha=0.5, cmap=plt.get_cmap('coolwarm')):
    r"""
    Creates patch heatmaps (saved individually)
    
    Args:
    - patch (PIL.Image):        256 x 256 Image 
    - model256 (torch.nn):      256-Level ViT 
    - output_dir (str):         Save directory / subdirectory
    - fname (str):              Naming structure of files
    - offset (int):             How much to offset (from top-left corner with zero-padding) the region by for blending 
    - alpha (float):            Image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): Colormap for creating heatmaps
    
    Returns:
    - None
    """
    patch1 = patch.copy()
    patch2 = add_margin(patch.crop((16,16,256,256)), top=0, left=0, bottom=16, right=16, color=(255,255,255))
    b256_1, a256_1 = get_patch_attention_scores(patch1, model256)
    b256_1, a256_2 = get_patch_attention_scores(patch2, model256)
    save_region = np.array(patch.copy())
    s = 256
    offset_2 = offset

    if threshold != None:
        for i in range(6):
            score256_1 = get_scores256(a256_1[:,i,:,:], size=(s,)*2)
            score256_2 = get_scores256(a256_2[:,i,:,:], size=(s,)*2)
            new_score256_2 = np.zeros_like(score256_2)
            new_score256_2[offset_2:s, offset_2:s] = score256_2[:(s-offset_2), :(s-offset_2)]
            overlay256 = np.ones_like(score256_2)*100
            overlay256[offset_2:s, offset_2:s] += 100
            score256 = (score256_1+new_score256_2)/overlay256

            mask256 = score256.copy()
            mask256[mask256 < threshold] = 0
            mask256[mask256 > threshold] = 0.95

            color_block256 = (cmap(mask256)*255)[:,:,:3].astype(np.uint8)
            region256_hm = cv2.addWeighted(color_block256, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            region256_hm[mask256==0] = 0
            img_inverse = save_region.copy()
            img_inverse[mask256 == 0.95] = 0
            Image.fromarray(region256_hm+img_inverse).save(os.path.join(output_dir, '%s_256th[%d].png' % (fname, i)))

    for i in range(6):
        score256_1 = get_scores256(a256_1[:,i,:,:], size=(s,)*2)
        score256_2 = get_scores256(a256_2[:,i,:,:], size=(s,)*2)
        new_score256_2 = np.zeros_like(score256_2)
        new_score256_2[offset_2:s, offset_2:s] = score256_2[:(s-offset_2), :(s-offset_2)]
        overlay256 = np.ones_like(score256_2)*100
        overlay256[offset_2:s, offset_2:s] += 100
        score256 = (score256_1+new_score256_2)/overlay256
        color_block256 = (cmap(score256)*255)[:,:,:3].astype(np.uint8)
        region256_hm = cv2.addWeighted(color_block256, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
        Image.fromarray(region256_hm).save(os.path.join(output_dir, '%s_256[%s].png' % (fname, i)))
        
        
def create_patch_heatmaps_concat(patch, model256, output_dir, fname, threshold=0.5,
                             offset=16, alpha=0.5, cmap=plt.get_cmap('coolwarm')):
    r"""
    Creates patch heatmaps (concatenated for easy comparison)
    
    Args:
    - patch (PIL.Image):        256 x 256 Image 
    - model256 (torch.nn):      256-Level ViT 
    - output_dir (str):         Save directory / subdirectory
    - fname (str):              Naming structure of files
    - offset (int):             How much to offset (from top-left corner with zero-padding) the region by for blending 
    - alpha (float):            Image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): Colormap for creating heatmaps
    
    Returns:
    - None
    """
    patch1 = patch.copy()
    patch2 = add_margin(patch.crop((16,16,256,256)), top=0, left=0, bottom=16, right=16, color=(255,255,255))
    b256_1, a256_1 = get_patch_attention_scores(patch1, model256)
    b256_1, a256_2 = get_patch_attention_scores(patch2, model256)
    save_region = np.array(patch.copy())
    s = 256
    offset_2 = offset

    if threshold != None:
        ths = []
        for i in range(6):
            score256_1 = get_scores256(a256_1[:,i,:,:], size=(s,)*2)
            score256_2 = get_scores256(a256_2[:,i,:,:], size=(s,)*2)
            new_score256_2 = np.zeros_like(score256_2)
            new_score256_2[offset_2:s, offset_2:s] = score256_2[:(s-offset_2), :(s-offset_2)]
            overlay256 = np.ones_like(score256_2)*100
            overlay256[offset_2:s, offset_2:s] += 100
            score256 = (score256_1+new_score256_2)/overlay256

            mask256 = score256.copy()
            mask256[mask256 < threshold] = 0
            mask256[mask256 > threshold] = 0.95

            color_block256 = (cmap(mask256)*255)[:,:,:3].astype(np.uint8)
            region256_hm = cv2.addWeighted(color_block256, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            region256_hm[mask256==0] = 0
            img_inverse = save_region.copy()
            img_inverse[mask256 == 0.95] = 0
            ths.append(region256_hm+img_inverse)
            
        ths = [Image.fromarray(img) for img in ths]
            
        getConcatImage([getConcatImage(ths[0:3]), 
                        getConcatImage(ths[3:6])], how='vertical').save(os.path.join(output_dir, '%s_256th.png' % (fname)))
    
    
    hms = []
    for i in range(6):
        score256_1 = get_scores256(a256_1[:,i,:,:], size=(s,)*2)
        score256_2 = get_scores256(a256_2[:,i,:,:], size=(s,)*2)
        new_score256_2 = np.zeros_like(score256_2)
        new_score256_2[offset_2:s, offset_2:s] = score256_2[:(s-offset_2), :(s-offset_2)]
        overlay256 = np.ones_like(score256_2)*100
        overlay256[offset_2:s, offset_2:s] += 100
        score256 = (score256_1+new_score256_2)/overlay256
        color_block256 = (cmap(score256)*255)[:,:,:3].astype(np.uint8)
        region256_hm = cv2.addWeighted(color_block256, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
        hms.append(region256_hm)
        
    hms = [Image.fromarray(img) for img in hms]
        
    getConcatImage([getConcatImage(hms[0:3]), 
                    getConcatImage(hms[3:6])], how='vertical').save(os.path.join(output_dir, '%s_256hm.png' % (fname)))

    
def hipt_forward_pass(region, model256, model4k, scale=1,
                                device256=torch.device('cuda:0'), 
                                device4k=torch.device('cuda:1')):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        )
    ])

    with torch.no_grad():   
        batch_256 = t(region).unsqueeze(0).unfold(2, 256, 256).unfold(3, 256, 256)
        batch_256 = rearrange(batch_256, 'b c p1 p2 w h -> (b p1 p2) c w h')
        batch_256 = batch_256.to(device256, non_blocking=True)
        features_256 = model256(batch_256)
        features_256 = features_256.unfold(0, 16, 16).transpose(0,1).unsqueeze(dim=0)
        features_4096 = model4k.forward(features_256.to(device4k))
        return features_4096


def get_region_attention_scores(region, model256, model4k, scale=1,
                                device256=torch.device('cuda:0'), 
                                device4k=torch.device('cuda:1')):
    r"""
    Forward pass in hierarchical model with attention scores saved.
    
    Args:
    - region (PIL.Image):       4096 x 4096 Image 
    - model256 (torch.nn):      256-Level ViT 
    - model4k (torch.nn):       4096-Level ViT 
    - scale (int):              How much to scale the output image by (e.g. - scale=4 will resize images to be 1024 x 1024.)
    
    Returns:
    - np.array: [256, 256/scale, 256/scale, 3] np.array sequence of image patches from the 4K x 4K region.
    - attention_256 (torch.Tensor): [256, 256/scale, 256/scale, 3] torch.Tensor sequence of attention maps for 256-sized patches.
    - attention_4k (torch.Tensor): [1, 4096/scale, 4096/scale, 3] torch.Tensor sequence of attention maps for 4k-sized regions.
    """
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        )
    ])

    with torch.no_grad():   
        batch_256 = t(region).unsqueeze(0).unfold(2, 256, 256).unfold(3, 256, 256)
        batch_256 = rearrange(batch_256, 'b c p1 p2 w h -> (b p1 p2) c w h')
        batch_256 = batch_256.to(device256, non_blocking=True)
        features_256 = model256(batch_256)

        attention_256 = model256.get_last_selfattention(batch_256)
        nh = attention_256.shape[1] # number of head
        attention_256 = attention_256[:, :, 0, 1:].reshape(256, nh, -1)
        attention_256 = attention_256.reshape(256, nh, 16, 16)
        attention_256 = nn.functional.interpolate(attention_256, scale_factor=int(16/scale), mode="nearest").cpu().numpy()

        features_4096 = features_256.unfold(0, 16, 16).transpose(0,1).unsqueeze(dim=0)
        attention_4096 = model4k.get_last_selfattention(features_4096.detach().to(device4k))
        nh = attention_4096.shape[1] # number of head
        attention_4096 = attention_4096[0, :, 0, 1:].reshape(nh, -1)
        attention_4096 = attention_4096.reshape(nh, 16, 16)
        attention_4096 = nn.functional.interpolate(attention_4096.unsqueeze(0), scale_factor=int(256/scale), mode="nearest")[0].cpu().numpy()

        if scale != 1:
            batch_256 = nn.functional.interpolate(batch_256, scale_factor=(1/scale), mode="nearest")

    return tensorbatch2im(batch_256), attention_256, attention_4096


def create_hierarchical_heatmaps_indiv(region, model256, model4k, output_dir, fname,
                             offset=128, scale=4, alpha=0.5, cmap = plt.get_cmap('coolwarm'), threshold=None):
    r"""
    Creates hierarchical heatmaps (Raw H&E + ViT-256 + ViT-4K + Blended Heatmaps saved individually).  
    
    Args:
    - region (PIL.Image):       4096 x 4096 Image 
    - model256 (torch.nn):      256-Level ViT 
    - model4k (torch.nn):       4096-Level ViT 
    - output_dir (str):         Save directory / subdirectory
    - fname (str):              Naming structure of files
    - offset (int):             How much to offset (from top-left corner with zero-padding) the region by for blending 
    - scale (int):              How much to scale the output image by 
    - alpha (float):            Image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): Colormap for creating heatmaps
    
    Returns:
    - None
    """
    
    region2 = add_margin(region.crop((128,128,4096,4096)), 
                     top=0, left=0, bottom=128, right=128, color=(255,255,255))
    region3 = add_margin(region.crop((128*2,128*2,4096,4096)), 
                     top=0, left=0, bottom=128*2, right=128*2, color=(255,255,255))
    region4 = add_margin(region.crop((128*3,128*3,4096,4096)), 
                     top=0, left=0, bottom=128*4, right=128*4, color=(255,255,255))
    
    b256_1, a256_1, a4k_1 = get_region_attention_scores(region, model256, model4k, scale)
    
    b256_2, a256_2, a4k_2 = get_region_attention_scores(region2, model256, model4k, scale)
    b256_3, a256_3, a4k_3 = get_region_attention_scores(region3, model256, model4k, scale)
    b256_4, a256_4, a4k_4 = get_region_attention_scores(region4, model256, model4k, scale)
    offset_2 = (offset*1)//scale
    offset_3 = (offset*2)//scale
    offset_4 = (offset*3)//scale
    s = 4096//scale
    save_region = np.array(region.resize((s, s)))
    
    if threshold != None:
        for i in range(6):
            score256_1 = concat_scores256(a256_1[:,i,:,:], size=(s//16,)*2)
            score256_2 = concat_scores256(a256_2[:,i,:,:], size=(s//16,)*2)
            new_score256_2 = np.zeros_like(score256_2)
            new_score256_2[offset_2:s, offset_2:s] = score256_2[:(s-offset_2), :(s-offset_2)]
            overlay256 = np.ones_like(score256_2)*100
            overlay256[offset_2:s, offset_2:s] += 100
            score256 = (score256_1+new_score256_2)/overlay256
            
            mask256 = score256.copy()
            mask256[mask256 < threshold] = 0
            mask256[mask256 > threshold] = 0.95
            
            color_block256 = (cmap(mask256)*255)[:,:,:3].astype(np.uint8)
            region256_hm = cv2.addWeighted(color_block256, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            region256_hm[mask256==0] = 0
            img_inverse = save_region.copy()
            img_inverse[mask256 == 0.95] = 0
            Image.fromarray(region256_hm+img_inverse).save(os.path.join(output_dir, '%s_256th[%d].png' % (fname, i)))
    
    if False:
        for j in range(6):
            score4k_1 = concat_scores4k(a4k_1[j], size=(s,)*2)
            score4k = score4k_1 / 100
            color_block4k = (cmap(score4k)*255)[:,:,:3].astype(np.uint8)
            region4k_hm = cv2.addWeighted(color_block4k, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            Image.fromarray(region4k_hm).save(os.path.join(output_dir, '%s_4k[%s].png' % (fname, j)))
        
    for j in range(6):
        score4k_1 = concat_scores4k(a4k_1[j], size=(s,)*2)
        score4k_2 = concat_scores4k(a4k_2[j], size=(s,)*2)
        score4k_3 = concat_scores4k(a4k_3[j], size=(s,)*2)
        score4k_4 = concat_scores4k(a4k_4[j], size=(s,)*2)

        new_score4k_2 = np.zeros_like(score4k_2)
        new_score4k_2[offset_2:s, offset_2:s] = score4k_2[:(s-offset_2), :(s-offset_2)]
        new_score4k_3 = np.zeros_like(score4k_3)
        new_score4k_3[offset_3:s, offset_3:s] = score4k_3[:(s-offset_3), :(s-offset_3)]
        new_score4k_4 = np.zeros_like(score4k_4)
        new_score4k_4[offset_4:s, offset_4:s] = score4k_4[:(s-offset_4), :(s-offset_4)]

        overlay4k = np.ones_like(score4k_2)*100
        overlay4k[offset_2:s, offset_2:s] += 100
        overlay4k[offset_3:s, offset_3:s] += 100
        overlay4k[offset_4:s, offset_4:s] += 100
        score4k = (score4k_1+new_score4k_2+new_score4k_3+new_score4k_4)/overlay4k

        color_block4k = (cmap(score4k)*255)[:,:,:3].astype(np.uint8)
        region4k_hm = cv2.addWeighted(color_block4k, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
        Image.fromarray(region4k_hm).save(os.path.join(output_dir, '%s_1024[%s].png' % (fname, j)))
        
    for i in range(6):
        score256_1 = concat_scores256(a256_1[:,i,:,:], size=(s//16,)*2)
        score256_2 = concat_scores256(a256_2[:,i,:,:], size=(s//16,)*2)
        new_score256_2 = np.zeros_like(score256_2)
        new_score256_2[offset_2:s, offset_2:s] = score256_2[:(s-offset_2), :(s-offset_2)]
        overlay256 = np.ones_like(score256_2)*100
        overlay256[offset_2:s, offset_2:s] += 100
        score256 = (score256_1+new_score256_2)/overlay256
        color_block256 = (cmap(score256)*255)[:,:,:3].astype(np.uint8)
        region256_hm = cv2.addWeighted(color_block256, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
        Image.fromarray(region256_hm).save(os.path.join(output_dir, '%s_256[%s].png' % (fname, i)))
    
    for j in range(6):
        score4k_1 = concat_scores4k(a4k_1[j], size=(s,)*2)
        score4k_2 = concat_scores4k(a4k_2[j], size=(s,)*2)
        score4k_3 = concat_scores4k(a4k_3[j], size=(s,)*2)
        score4k_4 = concat_scores4k(a4k_4[j], size=(s,)*2)

        new_score4k_2 = np.zeros_like(score4k_2)
        new_score4k_2[offset_2:s, offset_2:s] = score4k_2[:(s-offset_2), :(s-offset_2)]
        new_score4k_3 = np.zeros_like(score4k_3)
        new_score4k_3[offset_3:s, offset_3:s] = score4k_3[:(s-offset_3), :(s-offset_3)]
        new_score4k_4 = np.zeros_like(score4k_4)
        new_score4k_4[offset_4:s, offset_4:s] = score4k_4[:(s-offset_4), :(s-offset_4)]

        overlay4k = np.ones_like(score4k_2)*100
        overlay4k[offset_2:s, offset_2:s] += 100
        overlay4k[offset_3:s, offset_3:s] += 100
        overlay4k[offset_4:s, offset_4:s] += 100
        score4k = (score4k_1+new_score4k_2+new_score4k_3+new_score4k_4)/overlay4k

        for i in range(6):
            score256_1 = concat_scores256(a256_1[:,i,:,:], size=(s//16,)*2)
            score256_2 = concat_scores256(a256_2[:,i,:,:], size=(s//16,)*2)
            new_score256_2 = np.zeros_like(score256_2)
            new_score256_2[offset_2:s, offset_2:s] = score256_2[:(s-offset_2), :(s-offset_2)]
            overlay256 = np.ones_like(score256_2)*100*2
            overlay256[offset_2:s, offset_2:s] += 100*2
            score256 = (score256_1+new_score256_2)*2/overlay256

            factorize = lambda data: (data - np.min(data)) / (np.max(data) - np.min(data))
            score = (score4k*overlay4k+score256*overlay256)/(overlay4k+overlay256) #factorize(score256*score4k)
            color_block = (cmap(score)*255)[:,:,:3].astype(np.uint8)
            region_hm = cv2.addWeighted(color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            Image.fromarray(region_hm).save(os.path.join(output_dir, '%s_factorized_4k[%s]_256[%s].png' % (fname, j, i)))
            
    return


def create_hierarchical_heatmaps_concat(region, model256, model4k, output_dir, fname,
                             offset=128, scale=4, alpha=0.5, cmap = plt.get_cmap('coolwarm')):
    r"""
    Creates hierarchical heatmaps (With Raw H&E + ViT-256 + ViT-4K + Blended Heatmaps concatenated for easy comparison)
    
    Args:
    - region (PIL.Image):       4096 x 4096 Image 
    - model256 (torch.nn):      256-Level ViT 
    - model4k (torch.nn):       4096-Level ViT 
    - output_dir (str):         Save directory / subdirectory
    - fname (str):              Naming structure of files
    - offset (int):             How much to offset (from top-left corner with zero-padding) the region by for blending 
    - scale (int):              How much to scale the output image by 
    - alpha (float):            Image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): Colormap for creating heatmaps
    
    Returns:
    - None
    """
    
    region2 = add_margin(region.crop((128,128,4096,4096)), 
                     top=0, left=0, bottom=128, right=128, color=(255,255,255))
    region3 = add_margin(region.crop((128*2,128*2,4096,4096)), 
                     top=0, left=0, bottom=128*2, right=128*2, color=(255,255,255))
    region4 = add_margin(region.crop((128*3,128*3,4096,4096)), 
                     top=0, left=0, bottom=128*4, right=128*4, color=(255,255,255))
    
    b256_1, a256_1, a4k_1 = get_region_attention_scores(region, model256, model4k, scale)
    
    b256_2, a256_2, a4k_2 = get_region_attention_scores(region2, model256, model4k, scale)
    b256_3, a256_3, a4k_3 = get_region_attention_scores(region3, model256, model4k, scale)
    b256_4, a256_4, a4k_4 = get_region_attention_scores(region4, model256, model4k, scale)
    offset_2 = (offset*1)//scale
    offset_3 = (offset*2)//scale
    offset_4 = (offset*3)//scale
    s = 4096//scale
    save_region = np.array(region.resize((s, s)))

    for j in range(6):
        score4k_1 = concat_scores4k(a4k_1[j], size=(s,)*2)
        score4k_2 = concat_scores4k(a4k_2[j], size=(s,)*2)
        score4k_3 = concat_scores4k(a4k_3[j], size=(s,)*2)
        score4k_4 = concat_scores4k(a4k_4[j], size=(s,)*2)

        new_score4k_2 = np.zeros_like(score4k_2)
        new_score4k_2[offset_2:s, offset_2:s] = score4k_2[:(s-offset_2), :(s-offset_2)]
        new_score4k_3 = np.zeros_like(score4k_3)
        new_score4k_3[offset_3:s, offset_3:s] = score4k_3[:(s-offset_3), :(s-offset_3)]
        new_score4k_4 = np.zeros_like(score4k_4)
        new_score4k_4[offset_4:s, offset_4:s] = score4k_4[:(s-offset_4), :(s-offset_4)]

        overlay4k = np.ones_like(score4k_2)*100
        overlay4k[offset_2:s, offset_2:s] += 100
        overlay4k[offset_3:s, offset_3:s] += 100
        overlay4k[offset_4:s, offset_4:s] += 100
        score4k = (score4k_1+new_score4k_2+new_score4k_3+new_score4k_4)/overlay4k
        
        color_block4k = (cmap(score4k_1/100)*255)[:,:,:3].astype(np.uint8)
        region4k_hm = cv2.addWeighted(color_block4k, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
        
        for i in range(6):
            score256_1 = concat_scores256(a256_1[:,i,:,:], size=(s//16,)*2)
            score256_2 = concat_scores256(a256_2[:,i,:,:], size=(s//16,)*2)
            new_score256_2 = np.zeros_like(score256_2)
            new_score256_2[offset_2:s, offset_2:s] = score256_2[:(s-offset_2), :(s-offset_2)]
            overlay256 = np.ones_like(score256_2)*100*2
            overlay256[offset_2:s, offset_2:s] += 100*2
            score256 = (score256_1+new_score256_2)*2/overlay256
            
            color_block256 = (cmap(score256)*255)[:,:,:3].astype(np.uint8)
            region256_hm = cv2.addWeighted(color_block256, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
        
            factorize = lambda data: (data - np.min(data)) / (np.max(data) - np.min(data))
            score = (score4k*overlay4k+score256*overlay256)/(overlay4k+overlay256) #factorize(score256*score4k)
            color_block = (cmap(score)*255)[:,:,:3].astype(np.uint8)
            region_hm = cv2.addWeighted(color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            
            pad = 100
            canvas = Image.new('RGB', (s*2+pad,)*2, (255,)*3)
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.truetype("arial.ttf", 50)
            draw.text((1024*0.5-pad*2, pad//4), "ViT-256 (Head: %d)" % i, (0, 0, 0), font=font)
            canvas = canvas.rotate(90)
            draw = ImageDraw.Draw(canvas)
            draw.text((1024*1.5-pad, pad//4), "ViT-4K (Head: %d)" % j, (0, 0, 0), font=font)
            canvas.paste(Image.fromarray(save_region), (pad,pad))
            canvas.paste(Image.fromarray(region4k_hm), (1024+pad,pad))
            canvas.paste(Image.fromarray(region256_hm), (pad,1024+pad))
            canvas.paste(Image.fromarray(region_hm), (s+pad,s+pad))
            canvas.save(os.path.join(output_dir, '%s_4k[%s]_256[%s].png' % (fname, j, i)))

    return


def create_hierarchical_heatmaps_concat_select(region, model256, model4k, output_dir, fname,
                             offset=128, scale=4, alpha=0.5, cmap = plt.get_cmap('coolwarm')):
    r"""
    Creates hierarchical heatmaps (With Raw H&E + ViT-256 + ViT-4K + Blended Heatmaps concatenated for easy comparison)

    Note that only select attention heads are used.
    
    Args:
    - region (PIL.Image):       4096 x 4096 Image 
    - model256 (torch.nn):      256-Level ViT 
    - model4k (torch.nn):       4096-Level ViT 
    - output_dir (str):         Save directory / subdirectory
    - fname (str):              Naming structure of files
    - offset (int):             How much to offset (from top-left corner with zero-padding) the region by for blending 
    - scale (int):              How much to scale the output image by 
    - alpha (float):            Image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): Colormap for creating heatmaps
    
    Returns:
    - None
    """
    
    region2 = add_margin(region.crop((128,128,4096,4096)), 
                     top=0, left=0, bottom=128, right=128, color=(255,255,255))
    region3 = add_margin(region.crop((128*2,128*2,4096,4096)), 
                     top=0, left=0, bottom=128*2, right=128*2, color=(255,255,255))
    region4 = add_margin(region.crop((128*3,128*3,4096,4096)), 
                     top=0, left=0, bottom=128*4, right=128*4, color=(255,255,255))
    
    b256_1, a256_1, a4k_1 = get_region_attention_scores(region, model256, model4k, scale)
    
    b256_2, a256_2, a4k_2 = get_region_attention_scores(region2, model256, model4k, scale)
    b256_3, a256_3, a4k_3 = get_region_attention_scores(region3, model256, model4k, scale)
    b256_4, a256_4, a4k_4 = get_region_attention_scores(region4, model256, model4k, scale)
    offset_2 = (offset*1)//scale
    offset_3 = (offset*2)//scale
    offset_4 = (offset*3)//scale
    s = 4096//scale
    save_region = np.array(region.resize((s, s)))
    
    canvas = [[Image.fromarray(save_region), None, None], [None, None, None]]
    for idx_4k, j in enumerate([0,5]):
        score4k_1 = concat_scores4k(a4k_1[j], size=(s,)*2)
        score4k_2 = concat_scores4k(a4k_2[j], size=(s,)*2)
        score4k_3 = concat_scores4k(a4k_3[j], size=(s,)*2)
        score4k_4 = concat_scores4k(a4k_4[j], size=(s,)*2)

        new_score4k_2 = np.zeros_like(score4k_2)
        new_score4k_2[offset_2:s, offset_2:s] = score4k_2[:(s-offset_2), :(s-offset_2)]
        new_score4k_3 = np.zeros_like(score4k_3)
        new_score4k_3[offset_3:s, offset_3:s] = score4k_3[:(s-offset_3), :(s-offset_3)]
        new_score4k_4 = np.zeros_like(score4k_4)
        new_score4k_4[offset_4:s, offset_4:s] = score4k_4[:(s-offset_4), :(s-offset_4)]

        overlay4k = np.ones_like(score4k_2)*100
        overlay4k[offset_2:s, offset_2:s] += 100
        overlay4k[offset_3:s, offset_3:s] += 100
        overlay4k[offset_4:s, offset_4:s] += 100
        score4k = (score4k_1+new_score4k_2+new_score4k_3+new_score4k_4)/overlay4k
        
        color_block4k = (cmap(score4k_1/100)*255)[:,:,:3].astype(np.uint8)
        region4k_hm = cv2.addWeighted(color_block4k, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
        canvas[0][idx_4k+1] = Image.fromarray(region4k_hm)
        
        for idx_256, i in enumerate([2]):
            score256_1 = concat_scores256(a256_1[:,i,:,:], size=(s//16,)*2)
            score256_2 = concat_scores256(a256_2[:,i,:,:], size=(s//16,)*2)
            new_score256_2 = np.zeros_like(score256_2)
            new_score256_2[offset_2:s, offset_2:s] = score256_2[:(s-offset_2), :(s-offset_2)]
            overlay256 = np.ones_like(score256_2)*100*2
            overlay256[offset_2:s, offset_2:s] += 100*2
            score256 = (score256_1+new_score256_2)*2/overlay256
            
            color_block256 = (cmap(score256)*255)[:,:,:3].astype(np.uint8)
            region256_hm = cv2.addWeighted(color_block256, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            canvas[idx_256+1][0] = Image.fromarray(region256_hm)
            
            factorize = lambda data: (data - np.min(data)) / (np.max(data) - np.min(data))
            score = (score4k*overlay4k+score256*overlay256)/(overlay4k+overlay256) #factorize(score256*score4k)
            color_block = (cmap(score)*255)[:,:,:3].astype(np.uint8)
            region_hm = cv2.addWeighted(color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            canvas[idx_256+1][idx_4k+1] = Image.fromarray(region_hm)
            
    canvas = getConcatImage([getConcatImage(row) for row in canvas], how='vertical')
    canvas.save(os.path.join(output_dir, '%s_heatmap.png' % (fname)))
    return