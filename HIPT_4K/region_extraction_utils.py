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
Image.MAX_IMAGE_PIXELS = None
#import umap
#import umap.plot
from tqdm import tqdm

# Torch Dependencies
import torch
import torch.multiprocessing
import torchvision
from torchvision import transforms
from pl_bolts.models.self_supervised import resnets
from pl_bolts.utils.semi_supervised import Identity
from einops import rearrange, repeat
torch.multiprocessing.set_sharing_strategy('file_system')

# Dependencies from other packages / scripts
from attention_visualization_utils import get_vit256, get_vit4k, tensorbatch2im
from attention_visualization_utils import *

# Local Dependencies
import vision_transformer as vits
import vision_transformer4k as vits4k
from hipt4k_heatmap_utils import cmap_map


def prepare_img_tensor(img: torch.Tensor, patch_size=256):
	make_divisble = lambda l, patch_size: (l - (l % patch_size))
	b, c, w, h = img.shape
	load_size = make_divisble(w, patch_size), make_divisble(h, patch_size)
	w_4k, h_4k = w // patch_size, h // patch_size
	return transforms.CenterCrop(load_size)(img), w_4k, h_4k


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


def eval_transforms(is_imagenet=False, patch_size=256):
	"""
	"""
	if is_imagenet:
		mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
	else:
		mean, std = (0.5,0.5,0.5), (0.5,0.5,0.5)
	eval_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)])
	return eval_t


def concat_scores256(attns, w_256, h_256, size=(256,256)):
	r"""
	"""
	rank = lambda v: rankdata(v)*100/len(v)
	color_block = [rank(attn.flatten()).reshape(size) for attn in attns]
	color_hm = np.concatenate([
		np.concatenate(color_block[i:(i+h_256)], axis=1)
		for i in range(0,h_256*w_256,h_256)
	])
	return color_hm



class HIPT_4K(torch.nn.Module):
	"""
	HIPT Model (ViT-4K) for encoding non-square images (with [256 x 256] patch tokens), with 
	[256 x 256] patch tokens encoded via ViT-256 using [16 x 16] patch tokens.
	"""
	def __init__(self, 
		model256_path: str = '../Checkpoints/vit256_small_dino.pth',
		model4k_path: str = '../Checkpoints/vit4k_xs_dino.pth', 
		device256=torch.device('cuda:0'), 
		device4k=torch.device('cuda:1'),
		patch_filter_params: dict = {'isWhitePatch': {'satThresh': 5}, 'isBlackPatch': {'rgbThresh': 40}}):

		super().__init__()
		self.model256 = get_vit256(pretrained_weights=model256_path).to(device256)
		self.model4k = get_vit4k(pretrained_weights=model4k_path).to(device4k)
		self.device256 = device256
		self.device4k = device4k
		self.patch_filter_params = patch_filter_params
	
	def forward(self, x):
		"""
		Forward pass of HIPT (given an image tensor x), outputting the [CLS] token from ViT-4K.
		1. x is center-cropped such that the W / H is divisible by the patch token size in ViT-4K (e.g. - 256 x 256).
		2. x then gets unfolded into a "batch" of [256 x 256] images.
		3. A pretrained ViT-256 model extracts the CLS token from each [256 x 256] image in the batch.
		4. These batch-of-features are then reshaped into a 2D feature grid (of width "w_256" and height "h_256".)
		5. This feature grid is then used as the input to ViT-4K, outputting [CLS]_4K.
		
		Args:
			- x (torch.Tensor): [1 x C x W' x H'] image tensor.
		
		Return:
			- features_cls4k (torch.Tensor): [1 x 192] cls token (d_4k = 192 by default).
		"""
		batch_256, w_256, h_256 = self.prepare_img_tensor(x)                    # 1. [1 x 3 x W x H] 
		batch_256 = batch_256.unfold(2, 256, 256).unfold(3, 256, 256)           # 2. [1 x 3 x w_256 x h_256 x 256 x 256] 
		batch_256 = rearrange(batch_256, 'b c p1 p2 w h -> (b p1 p2) c w h')    # 2. [B x 3 x 256 x 256], where B = (1*w_256*h_256)
		
																			  
		features_cls256 = []
		for mini_bs in range(0, batch_256.shape[0], 256):                       # 3. B may be too large for ViT-256. We further take minibatches of 256.
			minibatch_256 = batch_256[mini_bs:mini_bs+256].to(self.device256, non_blocking=True)
			features_cls256.append(self.model256(minibatch_256).detach().cpu()) # 3. Extracting ViT-256 features from [256 x 3 x 256 x 256] image batches.

		features_cls256 = torch.vstack(features_cls256)                         # 3. [B x 384], where 384 == dim of ViT-256 [ClS] token.
		features_cls256 = features_cls256.reshape(w_256, h_256, 384).transpose(0,1).transpose(0,2).unsqueeze(dim=0) 
		features_cls256 = features_cls256.to(self.device4k, non_blocking=True)  # 4. [1 x 384 x w_256 x h_256]
		features_cls4k = self.model4k.forward(features_cls256)                  # 5. [1 x 192], where 192 == dim of ViT-4K [ClS] token.
		return features_cls4k
	
	
	def forward_asset_dict(self, x: torch.Tensor):
		"""
		Forward pass of HIPT (given an image tensor x), with certain intermediate representations saved in 
		a dictionary (that is to be stored in a H5 file). See walkthrough of how the model works above.
		
		Args:
			- x (torch.Tensor): [1 x C x W' x H'] image tensor.
		
		Return:
			- asset_dict (dict): Dictionary of intermediate feature representations of HIPT and other metadata.
				- features_cls256 (np.array): [B x 384] extracted ViT-256 cls tokens
				- features_mean256 (np.array): [1 x 384] mean ViT-256 cls token (exluding non-tissue patches)
				- contains_tissue_256 (np.array): [B,]-dim array with either 1 (tissue-containing) of 0 (white space) for each corresponding image in batch_256.
				- features_4k (np.array): [1 x 192] extracted ViT-4K cls token.
				- features_4k (np.array): [1 x 576] feature vector (concatenating mean ViT-256 + ViT-4K cls tokens)
	
		"""
		batch_256, w_256, h_256 = self.prepare_img_tensor(x)
		batch_256 = batch_256.unfold(2, 256, 256).unfold(3, 256, 256)
		batch_256 = rearrange(batch_256, 'b c p1 p2 w h -> (b p1 p2) c w h')
		
		features_cls256 = []
		for mini_bs in range(0, batch_256.shape[0], 256):
			minibatch_256 = batch_256[mini_bs:mini_bs+256].to(self.device256, non_blocking=True)
			features_cls256.append(self.model256(minibatch_256).detach().cpu())

		features_cls256 = torch.vstack(features_cls256)
		if self.patch_filter_params != None:
			### Creates a [B,]-dim np.array with either 1 (tissue-containing) of 0 (white space) for each corresponding image in batch_256.
			contains_tissue_256 = self.filter_tissue(batch_256)
			### Takes mean of ViT-256 features with only tissue-containing patches.
			features_mean256 = features_cls256[np.where(contains_tissue_256)].mean(dim=0).unsqueeze(dim=0)
		else:
			features_mean256 = features_cls256.mean(dim=0).unsqueeze(dim=0)

		features_grid256 = features_cls256.reshape(w_256, h_256, 384).transpose(0,1).transpose(0,2).unsqueeze(dim=0)
		features_grid256 = features_grid256.to(self.device4k, non_blocking=True)
		features_cls4k = self.model4k.forward(features_grid256).detach().cpu()
		features_mean256_cls4k = torch.cat([features_mean256, features_cls4k], dim=1)
		
		asset_dict = {
			'features_cls256': features_cls256.numpy(),
			'features_mean256': features_mean256.numpy(),
			'contains_tissue_256': contains_tissue_256,
			'features_cls4k': features_cls4k.numpy(),
			'features_mean256_cls4k': features_mean256_cls4k.numpy()
		}
		return asset_dict


	def _get_region_attention_scores(self, region, scale=1):
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
		eval_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		x = eval_transforms()(region).unsqueeze(dim=0)

		batch_256, w_256, h_256 = self.prepare_img_tensor(x)
		batch_256 = batch_256.unfold(2, 256, 256).unfold(3, 256, 256)
		batch_256 = rearrange(batch_256, 'b c p1 p2 w h -> (b p1 p2) c w h')
		batch_256 = batch_256.to(self.device256, non_blocking=True)
		features_cls256 = self.model256(batch_256)

		attention_256 = self.model256.get_last_selfattention(batch_256)
		nh = attention_256.shape[1] # number of head
		attention_256 = attention_256[:, :, 0, 1:].reshape(256, nh, -1)
		attention_256 = attention_256.reshape(w_256*h_256, nh, 16, 16)
		attention_256 = nn.functional.interpolate(attention_256, scale_factor=int(16/scale), mode="nearest").cpu().numpy()

		features_grid256 = features_cls256.reshape(w_256, h_256, 384).transpose(0,1).transpose(0,2).unsqueeze(dim=0)
		features_grid256 = features_grid256.to(self.device4k, non_blocking=True)
		features_cls4k = self.model4k.forward(features_grid256).detach().cpu()

		attention_4k = self.model4k.get_last_selfattention(features_grid256)
		nh = attention_4k.shape[1] # number of head
		attention_4k = attention_4k[0, :, 0, 1:].reshape(nh, -1)
		attention_4k = attention_4k.reshape(nh, w_256, h_256)
		attention_4k = nn.functional.interpolate(attention_4k.unsqueeze(0), scale_factor=int(256/scale), mode="nearest")[0].cpu().numpy()

		if scale != 1:
			batch_256 = nn.functional.interpolate(batch_256, scale_factor=(1/scale), mode="nearest")

		return tensorbatch2im(batch_256), attention_256, attention_4k


	def get_region_heatmaps(self, x, offset=128, scale=4, alpha=0.5, cmap = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.jet), threshold=None):
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
		region = Image.fromarray(tensorbatch2im(x)[0])
		w, h = region.size

		region2 = add_margin(region.crop((128,128,w,h)), 
						 top=0, left=0, bottom=128, right=128, color=(255,255,255))
		region3 = add_margin(region.crop((128*2,128*2,w,h)), 
						 top=0, left=0, bottom=128*2, right=128*2, color=(255,255,255))
		region4 = add_margin(region.crop((128*3,128*3,w,h)), 
						 top=0, left=0, bottom=128*4, right=128*4, color=(255,255,255))
		
		b256_1, a256_1, a4k_1 = self._get_region_attention_scores(region, scale)
		b256_2, a256_2, a4k_2 = self._get_region_attention_scores(region, scale)
		b256_3, a256_3, a4k_3 = self._get_region_attention_scores(region, scale)
		b256_4, a256_4, a4k_4 = self._get_region_attention_scores(region, scale)
		offset_2 = (offset*1)//scale
		offset_3 = (offset*2)//scale
		offset_4 = (offset*3)//scale
		w_s, h_s = w//scale, h//scale
		w_256, h_256 = w//256, h//256
		save_region = np.array(region.resize((w_s, h_s)))
		
		if threshold != None:
			for i in range(6):
				score256_1 = concat_scores256(a256_1[:,i,:,:], w_256, h_256, size=(w_s//w_256,h_s//h_256))
				score256_2 = concat_scores256(a256_2[:,i,:,:], w_256, h_256, size=(w_s//w_256,h_s//h_256))
				new_score256_2 = np.zeros_like(score256_2)
				new_score256_2[offset_2:w_s, offset_2:h_s] = score256_2[:(w_s-offset_2), :(h_s-offset_2)]
				overlay256 = np.ones_like(score256_2)*100
				overlay256[offset_2:w_s, offset_2:h_s] += 100
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
				score4k_1 = concat_scores4k(a4k_1[j], size=(h_s,w_s))
				score4k = score4k_1 / 100
				color_block4k = (cmap(score4k)*255)[:,:,:3].astype(np.uint8)
				region4k_hm = cv2.addWeighted(color_block4k, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
				Image.fromarray(region4k_hm).save(os.path.join(output_dir, '%s_4k[%s].png' % (fname, j)))
		
		hm4k, hm256, hm4k_256 = [], [], []
		for j in range(6):
			score4k_1 = concat_scores4k(a4k_1[j], size=(h_s,w_s))
			score4k_2 = concat_scores4k(a4k_2[j], size=(h_s,w_s))
			score4k_3 = concat_scores4k(a4k_3[j], size=(h_s,w_s))
			score4k_4 = concat_scores4k(a4k_4[j], size=(h_s,w_s))
			new_score4k_2 = np.zeros_like(score4k_2)
			new_score4k_2[offset_2:h_s, offset_2:w_s] = score4k_2[:(h_s-offset_2), :(w_s-offset_2)]
			new_score4k_3 = np.zeros_like(score4k_3)
			new_score4k_3[offset_3:h_s, offset_3:w_s] = score4k_3[:(h_s-offset_3), :(w_s-offset_3)]
			new_score4k_4 = np.zeros_like(score4k_4)
			new_score4k_4[offset_4:h_s, offset_4:w_s] = score4k_4[:(h_s-offset_4), :(w_s-offset_4)]

			overlay4k = np.ones_like(score4k_2)*100
			overlay4k[offset_2:h_s, offset_2:w_s] += 100
			overlay4k[offset_3:h_s, offset_3:w_s] += 100
			overlay4k[offset_4:h_s, offset_4:w_s] += 100
			score4k = (score4k_1+new_score4k_2+new_score4k_3+new_score4k_4)/overlay4k

			color_block4k = (cmap(score4k)*255)[:,:,:3].astype(np.uint8)
			region4k_hm = cv2.addWeighted(color_block4k, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
			hm4k.append(Image.fromarray(region4k_hm))


		for i in range(6):
			score256_1 = concat_scores256(a256_1[:,i,:,:], h_256, w_256, size=(256, 256))
			score256_2 = concat_scores256(a256_2[:,i,:,:], h_256, w_256, size=(256, 256))
			new_score256_2 = np.zeros_like(score256_2)
			new_score256_2[offset_2:h_s, offset_2:w_s] = score256_2[:(h_s-offset_2), :(w_s-offset_2)]
			overlay256 = np.ones_like(score256_2)*100
			overlay256[offset_2:h_s, offset_2:w_s] += 100
			score256 = (score256_1+new_score256_2)/overlay256
			color_block256 = (cmap(score256)*255)[:,:,:3].astype(np.uint8)
			region256_hm = cv2.addWeighted(color_block256, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
			hm256.append(Image.fromarray(region256_hm))
		
		for j in range(6):
			score4k_1 = concat_scores4k(a4k_1[j], size=(h_s,w_s))
			score4k_2 = concat_scores4k(a4k_2[j], size=(h_s,w_s))
			score4k_3 = concat_scores4k(a4k_3[j], size=(h_s,w_s))
			score4k_4 = concat_scores4k(a4k_4[j], size=(h_s,w_s))

			new_score4k_2 = np.zeros_like(score4k_2)
			new_score4k_2[offset_2:h_s, offset_2:w_s] = score4k_2[:(h_s-offset_2), :(w_s-offset_2)]
			new_score4k_3 = np.zeros_like(score4k_3)
			new_score4k_3[offset_3:h_s, offset_3:w_s] = score4k_3[:(h_s-offset_3), :(w_s-offset_3)]
			new_score4k_4 = np.zeros_like(score4k_4)
			new_score4k_4[offset_4:h_s, offset_4:w_s] = score4k_4[:(h_s-offset_4), :(w_s-offset_4)]

			overlay4k = np.ones_like(score4k_2)*100
			overlay4k[offset_2:h_s, offset_2:w_s] += 100
			overlay4k[offset_3:h_s, offset_3:w_s] += 100
			overlay4k[offset_4:h_s, offset_4:w_s] += 100
			score4k = (score4k_1+new_score4k_2+new_score4k_3+new_score4k_4)/overlay4k

			for i in range(6):
				score256_1 = concat_scores256(a256_1[:,i,:,:], h_256, w_256, size=(256, 256))
				score256_2 = concat_scores256(a256_2[:,i,:,:], h_256, w_256, size=(256, 256))
				new_score256_2 = np.zeros_like(score256_2)
				new_score256_2[offset_2:h_s, offset_2:w_s] = score256_2[:(h_s-offset_2), :(w_s-offset_2)]
				overlay256 = np.ones_like(score256_2)*100
				overlay256[offset_2:h_s, offset_2:w_s] += 100
				score256 = (score256_1+new_score256_2)/overlay256

				factorize = lambda data: (data - np.min(data)) / (np.max(data) - np.min(data))
				score = (score4k*overlay4k+score256*overlay256)/(overlay4k+overlay256) #factorize(score256*score4k)
				color_block = (cmap(score)*255)[:,:,:3].astype(np.uint8)
				region4k_256_hm = cv2.addWeighted(color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
				hm4k_256.append(Image.fromarray(region4k_256_hm))

		return hm4k, hm256, hm4k_256				

	def prepare_img_tensor(self, img: torch.Tensor, patch_size=256):
		"""
		Helper function that takes a non-square image tensor, and takes a center crop s.t. the width / height
		are divisible by 256.
		
		(Note: "_256" for w / h is should technically be renamed as "_ps", but may not be easier to read.
		Until I need to make HIPT with patch_sizes != 256, keeping the naming convention as-is.)
		
		Args:
			- img (torch.Tensor): [1 x C x W' x H'] image tensor.
			- patch_size (int): Desired patch size to evenly subdivide the image.
		
		Return:
			- img_new (torch.Tensor): [1 x C x W x H] image tensor, where W and H are divisble by patch_size.
			- w_256 (int): # of [256 x 256] patches of img_new's width (e.g. - W/256)
			- h_256 (int): # of [256 x 256] patches of img_new's height (e.g. - H/256)
		"""
		make_divisble = lambda l, patch_size: (l - (l % patch_size))
		b, c, w, h = img.shape
		load_size = make_divisble(w, patch_size), make_divisble(h, patch_size)
		w_256, h_256 = w // patch_size, h // patch_size
		img_new = transforms.CenterCrop(load_size)(img)
		return img_new, w_256, h_256

	
	def filter_tissue(self, batch_256: torch.Tensor):
		"""
		Helper function that filters each tissue patch in the batch as tissue-containing image (1) or white space (0).
		
		Args:
			- batch_256 (torch.Tensor): [B x C x 256 x 256] image tensor batch following unrolling the [1 x 3 x W x H] image tensor)
			into B [256 x 256 x 3] image patches).
			
		Return:
			- contains_tissue_256 (np.array): [B,]-dim array with either 1 (tissue-containing) of 0 (white space) for each corresponding image in batch_256.
		
		"""
		to_filter = np.array([filterPatch(img, patch_filter_params=self.patch_filter_params) 
							  for img in tensorbatch2im(batch_256)])
		contains_tissue_256 = 1-to_filter
		return contains_tissue_256