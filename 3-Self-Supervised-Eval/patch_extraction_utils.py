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
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from pl_bolts.models.self_supervised import resnets
from pl_bolts.utils.semi_supervised import Identity
device = torch.device('cuda:0')
torch.multiprocessing.set_sharing_strategy('file_system')

# Model Architectures
from nn_encoder_arch.vision_transformer import vit_small
from nn_encoder_arch.resnet_trunc import resnet50_trunc_baseline


### Helper Functions for Normalization + Loading in pytorch_lightning SSL encoder (for SimCLR)
def eval_transforms(pretrained=False):
    if pretrained:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        mean, std = (0.5,0.5,0.5), (0.5,0.5,0.5)
    trnsfrms_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)])
    return trnsfrms_val


def torchvision_ssl_encoder(name: str, pretrained: bool = False, return_all_feature_maps: bool = False):
    pretrained_model = getattr(resnets, name)(pretrained=pretrained, return_all_feature_maps=return_all_feature_maps)
    pretrained_model.fc = Identity()
    return pretrained_model

### Wrapper Classes for loading in patch datasets for BreastPathQ + BCSS (CRC100K uses the ImageFolder Dataset Class)
class CSVDataset_BreastPathQ(Dataset):    
    def __init__(self, dataroot, csv_path, transforms_eval=eval_transforms()):
        self.csv = pd.read_csv(csv_path)
        self.csv['img_path'] = dataroot+self.csv['slide'].astype(str) + "_" + self.csv['rid'].astype(str) + '.tif'
        self.transforms = transforms_eval
        
    def __getitem__(self, index):
        img = Image.open(self.csv['img_path'][index])
        return self.transforms(img), self.csv['y'][index]
    
    def __len__(self):
        return self.csv.shape[0]


class CSVDataset_BCSS(Dataset):    
    def __init__(self, dataset_csv, is_train=1, transforms_eval=eval_transforms()):
        self.csv = dataset_csv
        self.csv = self.csv[self.csv['train']==is_train]
        self.transforms = transforms_eval   
        
    def __getitem__(self, index):
        img = Image.open(self.csv.index[index])
        return self.transforms(img), self.csv.iloc[index]['label']
    
    def __len__(self):
        return self.csv.shape[0]

### Functions for Loading + Saving + Visualizing Patch Embeddings
def save_embeddings(model, fname, dataloader, dataset=None, is_imagefolder=False, 
                    save_patches=False, sprite_dim=128, overwrite=False):

    if os.path.isfile('%s.pkl' % fname) and (overwrite == False):
        return None

    embeddings, labels = [], []
    patches = []

    for batch, target in tqdm(dataloader):
        if save_patches:
            for img in batch:
                patches.append(tensor2im(input_image=img).resize(sprite_dim))
        
        with torch.no_grad():
            batch = batch.to(device)
            embeddings.append(model(batch).detach().cpu().numpy())
            labels.append(target.numpy())
            
            
    embeddings = np.vstack(embeddings)
    labels = np.vstack(labels).squeeze()
    
    if is_imagefolder:
        id2label = dict(map(reversed, dataset.class_to_idx.items()))
        labels = np.array(list(map(id2label.get, labels.ravel())))

    asset_dict = {'embeddings': embeddings, 'labels': labels}
    if save_patches:
        asset_dict.update({'patches': patches})
    with open('%s.pkl' % (fname), 'wb') as handle:
        pickle.dump(asset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
def create_UMAP(library_path, save_path, dataset, enc_name, n=15, d=0.1):
    path = os.path.join(library_path, '%s_%s.pkl' % (dataset, enc_name))
    with open(path, 'rb') as handle:
        asset_dict = pickle.load(handle)
        embeddings, labels = asset_dict['embeddings'], asset_dict['labels']

        if 'crc100k' in dataset:
            labels[labels=='MUS'] = 'STR'

    mapper = umap.UMAP(n_neighbors=n, min_dist=d).fit(embeddings)
    fig = plt.figure(figsize=(10, 10), dpi=100)
    umap.plot.points(mapper, labels=labels, width=600, height=600)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '%s_%s_umap_n%d_d%0.2f.jpg' % (dataset, enc_name, n, d)))


def create_embeddings(embeddings_dir, enc_name, dataset, save_patches=False, sprite_dim=128, 
                      patch_datasets='path/to/patch/datasets', assets_dir ='./ckpts/',
                      disentangle=-1, stage=-1):
    print("Extracting Features for '%s' via '%s'" % (dataset, enc_name))
    if enc_name == 'resnet50_trunc':
        model = resnet50_trunc_baseline(pretrained=True)
        eval_t = eval_transforms(pretrained=True)
    elif 'dino' in enc_name:
        ckpt_path = os.path.join(assets_dir, enc_name+'.pt')
        assert os.path.isfile(ckpt_path)
        model = vit_small(patch_size=16)
        state_dict = torch.load(ckpt_path, map_location="cpu")['teacher']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        #print("Missing Keys:", missing_keys)
        #print("Unexpected Keys:", unexpected_keys)
        eval_t = eval_transforms(pretrained=False)
    elif 'simclr' in enc_name:
        ckpt_path = os.path.join(assets_dir, enc_name+'.pt')
        assert os.path.isfile(ckpt_path)
        model = torchvision_ssl_encoder('resnet50', pretrained=True)
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(ckpt_path), strict=False)
        #print("Missing Keys:", missing_keys)
        #print("Unexpected Keys:", unexpected_keys)
        eval_t = eval_transforms(pretrained=False)
    else:
        pass

    model = model.to(device)
    model.eval()

    if 'simclr' in enc_name or 'simsiam' in enc_name:
        _model = model
        model = lambda x: _model.forward(x)[0]
    elif 'dino' in enc_name:
        _model = model
        if stage == -1:
            model = _model
        else:
            model = lambda x: torch.cat([x[:, 0] for x in _model.get_intermediate_layers(x, stage)], dim=-1)

    if stage != -1:
        _stage = '_s%d' % stage
    else:
        _stage = ''
    
    if dataset == 'crc100k':
        ### Train
        dataroot = os.path.join(patch_datasets, 'NCT-CRC-HE-100K/')
        dataset = torchvision.datasets.ImageFolder(dataroot, transform=eval_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=4)
        fname = os.path.join(embeddings_dir, 'crc100k_train_%s%s' % (enc_name, _stage))
        save_embeddings(model=model, fname=fname, dataloader=dataloader, dataset=dataset,
                        save_patches=save_patches, sprite_dim=sprite_dim, is_imagefolder=True)
        
        ### Test
        dataroot = os.path.join(patch_datasets, 'CRC-VAL-HE-7K/')
        dataset = torchvision.datasets.ImageFolder(dataroot, transform=eval_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        fname = os.path.join(embeddings_dir, 'crc100k_val_%s%s' % (enc_name, _stage))
        save_embeddings(model=model, fname=fname, dataloader=dataloader, dataset=dataset,
                        save_patches=save_patches, sprite_dim=sprite_dim, is_imagefolder=True)

    elif dataset == 'crc100knonorm':
        ### Train
        dataroot = os.path.join(patch_datasets, 'NCT-CRC-HE-100K-NONORM/')
        dataset = torchvision.datasets.ImageFolder(dataroot, transform=eval_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=4)
        fname = os.path.join(embeddings_dir, 'crc100knonorm_train_%s%s' % (enc_name, _stage))
        save_embeddings(model=model, fname=fname, dataloader=dataloader, dataset=dataset,
                        save_patches=save_patches, sprite_dim=sprite_dim, is_imagefolder=True)
        
        ### Test
        dataroot = os.path.join(patch_datasets, 'CRC-VAL-HE-7K/')
        dataset = torchvision.datasets.ImageFolder(dataroot, transform=eval_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        fname = os.path.join(embeddings_dir, 'crc100knonorm_val_%s%s' % (enc_name, _stage))
        save_embeddings(model=model, fname=fname, dataloader=dataloader, dataset=dataset,
                        save_patches=save_patches, sprite_dim=sprite_dim, is_imagefolder=True)

    elif dataset == 'breastpathq':
        train_dataroot = os.path.join(patch_datasets, 'BreastPathQ/breastpathq/datasets/train/')
        val_dataroot = os.path.join(patch_datasets, 'BreastPathQ/breastpathq/datasets/validation/')
        train_csv = os.path.join(patch_datasets, 'BreastPathQ/breastpathq/datasets/train_labels.csv')
        val_csv = os.path.join(patch_datasets, 'BreastPathQ/breastpathq/datasets/val_labels.csv')

        train_dataset = CSVDataset_BreastPathQ(dataroot=train_dataroot, csv_path=train_csv, transforms_eval=eval_t)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
        val_dataset = CSVDataset_BreastPathQ(dataroot=val_dataroot, csv_path=val_csv, transforms_eval=eval_t)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
        
        train_fname = os.path.join(embeddings_dir, 'breastpathq_train_%s%s' % (enc_name, _stage))
        val_fname = os.path.join(embeddings_dir, 'breastpathq_val_%s%s' % (enc_name, _stage))
        save_embeddings(model=model, fname=train_fname, dataloader=train_dataloader, 
                        save_patches=save_patches, sprite_dim=sprite_dim)
        save_embeddings(model=model, fname=val_fname, dataloader=val_dataloader, 
                        save_patches=save_patches, sprite_dim=sprite_dim)

    
    elif dataset == 'bcss':
        dataroot = os.path.join(patch_datasets, 'BCSS/40x/patches/All/')
        csv_path = os.path.join(patch_datasets, 'BCSS/40x/patches/summary.csv')
        
        dataset_csv = pd.read_csv(csv_path, sep=' ')['filename,train'].str.split(',', expand=True).astype(int)
        dataset_csv.columns = ['label', 'train']
        dataset_csv = dataset_csv[dataset_csv['label'].isin([0,1,2,3])]
        dataset_csv.index = [os.path.join(dataroot, fname+'.png') for fname in dataset_csv.index]

        train_dataset = CSVDataset_BCSS(dataset_csv=dataset_csv, is_train=1, transforms_eval=eval_t)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)
        val_dataset = CSVDataset_BCSS(dataset_csv=dataset_csv, is_train=0, transforms_eval=eval_t)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

        train_fname = os.path.join(embeddings_dir, 'bcss_train_%s%s' % (enc_name, _stage))
        val_fname = os.path.join(embeddings_dir, 'bcss_val_%s%s' % (enc_name, _stage))
        save_embeddings(model=model, fname=train_fname, dataloader=train_dataloader, 
                        save_patches=save_patches, sprite_dim=sprite_dim)
        save_embeddings(model=model, fname=val_fname, dataloader=val_dataloader, 
                        save_patches=save_patches, sprite_dim=sprite_dim)