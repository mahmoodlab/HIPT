# Base Dependencies
import os
import pickle
import sys
j_ = os.path.join

# LinAlg / Stats / Plotting Dependencies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# Scikit-Learn Imports
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold

#Torch Imports
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
torch.multiprocessing.set_sharing_strategy('file_system')


def series_intersection(s1, s2):
    r"""
    Takes the intersection of two pandas.Series (pd.Series) objects.
    
    Args:
        - s1 (pd.Series): pd.Series object.
        - s2 (pd.Series): pd.Series object.
    Return:
        - pd.Series: Intersection of s1 and s2.
    """
    return pd.Series(list(set(s1) & set(s2)))


def save_embeddings_mean(save_pickle_fpath, dataset):
    r"""
    Saves+Pickle each WSI in a SlideEmbeddingDataset Object as the average of its instance-level embeddings
    
    Args:
        - save_fpath (str): Save filepath for the pickle object.
        - dataset (torch.utils.data.dataset): SlideEmbeddingDataset_WS object that iterates+loads each WSI in a folder
    
    Return:
        - None
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    embeddings, labels = [], []

    for batch, target in dataloader:
        with torch.no_grad():
            embeddings.append(batch.squeeze(dim=0).mean(dim=0).numpy())
            labels.append(target.numpy())
            
    embeddings = np.vstack(embeddings)
    labels = np.vstack(labels).squeeze()

    asset_dict = {'embeddings': embeddings, 'labels': labels}

    with open(save_pickle_fpath, 'wb') as handle:
        pickle.dump(asset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


class SlideEmbeddingSplitDataset(Dataset):
    r"""
    torch.utils.data.dataset object that iterates+loads each WSI from a split CSV file
    
    Args:
        - dataroot (str): Path to wsi_labels.csv.
        - tcga_csv (pd.DataFrame): Clinical CSV (as a pd.DataFrame object) for a TCGA Study
        - pt_path (str): Path to folder of saved instance-level feature embeddings for each WSI.
        - splits_csv (pd.DataFrame): DataFrame which contains slide_ids for train / val / test
        - label_col (str): Which column to use as labels in tcga_csv
        - label_dict (dict): Dictionary for categorizing labels
    Return:
        - None
    """
    def __init__(self, dataroot, tcga_csv, pt_path, splits_csv=None,
                 label_col='oncotree_code', label_dict={'LUSC':0, 'LUAD':1}):        
        self.csv = pd.read_csv(os.path.join(dataroot, 'tcga_wsi_labels.csv'))
        self.csv['slide_path'] = pt_path+self.csv['slide_id']
        self.csv = self.csv.set_index('slide_id', drop=True).drop(['Unnamed: 0'], axis=1)
        self.csv.index = self.csv.index.str[:-3]
        self.csv.index.name = None
        self.csv = self.csv.join(tcga_csv, how='inner')
        if splits_csv is not None:
            self.csv = self.csv.loc[series_intersection(splits_csv.dropna(), self.csv.index)]
            
        self.label_col = label_col
        self.label_dict = label_dict
        
        ### If using DINO Features, subset and save only the last 384-dim features.
        if 'dino_pt_patch_features' in pt_path:
            self.last_stage = True
        else:
            self.last_stage = False
            
    def __getitem__(self, index):
        x = torch.load(self.csv['slide_path'][index])
        if self.last_stage and x.shape[1] == 1536:
            x = x[:,(1536-384):1536]
        label = torch.Tensor([self.label_dict[self.csv[self.label_col][index]]]).to(torch.long)
        return x, label
    
    def __len__(self):
        return self.csv.shape[0]
    
    
def create_slide_embeddings(dataroot, saveroot, enc_name, study):
    r"""
    """
    
    path2csv = '../Weakly-Supervised-Subtyping/dataset_csv/'
    path2splits = '../Weakly-Supervised-Subtyping/splits/'

    splits_folder = j_(path2splits, '10foldcv_subtype', study)
    tcga_csv = pd.read_csv(j_(path2csv, f'{study}_subset.csv.zip'), index_col=2)['oncotree_code']
    tcga_csv.index = tcga_csv.index.str[:-4]
    tcga_csv.index.name = None
    
    save_embedding_dir = j_(saveroot, enc_name)
    os.makedirs(save_embedding_dir, exist_ok=True)

    if enc_name == 'vit256mean':
        pt_path = j_(dataroot, 'vit256mean_tcga_slide_embeddings')
    elif enc_name == 'vit16mean':
        extracted_dir = f'{study}/extracted_mag20x_patch256_fp/vits_tcga_pancancer_dino_pt_patch_features/'
        pt_path = j_(dataroot, extracted_dir)
    elif enc_name == 'resnet50mean':
        extracted_dir = f'{study}/extracted_mag20x_patch256_fp/resnet50_trunc_pt_patch_features/'
        pt_path = j_(dataroot, extracted_dir)

    if study == 'tcga_brca':
        label_dict={'IDC':0, 'ILC':1}
        tcga_csv = tcga_csv[tcga_csv.str.contains('IDC|ILC')]
    elif study == 'tcga_kidney':
        label_dict={'CCRCC':0, 'PRCC':1, 'CHRCC': 2}
    elif study == 'tcga_lung':
        label_dict={'LUSC':0, 'LUAD':1}
    
    for i in tqdm(range(10)):
        splits_csv = pd.read_csv(os.path.join(splits_folder, f'splits_{i}.csv'), index_col=0)
        train = SlideEmbeddingSplitDataset(dataroot=dataroot, tcga_csv=tcga_csv, pt_path=pt_path,
                                   splits_csv=splits_csv['train'], label_dict=label_dict)
        test = SlideEmbeddingSplitDataset(dataroot=dataroot, tcga_csv=tcga_csv, pt_path=pt_path,
                                  splits_csv=splits_csv['test'], label_dict=label_dict)

        save_embeddings_mean(j_(save_embedding_dir, f'{study}_{enc_name}_class_split_train_{i}.pkl'), train)
        save_embeddings_mean(j_(save_embedding_dir, f'{study}_{enc_name}_class_split_test_{i}.pkl'), test)