# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from skimage.transform import resize
from skimage.exposure import equalize_hist
import os 
import torch
import pydicom

'''
Dataloader for the Kaggle Pneumonia dataset
'''
class RSNADataset(Dataset):
    def __init__(self, num_samples, img_dimension, num_samples_U=0, num_samples_L=0, img_hist_eq=True, data_path='None', KAGGLE=False,):
        '''
        Dataset class for the Kaggle X-ray challenge. Num_samples_U and num_samples_L are used to keep track of how many examples have been
        loaded, and then load from the index, so the labelled, unlabelled and test dataset is not overlapping.
        
        Parameters:
            num_samples:          number of samples to include
            img_dimension:        the dimension the images are resized to [height, width]
            num_samples_U:        number of unlabelled training examples 
            num_samples_L:        number of labelled observations
            img_hist_eq:          whether to perfrom histogram equalization 
            data_path:            path to data if stores locally 
            KAGGLE:               if the running the model in a Kaggle notebook
        Output:
            image:                list with X-ray image
            labels:               labels of the image
        '''
        if KAGGLE:
            self.kaggleLabelPath = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv'
            self.kaggleDataPath = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/'

    
        # load data
        self.images, self.labels = self.load_data(num_samples, img_dimension, num_samples_U, num_samples_L, img_hist_eq, data_path, KAGGLE)
        print('loaded', len(self.images), 'samples')
        
        # index for each class for balancing
        self.idx_0 = np.where(self.labels == 0)[0]
        self.idx_1 = np.where(self.labels == 1)[0]
        if not len(self.idx_0) == len(self.idx_1):
            raise Exception('classes were not properly balanced')
    
    def __len__(self):
        return len(self.idx_0)
    
    def __getitem__(self, idx):
        obs_idx_0 = self.idx_0[idx]
        obs_idx_1 = self.idx_1[idx]
        return [self.images[obs_idx_0], self.images[obs_idx_1]], [self.labels[obs_idx_0], self.labels[obs_idx_1]]
    
    '''
    helper functions for loading the data
    '''
    def load_data(self, num_samples, img_dimension, num_samples_U, num_samples_L, img_hist_eq, data_path, KAGGLE):    
        # Load the labels
        mapping = {'Normal': 0, 'Lung Opacity': 1, 'No Lung Opacity / Not Normal': 2}
        if KAGGLE:
            df = pd.read_csv(self.kaggleLabelPath)
        else:
            df = pd.read_csv(os.path.join(data_path,'stage_2_detailed_class_info.csv'))
        df.rename(columns={'class': 'Target'}, inplace=True)
        df = df.replace({'Target': mapping})
        # keep only unique persons 
        unq, idx = np.unique(df['patientId'], return_index = True)
        df = df.iloc[idx]
        df = df.reset_index(drop=True)
        
        # Use to ballance the classes
        idx_0 = np.where(df['Target'] == 0)[0]
        idx_1 = np.where(df['Target'] == 1)[0]
        
        images = []
        labels = []
        
        # Load the images for class 0
        print('Loading class 0..')
        for i in range(num_samples_U//2+num_samples_L//2, num_samples_U//2+num_samples_L//2+num_samples//2):
            if i%100 == 0:
                print("loaded: ", i)
            patientId = df['patientId'][idx_0[i]]
            if KAGGLE:
                dcm_file = os.path.join(self.kaggleDataPath, patientId+'.dcm') # find the image-file corresponding to the patient id
            else:
                dcm_file = os.path.join(data_path,'stage_2_train_images', patientId+'.dcm') 
                
            dcm_data = pydicom.read_file(dcm_file) # Load the image 
            image_temp = resize(dcm_data.pixel_array, output_shape=img_dimension, mode='reflect')
            if img_hist_eq:
                image_temp = equalize_hist(image_temp)
            # convert to Tensor
            image_temp = torch.Tensor(image_temp).float().unsqueeze(axis=0)
            images.append(image_temp)
            labels.append(0)
       
        # Load the images for class 1
        print('Loading class 1..')
        for i in range(num_samples_U//2+num_samples_L//2, num_samples_U//2+num_samples_L//2+num_samples//2):
            if i%100 == 0:
                print("loaded: ", i)
            patientId = df['patientId'][idx_1[i]]
            if KAGGLE:
                dcm_file = dcm_file = os.path.join(self.kaggleDataPath, patientId+'.dcm') # find the image-file corresponding to the patient id
            else:
                dcm_file = os.path.join(data_path,'stage_2_train_images', patientId+'.dcm') 
                
            dcm_data = pydicom.read_file(dcm_file) # Load the image 
            image_temp = resize(dcm_data.pixel_array, output_shape=img_dimension, mode='reflect')
            if img_hist_eq:
                image_temp = equalize_hist(image_temp)
                
            image_temp = torch.Tensor(image_temp).float().unsqueeze(axis=0)
            images.append(image_temp)
            labels.append(1)
        
        # convert the labels to Tensor
        labels = torch.Tensor(labels).long()
        return images, labels


'''
Collate function for when batches are balanced
'''
def collate_fn_balanced_batch(batch):
    '''
    Takes ekstra dimension from drawing two observations each time the dataset is iterated, 
    and merging them into one batch dimention
    Input:
        batch:      List of tupples: [2, batch_size, [(channel, height, width), (channel, height, width)] ]
    Output:
        obs:        CT images [2*batch_size, channels, height, width]
        rois:       rois [2*batch_size, height, width]
    '''
    
    obs, labels = zip(*batch)
    # [batch_size, num_classes, [C, H, W]]
    obs_temp = []
    for batch_obs in obs:
        for class_obs in batch_obs:
            obs_temp.append(class_obs.float())
    obs = torch.stack(obs_temp)
    
    labels_temp = []
    for batch_labels in labels:
        for class_labels in batch_labels:
            labels_temp.append(class_labels)
    labels = torch.Tensor(labels_temp)
    
    return obs, labels

