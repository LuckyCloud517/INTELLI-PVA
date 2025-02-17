# -*- coding: utf-8 -*-


import os
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
from utils.utils_generic import pre_process, custom_augment

# Function to generate a fully-supervised dataset
# maxlen: The maximum length of the waveform data
# savepath: Path to save the processed data
def fully_supervised_dataset(maxlen=300, savepath='./traindata/'):
    """
    Load and process the dataset for supervised learning.

    Parameters:
    - maxlen: Maximum length of the waveform data.
    - savepath: Directory to save processed data.
    - data_format: The format of data, where 2 indicates pressure and flow data.

    Returns:
    - x: Processed waveform data.
    - y: Labels corresponding to each data sample.
    - x_index: List of file paths corresponding to each data sample.
    """
    datapath = ''
    
    # Create the directory to save processed data if it does not exist
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
        
    n1 = list(os.path.join(datapath+'/1/',name) for name in os.listdir(datapath+'/1/'))
    n2 = list(os.path.join(datapath+'/2/',name) for name in os.listdir(datapath+'/2/'))
    n3 = list(os.path.join(datapath+'/3/',name) for name in os.listdir(datapath+'/3/'))
    n4 = list(os.path.join(datapath+'/4/',name) for name in os.listdir(datapath+'/4/'))
    n5 = list(os.path.join(datapath+'/0/',name) for name in os.listdir(datapath+'/0/'))
    
    # Load preprocessed data if already saved
    if os.path.exists(os.path.join(savepath, 'x_all.npy')) and os.path.exists(os.path.join(savepath, 'y_all.npy')) and os.path.exists(os.path.join(savepath, 'x_index.npy')):
        print('[I] load stored data...')
        x = np.load(os.path.join(savepath,'x_all.npy'))
        y = np.load(os.path.join(savepath,'y_all.npy'))
        x_index = np.load(os.path.join(savepath,'x_index.npy'))
    else:
        print('[I] Processing data and saving...')
        # Concatenate all file paths for processing
        x_index = n1 + n2 + n3 + n4 + n5
        # Assign labels for each type of data
        y = [1] * len(n1) + [2] * len(n2) + [3] * len(n3) + [4] * len(n4) + [0] * len(n5)
        
        x = []
        # Process each file in the dataset
        for item in tqdm(x_index):
            temp = loadmat(item)
            waveData = temp["waveData"]
            waveData = waveData.T
            # Swap columns: Pressure in the first column, Flow rate in the second column
            waveData_transposed = waveData[:,[1,0]]
            # Preprocess the waveform data and truncate to maxlen
            x.append(pre_process(waveData_transposed, length=maxlen))
        
        # Shuffle the data
        c = list(zip(x, y, x_index))
        np.random.shuffle(c)
        x[:], y[:], x_index[:] = zip(*c)
        
        # Save the processed data for later use
        np.save(os.path.join(savepath,'x_all.npy'), x)
        np.save(os.path.join(savepath,'y_all.npy'), y)
        np.save(os.path.join(savepath,'x_index.npy'), x_index)
        
    return x, y, x_index
        
        
#%%
# Function to generate a SimCLR dataset with labeled pairs for contrastive learning
# maxlen: The maximum length of the waveform data
# savepath: Path to save the processed data
# method: Augmentation method used for creating pairs ('Combinations' or others)
def simclr_dataset_with_label(maxlen=300, savepath = './traindata/', method='Combinations'):
    """
    Generate SimCLR-style dataset with pairs for contrastive learning.
    
    Parameters:
    - maxlen: Maximum length of the waveform data.
    - savepath: Directory to save processed dataset.
    - method: Augmentation method for generating pairs ('Combinations' or others).
    
    Returns:
    - ssl_ds_one: First augmented version of the dataset.
    - ssl_ds_two: Second augmented version of the dataset.
    """
    
    datapath = ''
    
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
        
    n1 = list(os.path.join(datapath+'/1/',name) for name in os.listdir(datapath+'/1/'))
    n2 = list(os.path.join(datapath+'/2/',name) for name in os.listdir(datapath+'/2/'))
    n3 = list(os.path.join(datapath+'/3/',name) for name in os.listdir(datapath+'/3/'))
    n4 = list(os.path.join(datapath+'/4/',name) for name in os.listdir(datapath+'/4/'))
    n5 = list(os.path.join(datapath+'/0/',name) for name in os.listdir(datapath+'/0/'))
    # Combine all file paths for processing
    n = n1 + n2 + n3 + n4 + n5
    
    # Load preprocessed data if already saved
    if os.path.exists(os.path.join(savepath, f'ssl_ds_one_{method}.npy')) and os.path.exists(os.path.join(savepath, f'ssl_ds_two_{method}.npy')):
        print('[I] load stored data...')
        ssl_ds_one = np.load(os.path.join(savepath, f'ssl_ds_one_{method}.npy'))
        ssl_ds_two = np.load(os.path.join(savepath, f'ssl_ds_two_{method}.npy'))

    else:
        print('[I] Processing data and saving...')

        np.random.seed(1234)    
        np.random.shuffle(n)
        
        ssl_ds_one, ssl_ds_two = [], []
        # Process each file and generate augmented pairs
        for item in tqdm(n):
            temp = loadmat(item)
            waveData = temp["waveData"]
            waveData = waveData.T
            
            # Swap columns: Pressure in the first column, Flow rate in the second column
            waveData_transposed = waveData[:,[1,0]] 
            
            # Apply augmentations for both pairs of data
            trans1 = custom_augment(waveData_transposed, method=method)
            trans2 = custom_augment(waveData_transposed, method=method)
            
            # Preprocess both augmented versions and truncate to maxlen
            one = pre_process(trans1, length = maxlen)
            ssl_ds_one.append(one)
        
            two = pre_process(trans2, length = maxlen)
            ssl_ds_two.append(two)
        
        # Reshape data for training
        ssl_ds_one = np.reshape(ssl_ds_one, (-1, maxlen, 2)).astype('float32')
        ssl_ds_two = np.reshape(ssl_ds_two, (-1, maxlen, 2)).astype('float32')
        
        # Save the processed data for later use
        np.save(os.path.join(savepath, f'ssl_ds_one_{method}.npy'), ssl_ds_one)
        np.save(os.path.join(savepath, f'ssl_ds_two_{method}.npy'), ssl_ds_two)
        
    return ssl_ds_one, ssl_ds_two