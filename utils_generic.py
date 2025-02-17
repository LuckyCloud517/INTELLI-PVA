# -*- coding: utf-8 -*-

import os
import glob
import tensorflow as tf
import numpy as np
from scipy import interpolate
from sklearn import preprocessing
from utils.utils_augmentations import (Jittering, Scaling, MagnitudeWarping, TimeWarping, 
                                       Permutation, RandSampling, FlipX, FlipY, 
                                       Masked, Crop_and_Resize, RandoomSmoothing)
#%%

def random_apply(func, x, p):
    """
    Applies a given function to input `x` with a probability `p`.

    Parameters:
    - func: The function to apply to `x` (e.g., a transformation or augmentation function).
    - x: The input data (could be a tensor or array) on which the function is applied.
    - p: The probability (between 0 and 1) that the function will be applied to `x`.

    Returns:
    - If the random number is less than `p`, the function `func` is applied to `x`.
    - Otherwise, returns `x` unchanged.
    """
    
    # Generate a random number between 0 and 1
    random_value = tf.random.uniform([], minval=0, maxval=1)
    
    # If the random value is less than p, apply the function
    if random_value < p:
        return func(x)
    else:
        # If not, return the original input
        return x


def custom_augment(data, method='Combinations'):
    """
    Applies a random data augmentation technique to the input `data` based on the chosen method.
    
    Parameters:
    - data: The input data to be augmented.
    - method: The augmentation method to use (e.g., 'None', 'Jittering', 'Scaling', etc.).
    
    Returns:
    - The augmented data after applying the selected augmentation method.
    """
    
    # No augmentation if method is 'None'
    if method == 'None':
        return data
    
    # Apply the chosen augmentation method
    if method == 'Jittering':
        data = random_apply(Jittering, data, p=0.8)
    elif method == 'Scaling':
        data = random_apply(Scaling, data, p=0.8)
    elif method == 'MagnitudeWarping':
        data = random_apply(MagnitudeWarping, data, p=0.8)
    elif method == 'TimeWarping':
        data = random_apply(TimeWarping, data, p=0.8)
    elif method == 'Permutation':
        data = random_apply(Permutation, data, p=0.8)
    elif method == 'RandSampling':
        data = random_apply(RandSampling, data, p=0.8)
    elif method == 'FlipX':
        data = random_apply(FlipX, data, p=0.8)
    elif method == 'FlipY':
        data = random_apply(FlipY, data, p=0.8)
    elif method == 'Masked':
        data = random_apply(Masked, data, p=0.8)
    elif method == 'Crop_and_Resize':
        data = random_apply(Crop_and_Resize, data, p=0.8)
    elif method == 'RandoomSmoothing':
        data = random_apply(RandoomSmoothing, data, p=0.8)
    
    # If method is 'Combinations', apply a series of augmentations
    elif method == 'Combinations':
        data = random_apply(Jittering, data, p=0.8)
        data = random_apply(Scaling, data, p=0.8)
        data = random_apply(MagnitudeWarping, data, p=0.8)
        data = random_apply(TimeWarping, data, p=0.8)
        # data = random_apply(Permutation, data, p=0.8)  # Uncomment to include Permutation
        data = random_apply(RandSampling, data, p=0.8)
        # data = random_apply(FlipX, data, p=0.8)  # Uncomment to include FlipX
        # data = random_apply(FlipY, data, p=0.8)  # Uncomment to include FlipY
        data = random_apply(Masked, data, p=0.8)
        data = random_apply(Crop_and_Resize, data, p=0.8)
        data = random_apply(RandoomSmoothing, data, p=0.8)
    
    return data

def pre_process(data, kind='linear', length=300):
    """
    Pre-process the input data by resampling and standardizing (z-score normalization).
    
    Parameters:
    - data: The input data to be preprocessed (should be a 2D array).
    - kind: The interpolation method to use (e.g., 'linear', 'nearest', etc.).
    - length: The desired length of the output data after resampling.
    
    Returns:
    - r: The preprocessed data, which has been resampled and standardized.
    """
    
    # Resampling: Create a scale that represents the original data indices
    scale = np.linspace(1, np.shape(data)[0], np.shape(data)[0])  # Original indices of the data
    
    # Create a new scale with the desired length for resampling
    new_scale = np.linspace(1, np.shape(data)[0], length)
    
    # Interpolate the data to resample it to the new scale (for the first column)
    interp_c1 = interpolate.interp1d(scale, data[:, 0], kind=kind)
    c1_resampled = interp_c1(new_scale)
    
    # Interpolate the data to resample it to the new scale (for the second column)
    interp_c2 = interpolate.interp1d(scale, data[:, 1], kind=kind)
    c2_resampled = interp_c2(new_scale)
    
    # Combine the resampled columns into a 2D array
    resampled_data = np.array([c1_resampled, c2_resampled]).T
    
    # Z-score standardization of the resampled data
    standardized_data = preprocessing.scale(resampled_data)
    
    # Return the standardized data
    return standardized_data



def walkFile(path, file_type):
    """
    Walks through all the files and directories within the given path.
    
    Parameters:
    - path: The directory path to start walking through.
    - file_type: The file extension to filter the files. If empty, no filtering is applied.
    
    Returns:
    - file_paths: A list of full paths for the files that match the given file extension.
    - dir_paths: A list of full paths for the directories within the given path.
    """
    file_paths = []  # List to store paths of files
    dir_paths = []   # List to store paths of directories
    
    # Traverse through all files and directories in the given path
    for root, dirs, files in os.walk(path):
        """
        Args:
        root: The current directory being accessed.
        dirs: A list of directories in the current directory.
        files: A list of files in the current directory.
        """
        # Loop through the files
        for file in files:
            if file_type != '':
                if file.endswith(file_type):  # Filter files based on the extension
                    file_paths.append(os.path.join(root, file))
            else:
                file_paths.append(os.path.join(root, file))  # No filtering, append all files

        # Loop through the directories
        for d in dirs:
            dir_paths.append(os.path.join(root, d))  # Append directory paths

    return file_paths, dir_paths


def walkFile_current_path(path, file_type):
    """
    Retrieves all files matching a specific extension in the current directory.
    
    Parameters:
    - path: The current directory path to search.
    - file_type: The file extension to filter the files.
    
    Returns:
    - files: A list of files matching the given extension in the current directory.
    """
    # Use glob to get all files matching the file type in the current path
    files = glob.glob(path + '/*' + file_type)
    
    return files

