# -*- coding: utf-8 -*-

import copy
import numpy as np
from scipy import interpolate
from scipy.interpolate import CubicSpline      
 
#%%
def pre_process(data, length=300):
    """
    This function performs linear interpolation to resize the input `data` to a specified length.

    Parameters:
    - data: A 1D NumPy array representing the input data (e.g., a time-series waveform).
    - length: The desired length of the output data (default is 300).
    
    Returns:
    - data_new: A 1D NumPy array representing the resized data, interpolated to the desired length.
    """
    
    # Create a scaling factor for the original data based on its length
    original_scale = np.linspace(1, len(data), len(data))
    
    # Create a new scaling factor for the desired length of the output data
    new_scale = np.linspace(1, len(data), length)
    
    # Perform linear interpolation using the scaling factors
    interpolation_func = interpolate.interp1d(original_scale, data, kind='linear')
    
    # Apply the interpolation to generate the new resized data
    data_new = interpolation_func(new_scale)
    
    return data_new


def resize(data, kind='linear', length=300):
    """
    This function resizes a 2D array `data` by performing linear interpolation for each column 
    (representing each feature/variable) to the specified length.

    Parameters:
    - data: A 2D NumPy array (e.g., each column represents a feature at different time steps).
    - kind: The interpolation method to use (default is 'linear').
    - length: The desired length of the output data (default is 300).
    
    Returns:
    - waveData: A 2D NumPy array with the resized data (same number of columns as `data`, with resized rows).
    """
    
    # Create a scaling factor for the original data based on its number of rows
    original_scale = np.linspace(1, np.shape(data)[0], np.shape(data)[0])
    
    # Create a new scaling factor for the desired length of the output data
    new_scale = np.linspace(1, np.shape(data)[0], length)
    
    # Interpolate for the first column (channel 1)
    interpolate_func_c1 = interpolate.interp1d(original_scale, data[:, 0], kind=kind)
    c1_new = interpolate_func_c1(new_scale)
    
    # Interpolate for the second column (channel 2)
    interpolate_func_c2 = interpolate.interp1d(original_scale, data[:, 1], kind=kind)
    c2_new = interpolate_func_c2(new_scale)
  
    # Combine the interpolated results from both channels into a new array
    waveData = np.array([c1_new, c2_new]).T
    
    return waveData


#%% 1.Jitter
def Jittering(X, sigma=0.3):
    """
    Adds Gaussian noise (Jittering) to the input data X.

    Parameters:
    - X: Input data, typically a NumPy array with shape (n_samples, n_features), where each row represents a sample 
      and each column represents a feature (e.g., time-series data).
    - sigma: The standard deviation of the Gaussian noise, which controls the noise intensity. Default is 0.3.
      A larger sigma will introduce stronger noise.

    Returns:
    - r: The data with added noise, having the same shape as input X.
    """
    # Generate Gaussian noise with the same shape as X
    # loc=0 means the noise has a mean of 0, scale=sigma means the noise has a standard deviation of sigma
    noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    
    # Add the noise to the original data X to create the jittered data
    r = X + noise
    
    return r


#%% 2.Scaling 
def Scaling(X, sigma=0.2):
    """
    Applies random scaling (with noise) to the input data X.

    Parameters:
    - X: Input data, typically a NumPy array with shape (n_samples, n_features), where each row represents a sample
      and each column represents a feature (e.g., time-series data).
    - sigma: The standard deviation of the Gaussian noise used to generate the scaling factor. Default is 0.2.
      A larger sigma will result in greater variation in the scaling factor.

    Returns:
    - r: The scaled data, with the same shape as the input X.
    """
    # Generate a scaling factor drawn from a normal distribution with mean=1 and standard deviation=sigma
    scalingFactor = np.random.normal(loc=1, scale=sigma, size=(1, X.shape[1]))
    
    # Create a matrix of ones with shape (X.shape[0], 1), then multiply it by the scaling factor
    # This ensures that each feature in X is scaled by the same factor (but different features may have different factors)
    noise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    
    # Apply the scaling by multiplying the original data X with the scaling factors
    r = X * noise
    
    return r


#%% 3. Magnitude Warping

def GenerateRandomCurves(X, sigma=0.2, knot=4):
    """
    Generates random curves that will be used for magnitude warping of the input data.

    Parameters:
    - X: Input data (e.g., time-series data), a NumPy array of shape (n_samples, n_features).
    - sigma: Standard deviation for the random noise used in curve generation. Default is 0.2.
    - knot: Number of knots to use in the cubic spline interpolation. Default is 4.

    Returns:
    - A NumPy array of the same shape as the input data X, representing the generated random curves.
    """
    # Generate x-values for the cubic spline interpolation. These are linearly spaced points between 0 and the last index of X.
    xx = (np.ones((X.shape[1], 1)) * (np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1)))).transpose()
    
    # Generate random y-values (curve values) with Gaussian noise centered at 1.0 with standard deviation 'sigma'.
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
    
    # Define the range of x-values from 0 to the last index of X
    x_range = np.arange(X.shape[0])
    
    # Create cubic spline interpolations for each feature in X
    cs_x = CubicSpline(xx[:, 0], yy[:, 0])  # Cubic spline for the first feature
    cs_y = CubicSpline(xx[:, 1], yy[:, 1])  # Cubic spline for the second feature
    
    # Return the interpolated values for the entire x_range for both features
    return np.array([cs_x(x_range), cs_y(x_range)]).transpose()

def MagnitudeWarping(X, sigma=0.2, knot=4):
    """
    Applies magnitude warping to the input data X by generating random curves and multiplying them element-wise.

    Parameters:
    - X: Input data (e.g., time-series data), a NumPy array of shape (n_samples, n_features).
    - sigma: Standard deviation for the random noise used in curve generation. Default is 0.2.
    - knot: Number of knots to use in the cubic spline interpolation. Default is 4.

    Returns:
    - A NumPy array of the same shape as the input data X, with magnitude warping applied.
    """
    # Generate random curves using the GenerateRandomCurves function
    cs = GenerateRandomCurves(X, sigma=sigma, knot=knot)
    
    # Apply magnitude warping by element-wise multiplying the input data X with the generated curves
    r = X * cs
    
    return r


#%% 4. Time Warping

def DistortTimesteps(X, sigma=0.2, knot=4):
    """
    Distorts the time steps of the input data by generating random curves and using them as time intervals.

    Parameters:
    - X: Input data (e.g., time-series data), a NumPy array of shape (n_samples, n_features).
    - sigma: Standard deviation for the random noise used in curve generation. Default is 0.2.
    - knot: Number of knots to use in the cubic spline interpolation. Default is 4.

    Returns:
    - A NumPy array of the same shape as X, representing the distorted time steps.
    """
    # Generate random curves for time intervals
    tt = GenerateRandomCurves(X, sigma, knot)  # Regard these samples as time intervals
    
    # Compute the cumulative sum of the time intervals to generate a cumulative time axis
    tt_cum = np.cumsum(tt, axis=0)
    
    # Scale the cumulative time to ensure that the last value matches X.shape[0] (the number of time steps)
    t_scale = [(X.shape[0] - 1) / tt_cum[-1, 0], (X.shape[0] - 1) / tt_cum[-1, 1]]
    
    # Apply scaling to the cumulative time values
    tt_cum[:, 0] = tt_cum[:, 0] * t_scale[0]
    tt_cum[:, 1] = tt_cum[:, 1] * t_scale[1]
    
    # Return the scaled cumulative time intervals
    return tt_cum

def TimeWarping(X, sigma=0.2, knot=4):
    """
    Applies time warping to the input data by distorting the time steps and resampling the data.

    Parameters:
    - X: Input data (e.g., time-series data), a NumPy array of shape (n_samples, n_features).
    - sigma: Standard deviation for the random noise used in curve generation. Default is 0.2.
    - knot: Number of knots to use in the cubic spline interpolation. Default is 4.

    Returns:
    - A NumPy array of the same shape as X, with time warping applied.
    """
    # Generate the distorted time steps using the DistortTimesteps function
    tt_new = DistortTimesteps(X, sigma, knot)
    
    # Initialize an array to store the warped data
    X_new = np.zeros(X.shape)
    
    # Create a range of x-values (time steps) from 0 to X.shape[0] - 1
    x_range = np.arange(X.shape[0])
    
    # Resample the data for each feature based on the distorted time intervals
    X_new[:, 0] = np.interp(x_range, tt_new[:, 0], X[:, 0])
    X_new[:, 1] = np.interp(x_range, tt_new[:, 1], X[:, 1])
    
    # Return the time-warped data
    return X_new

#%% 5. Permutation
def Permutation(X, maxPerm=8, minSegLength=10):
    """
    This function performs random permutation of segments in the input time-series data `X`.

    Parameters:
    - X: A 2D NumPy array where each row represents a time step and each column represents a feature.
    - maxPerm: The maximum number of segments to split the data into for permutation (default is 4).
    - minSegLength: The minimum length of each segment (default is 10).

    Returns:
    - A 2D NumPy array of the same shape as `X`, with permuted segments.
    """
    # Ensure that the minimum segment length is less than the total length of the data
    if minSegLength < X.shape[0]:
        # Randomly determine the number of segments to create
        nPerm = np.random.randint(2, maxPerm)
        
        # Initialize a new array to store the permuted data
        X_new = np.zeros(X.shape)
        
        # Generate a random permutation of segment indices
        idx = np.random.permutation(nPerm)
        
        # Variable to control the while loop
        bWhile = True
        
        # Keep generating segments until they satisfy the minimum segment length
        while bWhile == True:
            # Initialize an array to store the segment boundaries
            segs = np.zeros(nPerm + 1, dtype=int)
            
            # Randomly generate segment boundaries, ensuring that the segments are of valid length
            segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0] - minSegLength, nPerm - 1))
            
            # Set the last boundary to the total length of the data
            segs[-1] = X.shape[0]
            
            # Check if the minimum segment length condition is satisfied
            if np.min(segs[1:] - segs[0:-1]) > minSegLength:
                bWhile = False
        
        # Initialize a pointer to track the position for placing permuted data
        pp = 0
        
        # Permute the segments and place them in the new array
        for ii in range(nPerm):
            # Extract the segment specified by the current permutation
            x_temp = X[segs[idx[ii]]:segs[idx[ii] + 1], :]
            
            # Place the permuted segment in the new array
            X_new[pp:pp + len(x_temp), :] = x_temp
            
            # Update the pointer for the next segment
            pp += len(x_temp)
    else:
        # If the segment length is not valid, return the original data
        X_new = X
    
    return X_new


#%% 6. Random Sampling
def RandSampleTimesteps(X, nSample=300):
    """
    This function generates random time steps for sampling from the input time-series data.

    Parameters:
    - X: A 2D NumPy array where each row represents a time step and each column represents a feature.
    - nSample: The number of sample points to generate (default is 300).

    Returns:
    - A 2D NumPy array of shape (nSample, 2), containing the randomly sampled time steps for each feature.
    """
    # Initialize an array to store the sampled time steps
    tt = np.zeros((nSample, X.shape[1]), dtype=int)
    
    # Generate random sample points for the first feature (column)
    tt[1:-1, 0] = np.sort(np.random.randint(1, X.shape[0] - 1, nSample - 2))
    
    # Generate random sample points for the second feature (column)
    tt[1:-1, 1] = np.sort(np.random.randint(1, X.shape[0] - 1, nSample - 2))
    
    # Set the last row of tt to the last time step (X.shape[0] - 1)
    tt[-1, :] = X.shape[0] - 1
    
    return tt

def RandSampling(X, nSample=300):
    """
    This function samples the input time-series data using the randomly generated time steps.

    Parameters:
    - X: A 2D NumPy array where each row represents a time step and each column represents a feature.
    - nSample: The number of sample points to generate (default is 300).

    Returns:
    - A 2D NumPy array of the same shape as `X`, where the values are sampled based on the random time steps.
    """
    # Generate random sample time steps using the RandSampleTimesteps function
    tt = RandSampleTimesteps(X, nSample)
    
    # Initialize an array to store the sampled data
    X_new = np.zeros(X.shape)
    
    # Perform interpolation to sample the data along the first feature (column)
    X_new[:, 0] = np.interp(np.arange(X.shape[0]), tt[:, 0], X[tt[:, 0], 0])
    
    # Perform interpolation to sample the data along the second feature (column)
    X_new[:, 1] = np.interp(np.arange(X.shape[0]), tt[:, 1], X[tt[:, 1], 1])
    
    return X_new



#%% 7.filp x
def FlipX(X):
    """
    This function flips the input time-series data `X` along the horizontal axis (i.e., along the time axis).
    
    Parameters:
    - X: A 2D NumPy array where each row represents a time step and each column represents a feature.
    
    Returns:
    - A 2D NumPy array where the rows of `X` are reversed.
    """
    
    # Define the flip factor, which in this case is [-1, -1] indicating that the flipping happens along the rows
    flip_factor = [-1, -1]
    
    # Flip the data by multiplying it with the flip factor (this effectively reverses the rows)
    flipped_data = flip_factor * X[:,:]
    
    return flipped_data

#%% 8.flip y 
def FlipY(X):
    """
    This function flips the input time-series data along the time axis (reverses the rows).

    Parameters:
    - X: A 2D NumPy array where each row represents a time step and each column represents a feature.

    Returns:
    - A 2D NumPy array with the rows (time steps) reversed, i.e., the time-series data is flipped along the time axis.
    """
    # Create a list of indices that reverses the order of the rows (time steps)
    steps = list(np.arange(X.shape[0]-1, -1, -1))

    # Use the reversed indices to reorder the rows of the input data
    r = X[steps]
    
    return r


#%% 9.blockout
def Masked(X, minMaskLength=5, maxMaskLength=10, mask_mode="random"):
    """
    This function applies a masking operation to the input time-series data by setting random segments to zero.

    Parameters:
    - X: A 2D NumPy array where each row represents a time step and each column represents a feature.
    - minMaskLength: The minimum length of the segment to be masked (default is 5).
    - maxMaskLength: The maximum length of the segment to be masked (default is 10).
    - mask_mode: Specifies the type of masking. If "random", the segment length will be chosen randomly between `minMaskLength` and `maxMaskLength`. Otherwise, the segment length will be fixed to `minMaskLength`.

    Returns:
    - A 2D NumPy array with randomly masked segments (values set to zero) in the original data.
    """
    # Randomly select a starting index for masking in each feature (column)
    blockout_point = np.random.randint(0, X.shape[0], size=(X.shape[1]))

    # Create a deep copy of the original data to avoid modifying the original input
    X_new = copy.deepcopy(X)

    # Loop through each feature (column) of the data
    for i in range(X.shape[1]):
        # If mask_mode is "random", choose a random mask length
        if mask_mode == "random":
            # Randomly select a length for the mask between minMaskLength and maxMaskLength
            maskLength = np.random.randint(minMaskLength, maxMaskLength + 1, 1)[0]
            
            # Check if the mask fits within the bounds of the data
            if blockout_point[i] + maskLength < X_new.shape[0]:
                # Apply the mask by setting the selected segment to zero
                X_new[blockout_point[i]:blockout_point[i] + maskLength, i] = 0

        else:
            # If mask_mode is not "random", use the fixed mask length (minMaskLength)
            if blockout_point[i] + minMaskLength < X_new.shape[0]:
                # Apply the mask by setting the selected segment to zero
                X_new[blockout_point[i]:blockout_point[i] + minMaskLength, i] = 0

    return X_new

#%% 10.crop-and-resize

def Crop_and_Resize(X):
    """
    This function randomly crops a segment from the time-series data and resizes the result back to the original length.

    Parameters:
    - X: A 2D NumPy array where the rows represent the time steps and the columns represent features.

    Returns:
    - A 2D NumPy array with the same number of rows as the original input, but with a segment randomly removed and the result resized.
    """
    # Ensure the shape of X is valid (more than 20 rows)
    if X.shape[0] > 20:
        # Randomly select a point to crop (between 1 and one-sixth of the total length)
        crop_point = np.random.randint(1, X.shape[0] // 6)  # Number of points to crop
        
        # Randomly choose a starting index for the crop
        start_idx = np.random.randint(0, X.shape[0] - crop_point)  # Ensure we don't go out of bounds
        
        # Split the data into two parts: before and after the cropped segment
        part1 = X[:start_idx, :]  # Data before the crop
        part2 = X[start_idx + crop_point:, :]  # Data after the crop
        
        # Concatenate the remaining parts to create a new sequence
        X_new = np.concatenate([part1, part2], axis=0)
        
        # Resize the concatenated data back to the original length using linear interpolation
        r = resize(X_new, kind='linear', length=X.shape[0])
    else:
        # If the length of X is too small (<= 20), return the original data without changes
        r = X
    
    return r



#%% 11.randoom smoothing

def np_move_avg(a, n, mode="same"):
    """
    Applies a moving average filter to the input array.

    Parameters:
    - a: Input data (1D array), a NumPy array to be smoothed.
    - n: The window size for the moving average.
    - mode: Specifies the convolution mode ('same' by default, other option is 'valid').

    Returns:
    - A smoothed array (1D NumPy array).
    """
    return np.convolve(a, np.ones((n,))/n, mode=mode)

def RandoomSmoothing(X, minWindowsize=2, maxWindowsize=8, mode="same"):
    """
    Applies random smoothing (moving average) to the input data for each feature/column.

    Parameters:
    - X: Input data (2D array), each column represents a feature in the time-series data.
    - minWindowsize: Minimum window size for the moving average. Default is 2.
    - maxWindowsize: Maximum window size for the moving average. Default is 8.
    - mode: Specifies the convolution mode for the moving average ('same' by default).

    Returns:
    - A 2D array of the same shape as X, with random smoothing applied to each feature.
    """
    # Randomly choose a window size between the specified min and max values
    n = np.random.randint(minWindowsize, maxWindowsize)
    
    # If the window size is smaller than the number of samples, apply smoothing
    if n < X.shape[0]:
        # Initialize a new array to store the smoothed data
        X_new = np.zeros(X.shape)
        
        # Apply the moving average to each feature (column) of the input data
        for i in range(X.shape[1]):
            X_new[:, i] = np_move_avg(X[:, i], n, mode=mode)
        
    else:
        # If the window size is larger than the number of samples, return the original data
        X_new = X
        
    return X_new


