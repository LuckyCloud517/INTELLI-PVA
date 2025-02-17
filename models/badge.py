# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import pairwise_distances
import pdb

import tensorflow as tf
from tensorflow.keras import backend as K
from utils.utils_metris import plot_confusion_matrix

#%%
def slice_index(x, idx):
    tmp = []  # Initialize an empty list to store the sliced data
    for i in range(len(x)):  # Loop through each item in 'x'
        tmp.append(x[i][idx])  # For each item in 'x', append the element at index 'idx' to 'tmp'
    return tmp  # Return the list of sliced elements


def grad_cam(heatmap_model, true_label, waveData):
    # Start recording the gradients using a GradientTape context manager
    with tf.GradientTape() as gtape:
        # Get the convolutional output and predictions from the model
        conv_output, predictions = heatmap_model(waveData)
        
        # Check if the true label is not 0 (it’s a multi-class classification)
        if true_label!=0:
            # If the true label is non-zero, get the prediction probability for the specific class
            prob = predictions[true_label-1][:, 1] # The predicted probability for the class
        else:
            # If true label is 0, calculate the average probability across all classes
            prob = (predictions[0][:, 0]+predictions[1][:, 0]+predictions[2][:, 0]+predictions[3][:, 0])/4
        
        # Calculate the gradients of the predicted probability with respect to the convolutional layer’s output
        grads = gtape.gradient(prob, conv_output)  # Gradient of the probability w.r.t. convolutional output
        # Perform global average pooling over the gradient values to get the weight for each feature map
        pooled_grads = K.mean(grads, axis=(0, 1))  # Pooling gradients across the spatial dimensions
    
    # Multiply the pooled gradients with the convolutional output to get the weighted feature map
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1) #权重与特征层相乘，512层求和平均
    #print(f"predict label:{np.argmax(Predictions[0])}")
    
    # Normalize the heatmap by dividing by the maximum value to ensure it’s between 0 and 1
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10# Avoid division by zero
    heatmap /= max_heat
    
    # Apply ReLU to the heatmap to set all negative values to zero
    heatmap = np.maximum(heatmap, 0)
    return heatmap


# Max-Min Normalization: Scales the input array `x` to the range [0, 1]
def MaxMinNormalization(x):
    # Normalize each element in x using the formula (x - min) / (max - min)
    # This scales the values such that the minimum becomes 0 and the maximum becomes 1
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x  # Return the normalized array



# Jaccard Similarity: Measures the similarity between two sets s1 and s2
# It is defined as the size of the intersection divided by the size of the union of the sets
def jaccard_similarity(s1, s2):
    s1, s2 = set(s1), set(s2)  # Convert the input lists to sets to eliminate duplicates and allow set operations
    # Compute the intersection and union of the sets and return the ratio of their sizes
    return len(s1 & s2) / len(s1 | s2)

# Jaccard Distance: Measures the dissimilarity between two sets s1 and s2
# It is defined as 1 minus the Jaccard similarity
def jaccard_distance(s1, s2):
    js = jaccard_similarity(s1, s2)  # Get the Jaccard similarity between the two sets
    return 1 - js  # The distance is the complement of the similarity


# Function to get the gradient embeddings for a batch of data samples
# It calculates embeddings based on the output of the active learner and the intermediate layer model
def get_grad_embedding(active_learner, intermediate_layer_model, X, p, embDim, n_heads, **kwargs):
    """
    Parameters:
    - active_learner: The model used for making predictions (classification output).
    - intermediate_layer_model: A model used to extract features from the intermediate layer.
    - X: Input data (batch of samples).
    - p: The number of samples in the batch.
    - embDim: The dimension of the feature embeddings.
    - n_heads: The number of heads for multi-head classification tasks.
    
    Returns:
    - embedding: A numpy array containing the gradient embeddings for each sample.
    """
    # Initialize the embedding array with zeros, shape [p, embDim * n_heads]
    embeddings = np.zeros([p, embDim * n_heads])

    # Get the predictions from the active learner (classification output)
    predictions = active_learner.predict(X)
    
    # Get the output from the intermediate layer model (used for feature extraction)
    intermediate_outputs = intermediate_layer_model.predict(X)
    
    # Iterate over each sample in the batch
    for sample_idx in range(p):
        # Iterate over each head of the multi-head task
        for head_idx in range(n_heads):
            # If the predicted class for the current sample and head is class 1 (positive class)
            if np.argmax(predictions[head_idx][sample_idx]) == 1:
                # Assign the intermediate layer output, scaled by the inverse of the predicted probability
                embeddings[sample_idx, embDim * head_idx : embDim * (head_idx + 1)] = intermediate_outputs[sample_idx] * (1 - predictions[head_idx][sample_idx][1])
            else:
                # Otherwise, scale the intermediate layer output by the negative of the predicted probability
                embeddings[sample_idx, embDim * head_idx : embDim * (head_idx + 1)] = intermediate_outputs[sample_idx] * (-1 * predictions[head_idx][sample_idx][0])
    
    return embeddings  # Return the calculated gradient embeddings

# Function to initialize cluster centers using the gradient embedding, similar to k-means initialization
def init_centers(gradEmbed, K):
    # Find the index of the sample that has the maximum Euclidean norm in the gradient embeddings
    ind = np.argmax([np.linalg.norm(s, 2) for s in gradEmbed])
    # Initialize the list of centroids with the sample corresponding to the largest norm
    mu = [gradEmbed[ind]]
    indsAll = [ind]  # List to store indices of selected centroids
    centInds = [0.] * len(gradEmbed)  # List to track which centroid a sample is closest to
    cent = 0  # Initialize centroid counter
    
    # Keep adding centroids until we have K centroids
    while len(mu) < K:
        # If there's only one centroid, compute pairwise distances between all embeddings and this centroid
        if len(mu) == 1:
            D2 = pairwise_distances(gradEmbed, mu).ravel().astype(float)
        else:
            # Otherwise, compute the pairwise distances from the last added centroid
            newD = pairwise_distances(gradEmbed, [mu[-1]]).ravel().astype(float)
            # Update the distance values and associate samples with the closest centroid
            for i in range(len(gradEmbed)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        
        # If no progress in distance values, trigger a breakpoint (this is a safety check)
        if sum(D2) == 0.0:
            pdb.set_trace()
        
        D2 = D2.ravel().astype(float)  # Flatten the distance array
        Ddist = (D2 ** 2) / sum(D2 ** 2)  # Normalize distances (squared) to create a probability distribution
        
        # Define a custom discrete distribution based on the normalized distances
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        
        # Sample a new index based on the custom distribution (selecting the sample that is farthest)
        ind = customDist.rvs(size=1)[0]
        
        # Ensure that the newly selected centroid index is not a duplicate
        while ind in indsAll:
            ind = customDist.rvs(size=1)[0]
        
        # Append the selected centroid to the list
        mu.append(gradEmbed[ind])
        indsAll.append(ind)
        cent += 1  # Increment the centroid counter
    
    return indsAll  # Return the indices of the selected centroids


def save_result_table(train_result, divs, cvgs, X, savepath, filename):
    columns = ['acc','sen','spe','ppv','npv','f1','auc','diversity','coverage', 'num']
    df = pd.DataFrame(index=['Other', 'Delayed Cycling', 'Premature Cycling', 'DoubleTrig', 'InefTrig', 'avg±std'], columns=columns)
    for k, v in train_result.items():
        for c in columns:
            try:
                df[c][k] = v[c]
            except:
                pass

    df['diversity'].iloc[:5] = divs
    df['diversity']['avg±std'] = f'{round(df.diversity.mean(),3)}±{round(df.diversity.std(ddof=0),3)}'
    df['coverage'].iloc[:5] = cvgs
    df['coverage']['avg±std'] = f'{round(df.coverage.mean(),3)}±{round(df.coverage.std(ddof=0),3)}'

    df['acc']['avg±std'] = f'{round(df.acc.mean(),3)}±{round(df.acc.std(ddof=0),3)}'
    df['sen']['avg±std'] = f'{round(df.sen.mean(),3)}±{round(df.sen.std(ddof=0),3)}'
    df['spe']['avg±std'] = f'{round(df.spe.mean(),3)}±{round(df.spe.std(ddof=0),3)}'
    df['ppv']['avg±std'] = f'{round(df.ppv.mean(),3)}±{round(df.ppv.std(ddof=0),3)}'
    df['npv']['avg±std'] = f'{round(df.npv.mean(),3)}±{round(df.npv.std(ddof=0),3)}'
    df['f1']['avg±std'] = f'{round(df.f1.mean(),3)}±{round(df.f1.std(ddof=0),3)}'
    df['auc']['avg±std'] = f'{round(df.auc.mean(),3)}±{round(df.auc.std(ddof=0),3)}'

    index_label = ['Other', 'Delayed Cycling', 'Premature Cycling', 'DoubleTrig', 'InefTrig']
    for it in range(len(index_label)):
        df['num'][index_label[it]] = len(np.where(X==it)[0])

    df.to_excel(os.path.join(savepath, filename), index = True)
    
    
# Function to compute diversity based on Jaccard similarity between heatmaps of labeled data
# The function compares the similarity of heatmaps generated for each pair of labeled samples
def get_diversity(heatmap_model, labeled_indices, x_train, y_train, labels=[0, 1, 2, 3, 4]):
    diversities, diversity_details = [], []  # Lists to store diversity metrics and pairwise diversity details
    
    # Iterate through each label class (e.g., 0, 1, 2, etc.)
    for label in labels:
        label_diversity = []  # List to store Jaccard distances for the current label class
        
        # Iterate over all labeled data samples
        for i in labeled_indices:  
            if y_train[i] == label:  # If the label matches the current class
                sample_data1 = np.expand_dims(x_train[i], axis=0)  # Expand the data sample for input to the model
                heatmap1 = grad_cam(heatmap_model, label, sample_data1)  # Generate the heatmap using grad-CAM
                heatmap1 = MaxMinNormalization(heatmap1)  # Normalize the heatmap
                heatmap1_indices = np.where(heatmap1[0] > 0.5)  # Find the indices where the heatmap intensity is above 0.5
                
                # Compare the current sample with all other labeled samples
                for j in labeled_indices:
                    if y_train[j] == label:  # If the second sample matches the current class
                        sample_data2 = np.expand_dims(x_train[j], axis=0)  # Expand the second sample for input
                        heatmap2 = grad_cam(heatmap_model, label, sample_data2)  # Generate the heatmap for the second sample
                        heatmap2 = MaxMinNormalization(heatmap2)  # Normalize the heatmap
                        heatmap2_indices = np.where(heatmap2[0] > 0.5)  # Find the indices where the heatmap intensity is above 0.5
                        
                        # Compute the Jaccard distance between the heatmaps of the two samples
                        jaccard_dist = jaccard_distance(heatmap1_indices[0], heatmap2_indices[0])
                        label_diversity.append(jaccard_dist)  # Add the Jaccard distance to the diversity list
                        
                        # Store the details of the pairwise comparison
                        diversity_details.append({'label': label, 'train_sample_a': i, 'train_sample_b': j, 'jaccard_distance': jaccard_dist})
        
        # Compute the mean diversity for the current label and add it to the list
        diversities.append(np.mean(label_diversity))
    
    # Convert the diversity details into a DataFrame for easier analysis and return it with the average diversity values
    diversity_details_df = pd.DataFrame(diversity_details, columns=['label', 'train_sample_a', 'train_sample_b', 'jaccard_distance'])
    return diversities, diversity_details_df


# Function to compute coverage based on Jaccard similarity between heatmaps of labeled and test data
# The function measures how well the labeled samples' heatmaps cover the test samples' heatmaps
def get_coverage(heatmap_model, labeled_indices, x_train, y_train, x_test, y_test, labels=[0, 1, 2, 3, 4]):
    coverages, coverage_details = [], []  # Lists to store coverage metrics and pairwise coverage details
    
    # Iterate through each label class (e.g., 0, 1, 2, etc.)
    for label in labels:
        label_coverage = []  # List to store coverage for the current label class
        
        # Iterate over all test samples
        for i in range(len(y_test)):
            coverage_for_test_sample = []  # Temporary list to store coverage values for the current test sample
            
            if y_test[i] == label:  # If the test sample matches the current class
                sample_data1 = np.expand_dims(x_test[i], axis=0)  # Expand the test sample for input to the model
                heatmap1 = grad_cam(heatmap_model, label, sample_data1)  # Generate the heatmap using grad-CAM
                heatmap1 = MaxMinNormalization(heatmap1)  # Normalize the heatmap
                heatmap1_indices = np.where(heatmap1[0] > 0.5)  # Find the indices where the heatmap intensity is above 0.5
                
                # Compare the current test sample with all labeled samples
                for j in labeled_indices:
                    if y_train[j] == label:  # If the labeled sample matches the current class
                        sample_data2 = np.expand_dims(x_train[j], axis=0)  # Expand the labeled sample for input
                        heatmap2 = grad_cam(heatmap_model, label, sample_data2)  # Generate the heatmap for the labeled sample
                        heatmap2 = MaxMinNormalization(heatmap2)  # Normalize the heatmap
                        heatmap2_indices = np.where(heatmap2[0] > 0.5)  # Find the indices where the heatmap intensity is above 0.5
                        
                        # Compute the Jaccard distance between the heatmaps of the test and labeled samples
                        jaccard_dist = jaccard_distance(heatmap1_indices[0], heatmap2_indices[0])
                        coverage_for_test_sample.append(1 - jaccard_dist)  # Add the coverage to the list (1 - Jaccard distance)
                        
                        # Store the details of the pairwise comparison
                        coverage_details.append({'label': label, 'train_sample': j, 'test_sample': i, '1_jaccard_distance': 1 - jaccard_dist})
                
                # Add the maximum coverage for the current test sample
                label_coverage.append(np.max(coverage_for_test_sample))
        
        # Compute the mean coverage for the current label and add it to the list
        coverages.append(np.mean(label_coverage))
    
    # Convert the coverage details into a DataFrame for easier analysis and return it with the average coverage values
    coverage_details_df = pd.DataFrame(coverage_details, columns=['label', 'train_sample', 'test_sample', '1_jaccard_distance'])
    return coverages, coverage_details_df


def forward_inference(active_learner, 
                      x_test, 
                      y_test,
                      x_test_id,
                      savepath,
                      labels=['Delayed Cycling', 'Premature Cycling', 'DoubleTrig', 'InefTrig', 'Other']
                      ):
    
    proba = active_learner.predict(x_test)

    pred_label = [[],[],[],[]] # Initialize lists to store predicted labels
    for i in range(len(proba[0])):
        for j in range(4):  # For each label column
            idx = np.argmax(proba[j][i])  # Get predicted label index
            pred_label[j].append(1 if idx != 0 else 0)  # Append binary classification results
    
                
    # Compute confusion matrix for each label
    for col in range(len(labels[:-1])):
        cm = np.zeros((2, 2))
        for i in range(len(proba[0])):
            if y_test[col][i] == 1:
                cm[0][0] = cm[0][0] + 1 if pred_label[col][i] == 1 else cm[0][0]
                cm[0][1] = cm[0][1] + 1 if pred_label[col][i] == 0 else cm[0][1]
            if y_test[col][i] == 0:
                cm[1][1] = cm[1][1] + 1 if pred_label[col][i] == 0 else cm[1][1]
                cm[1][0] = cm[1][0] + 1 if pred_label[col][i] == 1 else cm[1][0]
        plot_confusion_matrix(cm, savedir=savepath, savename=labels[col], class_names=[labels[col],'Other'] )
        
    return proba, pred_label, cm