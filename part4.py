# -*- coding: utf-8 -*-

import os, time
import numpy as np
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras.models import Model 
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

from data.dataset import fully_supervised_dataset
from models.model import get_projection, ContrastiveModel, RespNet
from utils.utils_metris import get_metris
from models.utils_loss import plot_progress_multi_task
from models.badge import (get_grad_embedding, init_centers, slice_index, 
                          save_result_table, get_diversity, get_coverage, forward_inference)

import warnings
warnings.filterwarnings("ignore")

#%%
T1 = time.perf_counter()

# Experiment configuration
exp_name = 'exp3'  # Name of the experiment
ablation_method = '12-Combinations' # Ablation method being used
previous_checkpoint_time = '2025-02-17-16-28-56'  # Pre-trained model checkpoint time

# Create necessary directories for saving results
rootpath = os.path.join(os.getcwd(), exp_name)
if not os.path.isdir(rootpath):
    os.makedirs(rootpath)
    os.makedirs(os.path.join(rootpath, 'weights'))
    os.makedirs(os.path.join(rootpath, 'predict'))


# Parameters
pretrained_encoder = True # Whether to use a pretrained encoder model
class_balanced_pool = True # Whether to use class-balanced pooling strategy

latent_dim = 128  # Latent space dimension
project_dim = 128  # Projection dimension for contrastive loss
temperature = 0.1  # Temperature for contrastive loss
num_epochs = 200 # Number of train epochs
batch_size = 128  # Batch size for training 
random_seed = 66  # Random seed for reproducibility
maxlen = 300  # Maximum sequence length
n_splits = 10  # Number of splits for cross-validation
initial_size = 50  # Initial size for active learning

count = 0
iteration = 0

# Labels for classification tasks
labels = ['Delayed Cycling', 'Premature Cycling', 'DoubleTrig', 'InefTrig', 'Other']    

# Load dataset
x, y, x_index = fully_supervised_dataset(maxlen=maxlen)

# Initialize StratifiedKFold for cross-validation
kfold = StratifiedKFold(n_splits=n_splits, random_state=random_seed, shuffle=True)    
    
for train_index, test_index in kfold.split(x, y):    
    # Split data into training and testing sets
    x_train, x_test, x_train_id, x_test_id, y_tra, y_te = np.array(x)[train_index], np.array(x)[test_index],np.array(x_index)[train_index], np.array(x_index)[test_index], np.array(y)[train_index], np.array(y)[test_index]
    
    # Prepare multi-labels for training and testing
    y_train = [np.where(np.array(y_tra)!=1, 0, 1), np.where(np.array(y_tra)!=2, 0, 1), np.where(np.array(y_tra)!=3, 0, 1), np.where(np.array(y_tra)!=4, 0, 1)]
    y_test = [np.where(np.array(y_te)!=1, 0, 1), np.where(np.array(y_te)!=2, 0, 1), np.where(np.array(y_te)!=3, 0, 1), np.where(np.array(y_te)!=4, 0, 1)]
    
    # Reshape input data to match the expected input shape
    x_train = np.reshape(x_train, (-1, maxlen, 2))
    x_test = np.reshape(x_test, (-1, maxlen, 2))
    
    #%% Creating classification model
    tf.keras.backend.clear_session()# Clear session for fresh model training
    
    if pretrained_encoder:
        # Initialize pre-trained contrastive model
        encoder = RespNet(maxlen=maxlen, include_top=True, classes=latent_dim)
        #plot_model(encoder, to_file=os.path.join(rootpath, 'weights', 'encoder.png'), show_shapes=True)
        
        # Initialize projection layer for contrastive loss
        projection_layer = get_projection(project_dim=project_dim)
        #plot_model(projection_layer, to_file=os.path.join(rootpath, 'weights', 'projection.png'), show_shapes=True)
        
        # Initialize contrastive model
        contrastive_model = ContrastiveModel(temperature, encoder, projection_layer)
        contrastive_model.compile(contrastive_optimizer=tf.keras.optimizers.Adam())
        
        # Load weights from the previous training checkpoint
        model_history_path = f'./exp2/{ablation_method}/weights/{previous_checkpoint_time}/model_010_0.236_3.338.h5'
        contrastive_model.build(input_shape=(None, 300, 2))
        contrastive_model.load_weights(model_history_path, by_name=True)
        
        # Active learner model setup
        active_learner = RespNet(maxlen=maxlen, active_learner=True)
        
        # Transfer pre-trained weights to active learner model
        for i in range(len(active_learner.layers)-6):
            print(f'Model Layer {i}: {active_learner.layers[i].name}, Encoder Layer {i}: {encoder.layers[i].name}')
            active_learner.layers[i].set_weights(encoder.layers[i].get_weights())
        
        for i in range(len(active_learner.layers)-6):
            active_learner.layers[i].trainable = True
    else:
        active_learner = RespNet(maxlen=maxlen, active_learner=True)
    
    # Compile the active learner model
    active_learner.compile(loss="sparse_categorical_crossentropy",
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=['accuracy'])
    
    # Save the initial active learner model
    active_learner.save(os.path.join(rootpath, 'active_learner_init.h5'))
    active_learner.summary()
    plot_model(active_learner, to_file=os.path.join(rootpath, 'active_learner.png'), show_shapes=True)

    #%% First round of training
    savepath = os.path.join(rootpath, 'predict', str(iteration))
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        
    labeled_idx, size, acc = [], [], []
    labels = list(set(y_tra))
    div, cvg = [], []
    
    np.random.seed(10)
    idx = np.random.choice(range(len(x_train)), size=initial_size, replace=False).tolist()
    
    model_name = f'active_learner_{iteration:03d}.h5'
    model_path = os.path.join(rootpath, 'weights', model_name)

    model_checkpoint_callback = ModelCheckpoint(model_path, verbose=1, save_best_only=True, period=1)
    early_stopper_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=1, restore_best_weights=True)
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, cooldown=5, min_lr=1e-9)
    
    
    hist = active_learner.fit(x_train[idx], 
                              slice_index(y_train, [idx]), 
                              epochs=num_epochs, 
                              batch_size=batch_size, 
                              validation_data=(x_test, y_test),
                              verbose=0,
                              callbacks=[early_stopper_callback, 
                                         reduce_lr_callback,
                                         model_checkpoint_callback])

    labeled_idx.extend(idx)

    #%% Calculate first round of diversity and coverage
    last_conv_layer = active_learner.get_layer('add')
    heatmap_model = Model([active_learner.inputs], [last_conv_layer.output, active_learner.output])

    # Diversity calculation
    divs, diversity_pair = get_diversity(heatmap_model, labeled_idx, x_train, y_tra, labels=[0,1,2,3,4])
    div.append(np.mean(divs))
    diversity_pair.to_excel(os.path.join(savepath, "diversity_pair.xlsx"), index=False)
    
    # Coverage calculation
    cvgs, coverage_pair = get_coverage(heatmap_model, labeled_idx, x_train, y_tra, x_test, y_te, labels=[0,1,2,3,4])
    cvg.append(np.mean(cvgs))
    coverage_pair.to_excel(os.path.join(savepath, "coverage_pair.xlsx"), index=False)
    
    # Plot training progress and save results
    plot_progress_multi_task(hist, save_dir=savepath, save_history=os.path.join(savepath, 'history.npy'))
    proba, pred_label, cm = forward_inference(active_learner, x_test, y_test, x_test_id, savepath)
    
    # Save evaluation results
    train_result = {}
    for col in range(len(labels[:-1])):
        train_result.update({labels[col] : get_metris(y_test[col], pred_label[col], proba[col][:,1])})
    
    save_result_table(train_result=train_result, 
                      divs=divs, 
                      cvgs=cvgs, 
                      X=y_tra[labeled_idx], 
                      savepath=savepath, 
                      filename=f"iteration{iteration}-samples{len(labeled_idx)}.xlsx")
    
    np.save(os.path.join(savepath, f'labeled_index_{iteration}.npy'), labeled_idx)

    print(f'iteration: {iteration}')
    for col in range(len(labels[:-1])):
        print(f'{labels[col]}:', get_metris(y_test[col], pred_label[col], proba[col][:,1]))
    print(f'diversity: {np.mean(divs)}, coverage: {np.mean(cvgs)}')

    #%% Active learning loop: Hybrid Strategy Sampling for subsequent iterations
    # Code continues as per your active learning setup with additional loops for model retraining and re-evaluation.
    # Execute active learning with mixed sampling strategy for 20 iterations
    iteration = 1
    k, p = 50, 5000  # Number of clusters (k) for center initialization, batch size (p) for sampling
    iterations = 21  # Total number of active learning iterations

    embDim = 128  # Embedding dimension
    n_head = 4  # Number of classification heads
    
    # Active learning loop for 20 iterations
    while iteration < iterations:
        print(f'iteration: {iteration}')
        # Find the indices of unlabeled samples (those not in the labeled set)
        unlabeled_idx = np.arange(x_train.shape[0])[np.logical_not(np.in1d(np.arange(x_train.shape[0]), labeled_idx))]
                
        layer_name = 'embedding'# Define the layer for extracting the embeddings
        intermediate_layer_model = Model(inputs=active_learner.input,
                                    outputs=active_learner.get_layer(layer_name).output)
        
        # Sampling strategy based on class balance
        if not class_balanced_pool:# Plan 1: Random sampling from the pool
            idx = np.random.choice(unlabeled_idx, size=p)# Randomly select 'p' samples
            
        if class_balanced_pool == True:# Plan 2: Class-balanced sampling based on coverage
            # Calculate sample weights inversely proportional to the coverage
            sample_weights = 1 / (cvgs / max(cvgs))  
            sample_size = np.around((p / sum(sample_weights)) * sample_weights[1:])  # Determine the sample size for each class
            sample_size = np.insert(sample_size, 0, p - sum(sample_size))  # Ensure the total size sums up to 'p'
            print(f'sample_size: {sample_size}')
        
            # Track the number of samples selected for each class
            label_cnt = np.zeros((5,))# There are 5 classes
            idx, idx_tmp = [], []# Indices for the current iteration's selected samples
            ct = 0 # Counter for the number of iterations in this loop
            labeled_idx_tmp = labeled_idx[:] # Temporary copy of labeled indices
            
            # Continue sampling until the class counts match the required distribution
            while not np.alltrue(label_cnt==sample_size):
                try:
                    # Get the remaining unlabeled indices
                    unlabeled_idx_tmp = np.arange(x_train.shape[0])[np.logical_not(np.in1d(np.arange(x_train.shape[0]), labeled_idx_tmp))]
                    # Randomly select 'p' samples from the unlabeled pool
                    idx_tmp = np.random.choice(unlabeled_idx_tmp, size=p, replace=False)
                    prob = np.array(active_learner.predict(x_train[idx_tmp]))# Get class probabilities for the selected samples
                    
                    # Distribute the selected samples into the appropriate classes
                    for m in range(len(prob[0])):
                        # Distribute samples based on the predicted class probabilities and update counts
                        if np.argmax(prob[0][m])==0 and np.argmax(prob[1][m])==0 and np.argmax(prob[2][m])==0 and np.argmax(prob[3][m])==0 and label_cnt[0]!=sample_size[0]:
                            label_cnt[0] += 1 
                            idx.append(idx_tmp[m])
                        if np.argmax(prob[0][m])==1 and label_cnt[1]!=sample_size[1]:
                            label_cnt[1] += 1 
                            idx.append(idx_tmp[m])
                        if np.argmax(prob[1][m])==1 and label_cnt[2]!=sample_size[2]:
                            label_cnt[2] += 1 
                            idx.append(idx_tmp[m])
                        if np.argmax(prob[2][m])==1 and label_cnt[3]!=sample_size[3]:
                            label_cnt[3] += 1 
                            idx.append(idx_tmp[m])
                        if np.argmax(prob[3][m])==1 and label_cnt[4]!=sample_size[4]:
                            label_cnt[4] += 1 
                            idx.append(idx_tmp[m])
                            
                    ct +=1
                    print(f'{ct}:{label_cnt}')
                    
                    # Add selected indices to the list of labeled samples
                    labeled_idx_tmp.extend(idx_tmp)
                except:
                    # If an error occurs (e.g., not enough samples), break the loop
                    break
            # Ensure unique indices (remove duplicates) 
            idx = np.array(list(set(idx)))
            
        #%% Embedding and gradient-based sampling
        # Generate embedding features for the selected batch (for analysis and clustering)
        gradEmbed = get_grad_embedding(active_learner=active_learner, 
                                        intermediate_layer_model=intermediate_layer_model, 
                                        X=x_train[idx], 
                                        p=len(idx), 
                                        embDim=embDim, 
                                        n_heads=n_head)
        # Initialize centers for clustering the embeddings
        ind = init_centers(gradEmbed, k)
        
        # Select the indices corresponding to the cluster centers
        index = idx[ind]
        
        # Add the newly labeled indices to the list of all labeled indices
        labeled_idx.extend(index)
        
        #Training the model with the new batch of labeled samples
        savepath = os.path.join(rootpath, 'predict', str(iteration))
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        model_name = f'active_learner_{iteration:03d}.h5'
        model_path = os.path.join(rootpath, 'weights', model_name)
        
        # Train the active learner model with the updated labeled data
        hist = active_learner.fit(x_train[labeled_idx], 
                                  slice_index(y_train, [labeled_idx]),
                                  epochs=iteration, 
                                  batch_size=batch_size, 
                                  validation_data=(x_test, y_test),
                                  verbose=0,
                                  callbacks=[early_stopper_callback, 
                                             reduce_lr_callback,
                                             model_checkpoint_callback])
        
        #%% Calculate and save diversity and coverage metrics
        # Get the last convolutional layer and create a heatmap model
        last_conv_layer = active_learner.get_layer('add')
        heatmap_model = Model([active_learner.inputs], [last_conv_layer.output, active_learner.output])
        
        # Diversity: calculate how diverse the selected samples are
        divs, diversity_pair = get_diversity(heatmap_model, labeled_idx, x_train, y_tra, labels=[0,1,2,3,4])
        div.append(np.mean(divs)) # Append the average diversity to the list
        diversity_pair.to_excel(os.path.join(savepath, "diversity_pair.xlsx"), index=False)# Save diversity results
        
        # Coverage: calculate how well the selected samples cover the whole dataset
        cvgs,coverage_pair = get_coverage(heatmap_model, labeled_idx, x_train, y_tra, x_test, y_te, labels=[0,1,2,3,4])
        cvg.append(np.mean(cvgs)) # Append the average coverage to the list
        coverage_pair.to_excel(os.path.join(savepath, "coverage_pair.xlsx"), index=False)# Save coverage results
  
        # Plot training progress and save results
        plot_progress_multi_task(hist, save_dir=savepath, save_history=os.path.join(savepath, 'history.npy'))
        
        # Get the model's predictions on the test set
        proba, pred_label, cm = forward_inference(active_learner, x_test, y_test, x_test_id, savepath)
        
        # Save the evaluation metrics (accuracy, etc.)
        train_result = {}
        for col in range(len(labels[:-1])):
            train_result.update({labels[col] : get_metris(y_test[col], pred_label[col], proba[col][:,1])})
    
        save_result_table(train_result=train_result, 
                          divs=divs, 
                          cvgs=cvgs, 
                          X=y_tra[labeled_idx], 
                          savepath=savepath, 
                          filename=f"iteration{iteration}-samples{len(labeled_idx)}.xlsx")
    
        np.save(os.path.join(savepath, f'labeled_index_{iteration}.npy'), labeled_idx)
        
        # Print the results for the current iteration
        print(f'iteration: {iteration}')
        for col in range(len(labels[:-1])):
            print(f'{labels[col]}:', get_metris(y_test[col], pred_label[col], proba[col][:,1]))
        print(f'diversity: {np.mean(divs)}, coverage: {np.mean(cvgs)}')
        
        # Increment the iteration count
        iteration = iteration + 1

    count = count + 1
    if count == 1:
        break
        
T2 = time.perf_counter()
print(f'Spend about {round((T2 - T1)/3600, 2)} h')  


