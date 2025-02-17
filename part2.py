# -*- coding: utf-8 -*-

import os
import time
from datetime import datetime
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from data.dataset import simclr_dataset_with_label
from models.loss import plot_contrast_training_curves
from models.model import RespNet, get_projection, ContrastiveModel

from tensorflow.keras.callbacks import ModelCheckpoint
#%% Algorithm hyperparameters
now = datetime.now()
date_time = now.strftime("%Y-%m-%d-%H-%M-%S")

T1 = time.perf_counter()

keep_train = False  # Flag to keep training from a previous checkpoint
previous_training_time = '2023-11-09-17-58-28'  # Previous model training time
previous_model_name = 'model_100_0.696_1.369.h5'  # Previous model file name

exp_name = 'exp2'  

ablation_methods = ['12-Combinations']  # List of ablation methods to try
# ablation_methods = ['0-None', '1-Jittering', '2-Scaling', '3-MagnitudeWarping', '4-TimeWarping',
#                     '5-Permutation', '6-RandSampling', '7-FlipX', '8-FlipY', 
#                     '9-Masked', '10-Crop_and_Resize', '11-RandoomSmoothing', '12-Combinations']


batch_size = 1024  # Batch size
epochs = 200  # Number of epochs
latent_dim = 128  # Latent space dimension
project_dim = 128  # Projection dimension for contrastive loss
temperature = 0.1  # Temperature for contrastive loss
seed = 26  # Random seed
maxlen = 300  # Maximum sequence length

for method in ablation_methods:
    # Define root and model save paths
    rootpath = os.path.join(os.getcwd(), exp_name, method)
    if not os.path.isdir(rootpath):
        os.makedirs(rootpath)
    
    model_savepath = os.path.join(rootpath, 'weights', date_time)
    if not os.path.isdir(model_savepath):
        os.makedirs(model_savepath)
    
    method_name = method.split('-')[1]  # Extract method name from the ablation method

    ssl_ds_one, ssl_ds_two = simclr_dataset_with_label(maxlen=maxlen, method=method_name)
    print(f'unlabeled_dataset_size:{len(ssl_ds_one)}')
    
    unlabeled_dataset_size = len(ssl_ds_one)
            
    steps_per_epoch = unlabeled_dataset_size // batch_size
    unlabeled_batch_size = unlabeled_dataset_size // steps_per_epoch

    # Prepare datasets for training
    dataset_one = tf.data.Dataset.from_tensor_slices(ssl_ds_one)
    dataset_one = (dataset_one.shuffle(10 * unlabeled_batch_size, seed=seed)
                   #.map(custom_augment, num_parallel_calls=AUTO)
                   .batch(unlabeled_batch_size)
                   .prefetch(tf.data.AUTOTUNE)
                   )
    
    dataset_two = tf.data.Dataset.from_tensor_slices(ssl_ds_two)
    dataset_two= (dataset_two.shuffle(10 * unlabeled_batch_size, seed=seed)
                  #.map(custom_augment, num_parallel_calls=AUTO)
                  .batch(unlabeled_batch_size)
                  .prefetch(tf.data.AUTOTUNE)
                  )    
    
    # Combine datasets into one
    ssl_ds = tf.data.Dataset.zip((dataset_one, dataset_two))
    
    # Ensure that the different versions of the dataset actually contain identical images.
    if not os.path.exists(os.path.join('traindata', 'sample_images_one.png')):
        sample_images_one = next(iter(dataset_one))
        fig = plt.figure(figsize=(15, 15))
        for n in range(25):
            ax = plt.subplot(5, 5, n + 1)
            plt.plot(sample_images_one[n])
            plt.axis("off")
        #plt.show()
        plt.savefig(os.path.join('traindata', 'sample_images_one.png'))
        plt.close(fig)
        
        sample_images_two = next(iter(dataset_two))
        fig = plt.figure(figsize=(15, 15))
        for n in range(25):
            ax = plt.subplot(5, 5, n + 1)
            plt.plot(sample_images_two[n])
            plt.axis("off")
        #plt.show()
        plt.savefig(os.path.join('traindata', 'sample_images_two.png'))
        plt.close(fig)
    
    
    #%% Contrastive pretraining
    tf.keras.backend.clear_session()
    # Create and compile encoder and projection layers
    encoder = RespNet(maxlen=maxlen, include_top=True, classes=latent_dim)
    plot_model(encoder, to_file=os.path.join(model_savepath, 'encoder.png'), show_shapes=True)
    
    projection_layer = get_projection(project_dim=project_dim)
    plot_model(projection_layer, to_file=os.path.join(model_savepath, 'projection.png'), show_shapes=True)
    
    # Create the contrastive model
    contrastive_model = ContrastiveModel(temperature, encoder, projection_layer)
    contrastive_model.compile(contrastive_optimizer=tf.keras.optimizers.Adam())
    
    # Load previous model weights if specified
    if keep_train:
        model_history_path = os.path.join(rootpath, 'weights', previous_training_time, previous_model_name)
        contrastive_model.build(input_shape=(None, 300, 2))
        contrastive_model.load_weights(model_history_path, by_name=True)
    
    # Define model checkpointing callback    
    checkpointer = ModelCheckpoint(os.path.join(model_savepath, 'model_{epoch:03d}_{c_acc:.03f}_{c_loss:.03f}.h5'),
                                   verbose=0, 
                                   save_weights_only=True, 
                                   period=1)
    
    # Train the model
    training_history = contrastive_model.fit(ssl_ds,
                                             epochs=epochs, 
                                             verbose=1,
                                             callbacks=[checkpointer])
    print("Maximal validation accuracy: {:.2f}%".format(max(training_history.history["c_acc"]) * 100))
    
    # Save encoder and projection weights
    encoder.save_weights(os.path.join(model_savepath, "encoder.h5"))
    projection_layer.save_weights(os.path.join(model_savepath, "projection.h5"))
    
    # Plot training curves
    plot_contrast_training_curves(training_history, 
                                  savepath=os.path.join(model_savepath, 'acc-loss_contrast_train.png'), 
                                  save_history=os.path.join(model_savepath, 'history_pretraining.npy'))
    
    T2 = time.perf_counter()
    print(f'Spend about {round((T2 - T1)/3600, 2)} h')