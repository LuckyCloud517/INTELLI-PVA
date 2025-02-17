# -*- coding: utf-8 -*-

import os
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from data.dataset import fully_supervised_dataset
from models.loss import plot_loss_multi_task
from models.model import get_projection, ContrastiveModel, RespNet
from utils.utils_metris import plot_confusion_matrix, get_metris, save_results_as_json

#%% Algorithm hyperparameters
T1 = time.perf_counter()

exp_name = 'exp2'  # Name of the experiment
ablation_method = '12-Combinations' # Ablation method being used
previous_training_checkpoint_time = '2025-02-17-16-28-56'  # Pre-trained model checkpoint time

# Create necessary directories for saving results
rootpath = os.path.join(os.getcwd(), exp_name, ablation_method)
if not os.path.isdir(rootpath):
    os.makedirs(rootpath)
    os.makedirs(os.path.join(rootpath, 'weights'))
    os.makedirs(os.path.join(rootpath, 'predict'))
    
    
latent_dim = 128  # Latent space dimension
project_dim = 128  # Projection dimension for contrastive loss
temperature = 0.1  # Temperature for contrastive loss

num_epochs = 1000  # Number of training epochs
batch_size = 256  # Batch size for training
seed = 66  # Random seed for reproducibility
maxlen = 300  # Max length of input sequence
n_splits = 10  # Number of splits for cross-validation
count = 0
train_result = {}

# Labels for classification tasks
labels = ['Delayed Cycling', 'Premature Cycling', 'DoubleTrig', 'InefTrig', 'Other']

# Load dataset
x, y, x_index = fully_supervised_dataset(maxlen=maxlen)

# Initialize StratifiedKFold for cross-validation
kfold = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True) 

#%% Start cross-validation loop
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
    
    # Create and load pre-trained contrastive model
    encoder = RespNet(maxlen=maxlen, include_top=True, classes=latent_dim)
    #plot_model(encoder, to_file=os.path.join(rootpath, 'weights', 'encoder.png'), show_shapes=True)
    
    projection_layer = get_projection(project_dim=project_dim)
    #plot_model(projection_layer, to_file=os.path.join(rootpath, 'weights', 'projection.png'), show_shapes=True)
    
    contrastive_model = ContrastiveModel(temperature, encoder, projection_layer)
    contrastive_model.compile(contrastive_optimizer=tf.keras.optimizers.Adam())
    
    # Load weights from the previous training checkpoint
    model_history_path = os.path.join(rootpath, 'weights', previous_training_checkpoint_time, 'model_010_0.236_3.338.h5')
    contrastive_model.build(input_shape=(None, 300, 2))
    contrastive_model.load_weights(model_history_path, by_name=True)
    
    
    #%% Set up classification model with pre-trained encoder weights
    model = RespNet(maxlen=maxlen, classify=True, active_learner=False, include_top=False)
    
    for i in range(len(model.layers)-6):
        print(f'model layer {i}: {model.layers[i].name},  encoder layer {i}: {encoder.layers[i].name}')
        model.layers[i].set_weights(encoder.layers[i].get_weights())
    
    for i in range(len(model.layers)-6):
        model.layers[i].trainable = False
        
    plot_model(model, to_file=os.path.join(rootpath, 'RespNet.png'), show_shapes=True)

    model_name = f'model_{count:03d}.h5'
    model_path = os.path.join(rootpath, 'weights', model_name)
    
    # Set up callbacks for model training
    model_checkpoint_callback = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
    early_stopper_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=1, restore_best_weights=True)
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, cooldown=5, min_lr=1e-9)

    # Compile and train the model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        shuffle=True,
                        verbose=0,
                        validation_split=0.3,
                        callbacks=[early_stopper_callback, reduce_lr_callback, model_checkpoint_callback])
    
    # Save loss plot and training history
    savepath = os.path.join(rootpath, 'predict', str(count))
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    plot_loss_multi_task(history, save_dir=savepath, save_history=os.path.join(savepath, 'history.npy'))

    #%% Validation  
    proba = model.predict(x_test, batch_size=batch_size)
    
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
    
    # Update results dictionary with metrics for each label
    tmp = {}
    for col in range(len(labels[:-1])):
        tmp.update({labels[col] : get_metris(y_test[col], pred_label[col], proba[col][:,1])})
    train_result.update({count : tmp})
    
    count = count + 1# Increment fold counter

# Save all results to a JSON file    
save_results_as_json(rootpath, train_result)
    
T2 = time.perf_counter()
print(f'Spend about {round((T2 - T1)/3600, 2)} h')   
    
    
    
