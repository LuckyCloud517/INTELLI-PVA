# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(history, savepath, save_history='history.npy'):
    # Function to plot training progress (accuracy and loss) and save it
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    # Plot training accuracy (red) and validation accuracy (blue)
    p1, = ax1.plot(history.history["accuracy"], 'r', label="train_acc")
    p2, = ax1.plot(history.history["val_accuracy"], 'b', label="val_acc") 
    ax1.set_ylabel('Accuracy')  # Y-axis label for accuracy
    ax1.set_xlabel('Epoch')  # X-axis label for epoch
    ax1.set_title("Training Progress")  # Title of the plot
    ax1.grid()  # Display grid on the plot
    
    ax2 = ax1.twinx()  # Create another Y-axis for loss
    p3, = ax2.plot(history.history["loss"], 'g', label="train_loss")  # Plot training loss (green)
    p4, = ax2.plot(history.history["val_loss"], 'k', label="val_loss")  # Plot validation loss (black)
    ax2.set_ylabel('Loss')  # Y-axis label for loss
    
    # Add a legend for both accuracy and loss curves
    ax1.legend(handles=[p1, p2, p3, p4], loc="upper right")
    #plt.show()
    
    # Save the plot as an image
    plt.savefig(savepath)
    
    # Optionally save the history as a numpy file
    if save_history!='':
        np.save(save_history, history.history)
        

def plot_contrast_training_curves(pretraining_history, savepath, save_history='history.npy'):
    # Function to plot contrastive training curves (accuracy and loss)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    # Plot contrastive accuracy (red)
    p1, = ax1.plot(pretraining_history.history["c_acc"], 'r', label="c_acc")
    ax1.set_ylabel('Accuracy')  # Y-axis label for accuracy
    ax1.set_xlabel('Epoch')  # X-axis label for epoch
    ax1.set_title("Training Progress")  # Title of the plot
    ax1.grid()  # Display grid on the plot
    
    ax2 = ax1.twinx()  # Create another Y-axis for loss
    p4, = ax2.plot(pretraining_history.history["c_loss"], 'k', label="c_loss")  # Plot contrastive loss (black)
    ax2.set_ylabel('Loss')  # Y-axis label for loss
    
    # Add a legend for accuracy and loss curves
    ax1.legend(handles=[p1, p4], loc="upper right")
    
    # Save the plot as an image
    #plt.show()
    plt.savefig(savepath)
    plt.close(fig)
    
    # Optionally save the history as a numpy file
    if save_history!='':
        np.save(save_history, pretraining_history.history)
        
   
def plot_loss_multi_task(history, save_dir, save_history='history.npy'):
    # Function to plot multi-task training progress (accuracy and loss for multiple tasks)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    # Calculate average accuracy across multiple tasks and plot (red for training accuracy)
    avg_train_acc = (
        np.array(history.history['DelayedCycling_accuracy']) +
        np.array(history.history['PrematureCycling_accuracy']) +
        np.array(history.history['DoubleTrig_accuracy']) +
        np.array(history.history['InefTrig_accuracy'])
    ) / 4
    p1, = ax1.plot(avg_train_acc, 'r', label="train_acc")
    
    # Calculate average validation accuracy and plot (blue for validation accuracy)
    avg_val_acc = (
        np.array(history.history['val_DelayedCycling_accuracy']) +
        np.array(history.history['val_PrematureCycling_accuracy']) +
        np.array(history.history['val_DoubleTrig_accuracy']) +
        np.array(history.history['val_InefTrig_accuracy'])
    ) / 4
    p2, = ax1.plot(avg_val_acc, 'b', label="val_acc")
    
    ax1.set_ylabel('Accuracy')  # Y-axis label for accuracy
    ax1.set_xlabel('Epoch')  # X-axis label for epoch
    ax1.set_title("Training Progress")  # Title of the plot
    ax1.grid()  # Display grid on the plot

    ax2 = ax1.twinx()  # Create another Y-axis for loss
    p3, = ax2.plot(history.history["loss"], 'g', label="train_loss")  # Plot training loss (green)
    p4, = ax2.plot(history.history["val_loss"], 'k', label="val_loss")  # Plot validation loss (black)
    ax2.set_ylabel('Loss')  # Y-axis label for loss

    # Add a legend for both accuracy and loss curves
    ax1.legend(handles=[p1, p2, p3, p4], loc="upper right")
    
    # Save the plot as an image in the specified directory
    plt.savefig(os.path.join(save_dir, "acc-loss.png"))
    #plt.show()
    plt.close(fig)
    
    # Optionally save the history as a numpy file
    if save_history!='':
        np.save(save_history, history.history)


def load_history(path):
    # Function to load saved training history from a numpy file
    history = np.load(path, allow_pickle=True)[()]
    return history

