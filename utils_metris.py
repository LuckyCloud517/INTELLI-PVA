# -*- coding: utf-8 -*-

import os, json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score
#%%
def get_metris(y_true, y_predict, y_score):
    
    # Initialize an empty dictionary to store the calculated metrics
    result = {}

    # Extract values from the confusion matrix (True Negatives, False Positives, False Negatives, True Positives)
    TN, FP, FN, TP = confusion_matrix(y_true, y_predict, labels=[0, 1]).ravel()

    # Calculate Accuracy: proportion of correct predictions
    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Calculate Sensitivity (Recall, True Positive Rate): proportion of actual positives correctly identified
    Sensitivity = TP / (TP + FN)

    # Calculate Specificity (True Negative Rate): proportion of actual negatives correctly identified
    Specificity = TN / (FP + TN)

    # Calculate Positive Predictive Value (Precision): proportion of predicted positives correctly identified
    PositivePredictiveValue = TP / (TP + FP)

    # Calculate Negative Predictive Value: proportion of predicted negatives correctly identified
    NegativePredictiveValue = TN / (TN + FN)

    # Calculate F1 Score: harmonic mean of Sensitivity and Precision
    F1_Score = 2 * (Sensitivity * PositivePredictiveValue) / (Sensitivity + PositivePredictiveValue)

    # Calculate AUC (Area Under the Receiver Operating Characteristic Curve): evaluates the model's ability to discriminate between classes
    auc = roc_auc_score(y_true, y_score)

    # Store the calculated metrics in the result dictionary
    result = {
        'acc': Accuracy,  # Accuracy
        'sen': Sensitivity,  # Sensitivity
        'spe': Specificity,  # Specificity
        'ppv': PositivePredictiveValue,  # Positive Predictive Value
        'npv': NegativePredictiveValue,  # Negative Predictive Value
        'f1': F1_Score,  # F1 Score
        'auc': auc  # AUC
    }

    return result


def plot_confusion_matrix(cm, savedir, savename, class_names):
    # Configure the font to support Chinese characters in the plot
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Set Seaborn style for the plot
    sns.set()
    
    # Create the figure for the heatmap
    fig = plt.figure(figsize=(14, 12), 
                     dpi=100,
                     tight_layout=True
                     )
    # Plot the confusion matrix as a heatmap
    sns.heatmap(
        cm,
        annot=True,  # Display values inside each cell of the heatmap
        annot_kws={"size": 14},  # Customize annotation text size
        cmap='hot_r',  # Use red-hot color scheme
        fmt='.20g',  # Set the format for displaying numbers
        square=True,  # Make the heatmap cells square
        xticklabels=class_names,  # Set the x-axis labels (class names)
        yticklabels=class_names,  # Set the y-axis labels (class names)
    )
    
    # Customize tick labels and add title
    plt.yticks(rotation=0)
    plt.tick_params(labelsize = 14)
    fig.axes[0].set_title('Confusion matrix', fontsize = 20) # Title                                                     
    fig.axes[0].set_xlabel('Predict', fontsize = 20) # x-axis label
    fig.axes[0].set_ylabel('True', fontsize = 20)  # y-axis label
    
    # Save the heatmap plot as an image file
    plt.savefig(os.path.join(savedir, f'confusion_matrix_{savename}.png'), dpi = 100)
    plt.close(fig)

         
def save_results_as_json(savedir, result):
    # Save the result dictionary as a JSON file with indentation for readability
    with open(os.path.join(savedir, "result.json"), "w", encoding='utf-8') as f: 
        f.write(json.dumps(result, indent=2))  


def load_results_from_json(filepath):
    # Load and return the contents of a JSON file as a Python dictionary
    with open(filepath,"r", encoding='utf-8'  ) as f:
        train_result = json.load(f)
    return train_result


