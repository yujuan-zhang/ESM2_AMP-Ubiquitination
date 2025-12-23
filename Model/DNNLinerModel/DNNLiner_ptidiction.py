import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    matthews_corrcoef, f1_score, recall_score, accuracy_score, roc_auc_score, average_precision_score
)
import random

### Define necessary components
###============================================================================

class DNNLinearModel(nn.Module):
    """
    A simple Deep Neural Network (DNN) model with a single linear layer.
    Used as a classification head in combined models.
    """
    
    def __init__(self, input_dim):
        """
        Initializes the DNNLinearModel.
        
        :param input_dim: Dimension of the input features.
        """
        super(DNNLinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)
        
        # Xavier Initialization for the weights
        init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        """
        Forward pass through the model.
        
        :param x: Input tensor
        :return: Output logits
        """
        x = self.fc1(x)
        return x

    def infor(self, x):
        """
        Alternative forward method that returns the same output as `forward`.
        
        :param x: Input tensor
        :return: Output logits
        """
        x = self.fc1(x)
        return x


# Define Dataset class
class SequenceDataset(Dataset):
    """
    Custom Dataset for handling sequence data grouped by 'ID'.
    """
    
    def __init__(self, features, info_df):
        """
        Initializes the dataset with features and corresponding information.
        
        :param features: Numpy array of feature vectors
        :param info_df: DataFrame containing additional information, including 'ID' and 'Label'
        """
        self.features = features
        self.info_df = info_df
        self.groups = self.info_df.groupby('ID')  # Group data by 'ID' column

    def __len__(self):
        """
        Returns the number of groups in the dataset.
        """
        return len(self.groups)

    def __getitem__(self, idx):
        """
        Retrieves the features and labels for a specific group.
        
        :param idx: Index of the group
        :return: Tuple of feature values and labels
        """
        group_key = list(self.groups.groups.keys())[idx]
        group = self.groups.get_group(group_key)
        feature_value = self.features[group.index]
        labels = group['Label'].values
        return feature_value, labels


def custom_collate_fn(batch):
    """
    Custom collate function to combine features and labels from different samples into a batch.
    
    :param batch: List of tuples (features, labels)
    :return: Batch of features and labels as tensors
    """
    features_list = []
    labels_list = []
    for features, labels in batch:
        # Convert features and labels to tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        features_list.append(features_tensor)
        labels_list.append(labels_tensor)
        
    # Stack features vertically
    batch_features = torch.vstack(features_list)
    # Concatenate labels
    batch_labels = torch.cat(labels_list)

    return batch_features, batch_labels


# Validation function
def cvae_val(model, device, val_loader, beta=1.0, gamma=1.0):
    """
    Validates the CVAE model on the validation dataset and computes evaluation metrics.
    
    :param model: The CVAE model to validate
    :param device: The device to run the validation on
    :param val_loader: DataLoader for the validation dataset
    :param beta: Weight for the VAE reconstruction loss
    :param gamma: Weight for the classification loss
    :return: Dictionary containing various evaluation metrics
    """
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.float().to(device)
            labels = labels.float().to(device)

            # Forward pass
            class_logits = model.infor(features)

            # Collect predictions for classification task
            predictions.extend(torch.sigmoid(class_logits).cpu().numpy().flatten())
            targets.extend(labels.cpu().numpy().flatten())

    # Convert probabilities to binary predictions using a threshold of 0.4
    predictions_binary = [1 if p > 0.4 else 0 for p in predictions]
    metrics = {
        "MCC": matthews_corrcoef(targets, predictions_binary),
        "F1": f1_score(targets, predictions_binary),
        "Recall": recall_score(targets, predictions_binary),
        "Accuracy": accuracy_score(targets, predictions_binary),
        "AUC": roc_auc_score(targets, predictions),
        "PR": average_precision_score(targets, predictions)
    }

    return metrics


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across various libraries and environments.
    
    :param seed: The seed value to set
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    
    # Set seed for NumPy's random generator
    np.random.seed(seed)
    
    # Set seed for PyTorch's random generator
    torch.manual_seed(seed)
    
    # Set seed for CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Set all GPU seeds
    
    # Disable cuDNN benchmarking to ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Call function to set seed
set_seed(42)
###############################################################################

# Enable anomaly detection for debugging purposes
torch.autograd.set_detect_anomaly(True)

# Define paths
open_path = r'Your path to project'  # Placeholder path

# Load data
test_features = np.load(os.path.join(open_path, 'ESM2-Ubiquitination-Prediction', 'Inference_test_data', 'ESM2_3B_2560', 'test_features.npy'))
test_info_df = pd.read_csv(os.path.join(open_path, 'ESM2-Ubiquitination-Prediction', 'Inference_test_data', 'ESM2_3B_2560', 'test_info.csv'))


## Independent test set
test_dataset = SequenceDataset(test_features, test_info_df)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn, drop_last=False)


# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


input_dim = 2560
model = DNNLinearModel(input_dim).to(device)

# Load the trained model weights
model.load_state_dict(torch.load(os.path.join(open_path, "ESM2-Ubiquitination-Prediction", "Model", "DNNLinerModel", "DNNLinermodel_checkpoint_epoch_34.pth")))
model.to(device)

# Model inference
val_metrics = cvae_val(
    model, device, test_loader, beta=1.0, gamma=1.0
)

# Print the evaluation metrics
print(val_metrics)
