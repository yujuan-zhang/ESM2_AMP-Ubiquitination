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

def get_activation_fn():
    """
    Returns the activation function to be used in the model.
    Currently returns LeakyReLU with in-place operations disabled.
    """
    return nn.LeakyReLU(inplace=False)  # Disable in-place operations
    # return nn.LeakyReLU()  # Alternative activation function


def get_nonlinearity():
    """
    Returns the name of the non-linearity used.
    """
    return 'leaky_relu'


class ResDNNModel(nn.Module):
    """
    Residual Deep Neural Network (DNN) Model used as a classifier.
    """
    
    class ResBlock(nn.Module):
        """
        Residual Block used in the ResDNNModel.
        """
        def __init__(self, in_dim, out_dim):
            """
            Initializes the Residual Block.
            
            :param in_dim: Input dimension.
            :param out_dim: Output dimension.
            """
            super(ResDNNModel.ResBlock, self).__init__()
            self.fc1 = nn.Linear(in_dim, out_dim)
            # Kaiming Normal Initialization for weights
            init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
            self.bn1 = nn.BatchNorm1d(out_dim)
            
            self.fc2 = nn.Linear(out_dim, out_dim)
            init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')
            self.bn2 = nn.BatchNorm1d(out_dim)
            
            self.activation_fn = get_activation_fn()
            
            if in_dim != out_dim:
                self.downsample = nn.Linear(in_dim, out_dim)
                init.kaiming_normal_(self.downsample.weight, nonlinearity='leaky_relu')
            else:
                self.downsample = None

        def forward(self, x):
            """
            Forward pass through the Residual Block.
            
            :param x: Input tensor.
            :return: Output tensor after applying residual connections.
            """
            identity = x
            out = self.activation_fn(self.bn1(self.fc1(x)))
            out = self.activation_fn(self.bn2(self.fc2(out)))
            
            if self.downsample is not None:
                identity = self.downsample(identity)
                
            out += identity
            return out

    def __init__(self, input_dim, layer_dims):
        """
        Initializes the ResDNNModel.
        
        :param input_dim: Dimension of the input features.
        :param layer_dims: List containing the dimensions of each layer.
        """
        super(ResDNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, layer_dims[0])
        init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        self.bn1 = nn.BatchNorm1d(layer_dims[0])
        
        self.resblocks = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.resblocks.append(self.ResBlock(layer_dims[i], layer_dims[i+1]))
        
        self.fc_out = nn.Linear(layer_dims[-1], 1)
        init.kaiming_normal_(self.fc_out.weight, nonlinearity='sigmoid')

    def forward(self, x):
        """
        Forward pass through the ResDNNModel.
        
        :param x: Input tensor.
        :return: Output logits.
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, inplace=False)  # Disable in-place operations
        for block in self.resblocks:
            x = block(x)
        x = self.fc_out(x)
        return x

    def infor(self, x):
        """
        Alternative forward method that returns the same output as `forward`.
        
        :param x: Input tensor.
        :return: Output logits.
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, inplace=False)  # Disable in-place operations
        for block in self.resblocks:
            x = block(x)
        x = self.fc_out(x)
        return x


# Define Dataset class
class SequenceDataset(Dataset):
    """
    Custom Dataset for handling sequence data grouped by 'ID'.
    """
    def __init__(self, features, info_df):
        """
        Initializes the dataset with features and corresponding information.
        
        :param features: Numpy array of feature vectors.
        :param info_df: DataFrame containing additional information, including 'ID' and 'Label'.
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
        
        :param idx: Index of the group.
        :return: Tuple of feature values and labels.
        """
        group_key = list(self.groups.groups.keys())[idx]
        group = self.groups.get_group(group_key)
        feature_value = self.features[group.index]
        labels = group['Label'].values
        return feature_value, labels


def custom_collate_fn(batch):
    """
    Custom collate function to combine features and labels from different samples into a batch.
    
    :param batch: List of tuples (features, labels).
    :return: Batch of features and labels as tensors.
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
    
    :param model: The CVAE model to validate.
    :param device: The device to run the validation on.
    :param val_loader: DataLoader for the validation dataset.
    :param beta: Weight for the VAE reconstruction loss.
    :param gamma: Weight for the classification loss.
    :return: Dictionary containing various evaluation metrics.
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
    
    :param seed: The seed value to set.
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


best_params = {
    'num_blocks': 1,
    'layer_0_dim': 1216,
    'layer_1_dim': 1216,
}

# Extract layer dimensions based on best parameters
num_blocks = best_params['num_blocks']
layer_dims = [int(best_params[f'layer_{i}_dim']) for i in range(int(num_blocks) + 1)]

input_dim = 2560
model = ResDNNModel(input_dim, layer_dims)
# Load the trained model weights
model.load_state_dict(torch.load(os.path.join(open_path, "ESM2-Ubiquitination-Prediction", "Model", "ResDNNModel", "ResDNNmodel_checkpoint_epoch_2.pth")))
model.to(device)

# Model inference
val_metrics = cvae_val(
    model, device, test_loader, beta=1.0, gamma=1.0
)

# Print the evaluation metrics
print(val_metrics)
