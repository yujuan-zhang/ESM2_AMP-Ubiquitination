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

class ContinuousResidualVAE(nn.Module):
    class ResBlock(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.fc = nn.Linear(in_dim, out_dim)
            self.bn = nn.BatchNorm1d(out_dim)
            self.dropout = nn.Dropout(0.3)
            if in_dim != out_dim:
                self.downsample = nn.Linear(in_dim, out_dim)
            else:
                self.downsample = None

        def forward(self, x):
            out = F.leaky_relu(self.bn(self.fc(x)))
            out = self.dropout(out)
            if self.downsample is not None:
                x = self.downsample(x)
            return out + x

    def __init__(self, input_dim, hidden_dim=1280, z_dim=100, loss_type='RMSE', reduction='mean'):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.resblock1 = self.ResBlock(hidden_dim, hidden_dim // 2)
        self.resblock2 = self.ResBlock(hidden_dim // 2, hidden_dim // 4)
        # Latent space
        self.fc21 = nn.Linear(hidden_dim // 4, z_dim)  # mu layer
        self.fc22 = nn.Linear(hidden_dim // 4, z_dim)  # logvariance layer
        # Decoder
        self.fc3 = nn.Linear(z_dim, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        self.dropout3 = nn.Dropout(0.3)
        self.resblock3 = self.ResBlock(hidden_dim // 4, hidden_dim // 2)
        self.resblock4 = self.ResBlock(hidden_dim // 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
        # Attributes for loss type and reduction type
        self.loss_type = loss_type
        self.reduction = reduction
        
        if reduction not in ['mean', 'sum']:
            raise ValueError("Invalid reduction type. Expected 'mean' or 'sum', but got %s" % reduction)

    def encode(self, x):
        h = F.leaky_relu(self.bn1(self.fc1(x)))
        h = self.dropout1(h)
        h = self.resblock1(h)
        h = self.resblock2(h)
        return self.fc21(h), self.fc22(h)  # mu, logvariance

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # Sample epsilon from a standard normal distribution with the same shape as std
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.leaky_relu(self.bn3(self.fc3(z)))
        h = self.dropout3(h)
        h = self.resblock3(h)
        h = self.resblock4(h)
        return self.fc4(h)  # No activation function here

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, x.shape[1]))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def get_model_inference_z(self, x, seed=None):
        """
        Takes input x and returns the corresponding latent vectors z.
        If a seed is provided, it makes the random number generator deterministic.
        """
        self.eval()  # Switch to evaluation mode
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():  # Disable gradient computation
            mu, logvar = self.encode(x.view(-1, x.shape[1]))
            z = self.reparameterize(mu, logvar)
        return z

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        """
        Computes the VAE loss including reconstruction loss and KL divergence.
        
        :param recon_x: Reconstructed input
        :param x: Original input
        :param mu: Mean from the encoder's latent space
        :param logvar: Log variance from the encoder's latent space
        :param beta: Weight for the reconstruction loss
        """
        if self.loss_type == 'MSE':
            self.REC = F.mse_loss(recon_x, x.view(-1, x.shape[1]), reduction=self.reduction)
        elif self.loss_type == 'RMSE':
            self.REC = torch.sqrt(F.mse_loss(recon_x, x.view(-1, x.shape[1]), reduction=self.reduction))
        else:
            raise ValueError(f'Invalid loss type: {self.loss_type}')

        if self.reduction == 'mean':
            self.KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else: 
            self.KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return beta * self.REC + self.KLD

    def print_neurons(self):
        """
        Prints the number of neurons in each layer of the encoder and decoder.
        """
        print("Encoder neurons:")
        print(f"Input: {self.fc1.in_features}, Output: {self.fc1.out_features}")
        print(f"ResBlock1 - Input: {self.resblock1.fc.in_features}, Output: {self.resblock1.fc.out_features}")
        print(f"ResBlock2 - Input: {self.resblock2.fc.in_features}, Output: {self.resblock2.fc.out_features}")
        
        print("Latent neurons:")
        print(f"mu - Input: {self.fc21.in_features}, Output: {self.fc21.out_features}")
        print(f"logvar - Input: {self.fc22.in_features}, Output: {self.fc22.out_features}")

        print("Decoder neurons:")
        print(f"Input: {self.fc3.in_features}, Output: {self.fc3.out_features}")
        print(f"ResBlock3 - Input: {self.resblock3.fc.in_features}, Output: {self.resblock3.fc.out_features}")
        print(f"ResBlock4 - Input: {self.resblock4.fc.in_features}, Output: {self.resblock4.fc.out_features}")
        print(f"Output: {self.fc4.in_features}, Output: {self.fc4.out_features}")

class DNNLinearModel(nn.Module):
    
    def __init__(self, input_dim):
        super(DNNLinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)
        
        # Xavier initialization for the weights
        init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        x = self.fc1(x)
        return x

class CVAEDNNLiner(ContinuousResidualVAE):
    def __init__(self, input_dim, hidden_dim=1280, z_dim=100, num_classes=1, loss_type='RMSE', reduction='mean'):
        super().__init__(input_dim, hidden_dim, z_dim, loss_type, reduction)
        
        # Use DNNLinearModel as the classification head
        self.classifier = DNNLinearModel(z_dim)

        # Use BCEWithLogitsLoss for classification loss
        self.classification_loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        """
        Forward pass that returns reconstructed output, classification logits, latent variables, and log variance.
        """
        mu, logvar = self.encode(x.view(-1, x.shape[1]))
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        class_logits = self.classifier(z)  # Classification head outputs logits
        return recon_x, class_logits, mu, logvar, z

    def infor(self, x):
        """
        Forward pass that returns classification logits.
        """
        mu, logvar = self.encode(x.view(-1, x.shape[1]))
        z = self.reparameterize(mu, logvar)
        class_logits = self.classifier(z)  # Classification head outputs logits
        return class_logits
    
    def get_z(self, x):
        """
        Forward pass that returns the latent vector z.
        """
        mu, logvar = self.encode(x.view(-1, x.shape[1]))
        z = self.reparameterize(mu, logvar)
        return z

    def loss_function(self, recon_x, x, class_logits, labels, mu, logvar, beta=1.0, gamma=1.0):
        """
        Computes the total loss including reconstruction loss, KL divergence, and classification loss.
        
        :param recon_x: Reconstructed output
        :param x: Original input
        :param class_logits: Classification head outputs
        :param labels: True labels for classification
        :param mu: Mean from the encoder's latent space
        :param logvar: Log variance from the encoder's latent space
        :param beta: Weight for the VAE reconstruction loss
        :param gamma: Weight for the classification loss
        """
        # VAE reconstruction loss and KL divergence
        vae_loss = super().loss_function(recon_x, x, mu, logvar, beta=beta)

        # Classification loss
        labels = labels.view_as(class_logits)
        classification_loss = self.classification_loss(class_logits, labels)

        # Total loss
        total_loss = vae_loss + gamma * classification_loss
        return total_loss, vae_loss, classification_loss

# Define Dataset class
class SequenceDataset(Dataset):
    def __init__(self, features, info_df):
        self.features = features
        self.info_df = info_df
        self.groups = self.info_df.groupby('ID')  # Group data by 'ID' column

    def __len__(self):
        return len(self.groups)
   
    def __getitem__(self, idx):
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

# Load test data
test_features = np.load(os.path.join(open_path, 'ESM2-Ubiquitination-Prediction', 'Inference_test_data', 'ESM2_3B_2560', 'test_features.npy'))
test_info_df = pd.read_csv(os.path.join(open_path, 'ESM2-Ubiquitination-Prediction', 'Inference_test_data', 'ESM2_3B_2560', 'test_info.csv'))

# Create an independent test set
test_dataset = SequenceDataset(test_features, test_info_df)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn, drop_last=False)

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize model parameters
input_dim = 2560
hidden_dim = 1280
z_dim = 100

# Initialize the CVAEDNNLiner model
model = CVAEDNNLiner(input_dim, hidden_dim, z_dim)

# Load the trained model weights
model.load_state_dict(torch.load(os.path.join(open_path, "ESM2-Ubiquitination-Prediction", "Model", "cVAE_DNNLinerModel", "CVAE_Z_checkpoint_epoch_75.pth")))

# Move the model to the specified device
model.to(device)

# Perform model inference on the test set
val_metrics = cvae_val(
    model, device, test_loader, beta=1.0, gamma=1.0
)

# Print the evaluation metrics
print(val_metrics)
