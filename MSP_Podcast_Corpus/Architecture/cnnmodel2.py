import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.nn.utils import weight_norm
import torch.nn.functional as F

class cnn(nn.Module):
    """
    1D Convolutional Neural Network (CNN) for audio emotion recognition.
    
    Architecture:
    - Input: (Batch, Time, Features) -> Permuted to (Batch, Features, Time)
    - Layer 1: Conv1d (64 filters, k=3) -> SELU -> MaxPool -> BatchNorm -> Dropout
    - Layer 2: Conv1d (32 filters, k=3) -> SELU -> MaxPool -> BatchNorm -> Dropout
    - Layer 3: Conv1d (64 filters, k=3) -> SELU -> MaxPool -> BatchNorm -> Dropout
    - Flatten
    - Dense Layers: 64->32 -> 32->16 -> 16->Classes
    
    Args:
        min_length (int): Minimum sequence length (used for slicing input).
        features (int): Number of input features (channels).
        classes (int): Number of output classes.
    """
    def __init__(self, min_length, features, classes):
        super(cnn, self).__init__()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.gelu = nn.GELU()
        self.selu = nn.SELU()
        self.sigmoid = nn.Sigmoid()
        self.tanH=nn.Tanh()

        # nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')   

        self.conv1 = nn.Conv1d(in_channels=features, out_channels=64, kernel_size=3, stride=1, padding=1)          
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.bn1=nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)  
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.bn2=nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.bn3=nn.BatchNorm1d(64)

        self.dropout= nn.Dropout(0.3)
        self.dense1 = nn.Linear(64, 32)
        self.dense2 = nn.Linear(32, 16)
        self.dense3 = nn.Linear(16, classes)
        
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, src_mask, utterance, min_length):   
        # print(x.shape)
        # print(x[:20])
        x=x[:,:min_length,:]
        x=x.float()
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.selu(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.selu(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.selu(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.dropout(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x
