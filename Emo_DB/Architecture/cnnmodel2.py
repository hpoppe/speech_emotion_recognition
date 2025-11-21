import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from Architecture import tcn

class cnn(nn.Module):
    """
    A 1D Convolutional Neural Network (CNN) with SELU activation and custom pooling.
    
    Architecture:
    - Input: (batch_size, features, time_steps)
      Note: Hardcoded to expect 40 input channels (features).
    - Conv1d (64 filters, kernel 3) -> SELU -> MaxPool (stride 3) -> BatchNorm -> Dropout
    - Conv1d (32 filters, kernel 3) -> SELU -> MaxPool (stride 3) -> BatchNorm -> Dropout
    - Conv1d (64 filters, kernel 3) -> SELU -> MaxPool (stride 3) -> BatchNorm -> Dropout
    - Flatten
    - Dense (32 units)
    - Dense (16 units)
    - Dense (classes) -> Output
    """
    def __init__(self, features, classes):
        super(cnn, self).__init__()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.gelu = nn.GELU()
        self.selu = nn.SELU()
        self.sigmoid = nn.Sigmoid()
        self.tanH=nn.Tanh()

        self.conv1 = nn.Conv1d(in_channels=40, out_channels=64, kernel_size=3, stride=1)          
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.bn1=nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1)  
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.bn2=nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.bn3=nn.BatchNorm1d(64)

        self.dropout= nn.Dropout(0.3)
        self.dense1 = nn.Linear(384, 32)
        self.dense2 = nn.Linear(32, 16)
        self.dense3 = nn.Linear(16, classes)
        
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, i):   
        # print(x.shape)
        # print(x[:20])
        x=x.float()
        ''' to separate mfcc and time domain features 
        x_t=x[:,:,188:]
        x_mfcc=x[:,:,:188]
        '''
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
