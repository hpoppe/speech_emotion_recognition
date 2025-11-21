import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class cnn(nn.Module):
    '''
    Architecture Description:
    1. Input Permutation: Permutes the input tensor to (Batch, Channels, Time) to align with PyTorch's Conv1d requirements.
    2. Activation Function: Uses Scaled Exponential Linear Units (SELU) for self-normalizing properties, aiding in training stability.
    3. Strided Pooling: Employs Max Pooling with a stride of 3 to aggressively downsample the temporal dimension after each convolution.
    4. Deep Convolutional Stack: Stacks three 1D convolutional layers with varying channel depths to capture hierarchical temporal patterns.
    5. Dense Classification: Concludes with a 3-layer fully connected network to classify the flattened feature vector.
    '''
    def __init__(self, features, classes):
        super(cnn, self).__init__()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.gelu = nn.GELU()
        self.selu = nn.SELU()
        self.sigmoid = nn.Sigmoid()
        self.tanH=nn.Tanh()

        self.conv1 = nn.Conv1d(in_channels=33, out_channels=64, kernel_size=3, stride=1)          
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.bn1=nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1)  
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.bn2=nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.bn3=nn.BatchNorm1d(64)

        self.dropout= nn.Dropout(0.3)
        self.dense1 = nn.Linear(64, 32)
        self.dense2 = nn.Linear(32, 16)
        self.dense3 = nn.Linear(16, classes)
        
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, i):   
        # print(x.shape)
        # print(x[:20])
        x=x.permute(0,2,1)
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
