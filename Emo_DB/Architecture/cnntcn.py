import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.nn.utils
import math

#  https://github.com/locuslab/TCN

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CTCN(nn.Module):
    """
    A hybrid CNN-TCN (Temporal Convolutional Network) model.
    
    Architecture:
    - Input: (batch_size, time_steps, features)
      Note: Slices input to keep first 188 features (x[:,:,:188]).
    - Reshaped to (batch, 1, time, features) for Conv2d
    - 3x CNN Blocks: Conv2d -> SELU -> MaxPool2d -> BatchNorm -> Dropout
    - Reshape to (batch, channels, length)
    - TemporalConvNet (TCN) layers
    - Linear layers for classification
    """
    def __init__(self, features, classes, cnn_layer=[64, 32, 16], dropout_cnn=0.2, tcn_layer=[16]*16, dropout_tcn=0.2):
        super().__init__()
        # CNN Block 1
        self.relu=nn.ReLU()
        self.selu=nn.SELU()
        self.elu=nn.ELU()

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=cnn_layer[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(cnn_layer[0])
        self.maxp1 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        
        self.cnn2 = nn.Conv2d(in_channels=cnn_layer[0], out_channels=cnn_layer[1], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(cnn_layer[1])
        self.maxp2 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        
        self.cnn3 = nn.Conv2d(in_channels=cnn_layer[1], out_channels=cnn_layer[2], kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(cnn_layer[2])
        self.maxp3 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)

        self.dropout = nn.Dropout(dropout_cnn)

        self.tcn = TemporalConvNet(num_inputs=cnn_layer[2], num_channels=tcn_layer, kernel_size=3, dropout=dropout_tcn)
        
        self.linear1 = nn.Linear(tcn_layer[-1] * 1, 512)
        self.linear2 = nn.Linear(512, classes)
        

    def forward(self, x, i):
        # print()
        x = x[:,:,:188] # only mfcc in the dataset
        x = x.float()
        x = x.unsqueeze(1)
        # print(x.shape)
        x = self.cnn1(x)
        x = self.selu(x)
        x = self.maxp1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        # print(x.shape)

        x = self.cnn2(x)
        x = self.selu(x)
        x = self.maxp2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        # print(x.shape)
        # print(x.shape)

        x = self.cnn3(x)
        x = self.selu(x)
        x = self.maxp3(x)
        x = self.bn3(x)
        x = self.dropout(x)
        # print(x.shape)

        x = x.view(x.size(0), x.size(1), -1)
        # print(x.shape)
        x = self.tcn(x)
        # print(x.shape)

        [n,c,l] = x.size()
        end_len = 1
        x = x[:,:,-end_len:]
        x = x.transpose(1,2).contiguous()
        x = x.view(n,c*end_len)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        # print(x.shape)
        return x