import torch.nn as nn

class cnn(nn.Module):
    '''
    Architecture Description:
    1. Input Processing: Accepts 1D input tensors (e.g., MFCCs) with shape (Batch, Features, Time).
    2. Convolutional Blocks: Consists of 3 sequential blocks, each containing a 1D convolution, ReLU activation, Max Pooling, Batch Normalization, and Dropout.
    3. Feature Flattening: Flattens the 3D output of the final convolutional layer into a 1D vector.
    4. Classification Head: Utilizes a Multi-Layer Perceptron (MLP) with three dense layers to map extracted features to class logits.
    5. Hardcoded Dimensions: Note that the first dense layer expects a fixed input size (2688), making the model dependent on a specific input sequence length.
    '''
    def __init__(self, features, classes):
        super(cnn, self).__init__()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=1)
        self.conv1 = nn.Conv1d(in_channels=features, out_channels=64, kernel_size=3, stride=1)          
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1)
        self.bn1=nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1)  
        self.bn2=nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, stride=1)  
        self.bn3=nn.BatchNorm1d(128)
        self.dropout= nn.Dropout(0.3)
        self.dense1 = nn.Linear(2688, 512)
        self.dense2 = nn.Linear(512, 64)
        self.dense3 = nn.Linear(64, classes)        
        self.flatten = nn.Flatten()
    
    def forward(self, x, i):   
        x = self.conv1(x.float())
        # print(x.shape)
        x = self.relu(x)
        x = self.pool(x)
        # print(x.shape)
        x = self.bn1(x)
        x = self.dropout(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        # print(x.shape) 
        x = self.bn2(x)
        x = self.dropout(x)
        # print(x.shape)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        # print(x.shape)
        x = self.bn3(x)
        x = self.dropout(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.dense1(x)
        # print(x.shape)
        x = self.dense2(x)
        # print(x.shape)
        x = self.dense3(x)
        # print(x.shape)
        return x
