import torch.nn as nn

class lstmcnn(nn.Module):
    """
    A hybrid LSTM-CNN model for audio classification.
    
    Architecture:
    - Input: (batch_size, time_steps, features)
      Note: Hardcoded to expect 188 features (input_size=188).
    - LSTM (2 layers, hidden 64) -> Take last time step
    - Reshape for Conv1d
    - Conv1d (7 filters) -> ELU -> AvgPool -> BatchNorm -> Dropout
    - Conv1d (32 filters) -> ELU -> AvgPool -> BatchNorm -> Dropout
    - Conv1d (32 filters) -> ELU -> AvgPool -> BatchNorm -> Dropout
    - Flatten
    - Dense layers -> Output
    """
    def __init__(self, features, classes):
        super(lstmcnn, self).__init__()
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.elu = nn.ELU()
        self.pool = nn.AvgPool1d(kernel_size=3, stride=3)

        self.lstm = nn.LSTM(input_size=188, hidden_size=64, num_layers=2, batch_first=True)  
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=7, kernel_size=3, stride=1)  
        self.bn1=nn.BatchNorm1d(7)

        self.conv2 = nn.Conv1d(in_channels=7, out_channels=32, kernel_size=3, stride=1)  
        self.bn2=nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1)  
        self.bn3=nn.BatchNorm1d(32)

        self.dropout= nn.Dropout(0.3)

        self.dense1 = nn.Linear(32, 16)
        self.dense2 = nn.Linear(16, classes)

        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)
    

    def forward(self, x, i):   
        x = x.float()
        x = x[:,:,:188]
        #print(x.shape)

        x, _ = self.lstm(x)
        #print(x.shape)

        x = x[:, -1, :] # for time series prediction https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.bn1(x)
        x = self.dropout(x)
        #print(x.shape)

        x = self.conv2(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.bn2(x)
        x = self.dropout(x)
        #print(x.shape)

        x = self.conv3(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.bn3(x)
        x = self.dropout(x)
        #print(x.shape)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        #print(x.shape)
        return x
