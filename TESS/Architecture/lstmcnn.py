import torch.nn as nn

class lstmcnn(nn.Module):
    '''
    Architecture Description:
    1. LSTM-CNN Hybrid: Integrates Long Short-Term Memory (LSTM) networks for temporal sequence processing with 1D Convolutional Neural Networks (CNN).
    2. LSTM Encoder: Processes the input sequence using a 2-layer LSTM to capture time-dependent features.
    3. Hidden State Extraction: Extracts the final hidden state of the LSTM to summarize the entire sequence into a single vector.
    4. Feature Refinement: Treats the LSTM's hidden state vector as a 1D signal and refines it using three 1D convolutional blocks.
    5. Classification: Flattens the refined features and passes them through two dense layers to predict the emotion class.
    '''
    def __init__(self, features, classes):
        super(lstmcnn, self).__init__()
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.elu = nn.ELU()
        self.pool = nn.AvgPool1d(kernel_size=3, stride=3)

        self.lstm = nn.LSTM(input_size=33, hidden_size=64, num_layers=2, batch_first=True)  
        
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
        print(x.shape)

        x, _ = self.lstm(x)
        print(x.shape)

        x = x[:, -1, :] # for time series prediction https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.bn1(x)
        x = self.dropout(x)
        print(x.shape)

        x = self.conv2(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.bn2(x)
        x = self.dropout(x)
        print(x.shape)

        x = self.conv3(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.bn3(x)
        x = self.dropout(x)
        print(x.shape)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        print(x.shape)
        return x
