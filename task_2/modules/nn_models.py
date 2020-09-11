import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegression(nn.Module):
    ''' Fully-connected Neural Network with 1 layer with sigmoid '''
    def __init__(self, height=84, width=64):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(height * width, 1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = self.fc(x)
        out = torch.sigmoid(out)
        return out


class FullyConnectedNN(nn.Module):
    ''' Fully-connected Neural Network with 3 layers, ReLU and sigmoid '''

    def __init__(self, height=84, width=64):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(height * width, 700)
        self.fc2 = nn.Linear(700, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = torch.sigmoid(out)
        return out


class ConvolutionalNN(nn.Module):
    ''' 
    Convolutional Neural Network with 9 layers 
    
    Layer 1:
    - 3x3 Conv C_in = 1, C_out = 16
    - Batch Normalization, ReLU
    
    Layer 2:
    - 3x3 Conv C_in = 16, C_out = 16
    - Batch Normalization, ReLU
    - 2x2 MaxPool 

    Layer 3:
    - 3x3 Conv C_in = 16, C_out = 32
    - Batch Normalization, ReLU
    
    Layer 4:
    - 3x3 Conv C_in = 32, C_out = 32
    - Batch Normalization, ReLU
    - 2x2 MaxPool 

    Layer 5:
    - 3x3 Conv C_in = 32, C_out = 64
    - Batch Normalization, ReLU
    - 2x2 MaxPool

    Layer 6:
    - 3x3 Conv C_in = 64, C_out = 64
    - Batch Normalization, ReLU
    - 2x2 MaxPool

    Layer 7:
    - Linear D_in = 64 * 4 * 4, D_out = 512; + ReLU

    Layer 8:
    - Linear D_in = 512, D_out = 64; + ReLU

    Layer 9:
    - Linear D_in = 64, D_out = 1; + Sigmoid

    All convolutions have stride = 1, padding = 1
    and all MaxPools have stride = 2
    '''
    def __init__(self):
        super(ConvolutionalNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 32 x 32
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 16 x 16
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 8 x 8
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 4 x 4
        
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = torch.sigmoid(out)
        return out
