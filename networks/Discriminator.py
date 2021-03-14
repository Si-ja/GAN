import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv1_bn = nn.BatchNorm2d(num_features=32)
        self.dropout1 = nn.Dropout(p=0.25)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2_bn = nn.BatchNorm2d(num_features=64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.dropout2 = nn.Dropout(p=0.4)
        
        self.fc1 = nn.Linear(in_features=7*7*64, out_features=512, bias=True)
        self.out = nn.Linear(in_features=512, out_features=1)
        self.activation = nn.Sigmoid()
        
    def forward(self, t):

        # (1) input layer
        t = t
        
        # (2) hidden conv layer
        t = self.conv1(t)
        t = self.conv1_bn(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = self.dropout1(t)
        
        # (3) hiddne conv layer
        t = self.conv2(t)
        t = self.conv2_bn(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = self.dropout2(t)
        
        # (4) hidden linear layer
        t = t.reshape(-1, 7*7*64)
        t = self.fc1(t)
        t = F.relu(t)
        
        # (5) output layer
        t = self.out(t)
        t = self.activation(t)
        #t = F.sigmoid(t)
        
        return t