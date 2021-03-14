import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        # I'm sure there is a mathematical explanation to how upsampling of correct dimensions happen
        # But I don't know it and had to eyeball it...good luck changing or debugging this future me.
        self.deconv1 = nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0)
        self.deconv1_bn = nn.BatchNorm2d(num_features=512)
        
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=0)
        self.deconv2_bn = nn.BatchNorm2d(num_features=256)
        
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=0)
        self.deconv3_bn = nn.BatchNorm2d(num_features=128)
        
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=6, stride=1, padding=0)
        self.activation = nn.Tanh()

    def forward(self, input):
        
        # (1) Hidden ConvTranspose2D layer
        t = self.deconv1(input)
        t = self.deconv1_bn(t)
        t = F.relu(t)
        
        # (2) Hidden ConvTranspose2D layer
        t = self.deconv2(t)
        t = self.deconv2_bn(t)
        t = F.relu(t)
        
        # (3) Hidden ConvTranspose2D layer
        t = self.deconv3(t)
        t = self.deconv3_bn(t)
        t = F.relu(t)
        
        # (4) Upscale and ouput layer
        t = self.deconv4(t)
        # t = F.tanh(t)
        t = self.activation(t)
        
        return t