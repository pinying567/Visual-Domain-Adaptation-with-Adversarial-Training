import torch
from torch import nn
from torch.nn import functional as F
import pdb

class FeatureExtractor(nn.Module):
    
    def __init__(self, in_dim=3, z_dim=512):
        super(FeatureExtractor, self).__init__()
        self.z_dim = z_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 64, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2)
        )
    
    def forward(self, x):
        h = self.conv(x).view(-1, self.z_dim) # (N, hidden_dims)
        return h
    
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

class Classifier(nn.Module):
    
    def __init__(self, input_size=512, num_classes=10):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, h):
        c = self.layer(h)
        return c
    
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

class Discriminator(nn.Module):
    
    def __init__(self, input_size=512):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, h):
        y = self.layer(h)
        return y

