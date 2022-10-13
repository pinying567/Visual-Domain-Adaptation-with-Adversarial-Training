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
        self.layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        h = self.conv(x).view(-1, self.z_dim) # (N, hidden_dims)
        h = self.layer(h)
        return h
    
    def load_state_dict(self, state_dict):
        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param)
        
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

class Classifier(nn.Module):
    
    def __init__(self, input_size=256, num_classes=10):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, h):
        c = self.fc(h)
        return c
    
    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        
        for name, param in state_dict.items():
            own_name = name.replace('layer.4', 'fc')
            if own_name in own_state:
                own_state[own_name].copy_(param)
        
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

class Discriminator(nn.Module):
    
    def __init__(self, input_size=256):
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