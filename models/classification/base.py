import torch
import torch.nn as nn


class BaseNet(nn.Module):
    def __init__(self, nclass):
        self.nclass = nclass
        
        self.encoder = nn.Sequential()
        self.classifier = nn.Sequential()

    def forward(self, input):
        encoding = self.encoder(input)
        output = self.classifier(encoding)
        return output
