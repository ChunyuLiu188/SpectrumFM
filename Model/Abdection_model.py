import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import copy


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Only 2 convolution layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 1), padding=(1, 0))

        # BatchNorm layers
        self.batchnorm1 = nn.BatchNorm2d(num_features=32)
        self.batchnorm2 = nn.BatchNorm2d(num_features=64)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 128 * 2, 128)
        self.fc2 = nn.Linear(128, 256)

    def forward(self, x):
        # Convolution + BatchNorm + ReLU
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))

        # Flatten the tensor before passing through fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # Fully connected layers
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64 * 128 * 2)

        # Only 2 transposed convolution layers
        self.t_conv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 1), padding=(1, 0))
        self.t_conv2 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(3, 1), padding=(1, 0))

        # BatchNorm layers for the transposed convolutions
        self.batchnorm6 = nn.BatchNorm2d(num_features=32)

    def forward(self, x):
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # Reshape to match the expected input shape for the transposed convolutions
        x = x.view(-1, 64, 128, 2)

        # Transposed convolution layers with BatchNorm and ReLU activations
        x = F.relu(self.batchnorm6(self.t_conv1(x)))

        # Final transposed convolution to get the original signal
        x = F.tanh(self.t_conv2(x))

        return x
       


class AEModel(nn.Module):
    def __init__(self):
        super(AEModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x_rec = self.decoder(self.encoder(x))
        loss = torch.mean(torch.pow(x - x_rec, 2), dim=[1,2,3])
        # loss = torch.mean(torch.abs(x - x_rec), dim=[1,2,3])
        
        return loss
    
    def continuetrain(self, x):
        x = x.unsqueeze(1)
        return self.encoder(x)
    
    def test(self, x):
        x = x.unsqueeze(1)
        x_rec = self.decoder(self.encoder(x))
        loss = torch.mean(torch.pow(x - x_rec, 2), dim=[1,2,3])
        return loss
    
        

        
        