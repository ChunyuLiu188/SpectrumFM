"""Official AWN (Zhang et al., TCCN 2023). Input (N, 2, T). Returns (logit, regu_list)."""
import torch
import torch.nn as nn

from .awn_lifting import LiftingScheme


class LevelTWaveNet(nn.Module):
    def __init__(self, in_planes, kernel_size, regu_details, regu_approx):
        super().__init__()
        self.regu_details = regu_details
        self.regu_approx = regu_approx
        self.wavelet = LiftingScheme(in_planes, kernel_size=kernel_size)

    def forward(self, x):
        L, H = self.wavelet(x)
        approx, details = L, H
        regu = x.new_zeros(())
        if self.regu_details:
            regu = regu + self.regu_details * H.abs().mean()
        if self.regu_approx:
            regu = regu + self.regu_approx * torch.dist(approx.mean(), x.mean(), p=2)
        return approx, details, regu


class AWN(nn.Module):
    def __init__(self, num_classes, num_levels=1, in_channels=64, kernel_size=3,
                 latent_dim=256, regu_details=0.01, regu_approx=0.01):
        super().__init__()
        self.num_levels = num_levels
        self.in_channels = in_channels
        self.out_channels = in_channels * (num_levels + 1)
        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(1, in_channels, kernel_size=(2, 7), stride=(1,), bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=(5,), stride=(1,), padding=(2,), bias=False),
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.levels = nn.ModuleList(
            [LevelTWaveNet(in_channels, kernel_size, regu_details, regu_approx) for _ in range(num_levels)]
        )
        self.SE_attention_score = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels // 4, bias=False),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels // 4, self.out_channels, bias=False),
            nn.Sigmoid(),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.out_channels, latent_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x).squeeze(2)
        x = self.conv2(x)
        regu_sum, det = [], []
        for level in self.levels:
            x, details, regu = level(x)
            regu_sum.append(regu)
            det.append(self.avgpool(details))
        det.append(self.avgpool(x))
        x = torch.cat(det, 1).view(x.size(0), -1)
        x = self.SE_attention_score(x) * x
        return self.fc(x), regu_sum
