import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from utils import create_mask
from einops import rearrange

class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        
        # 定义 GRU 层
        self.gru = nn.GRU(input_size=2, 
                          hidden_size=128, 
                          num_layers=2, 
                          batch_first=True, 
                          dropout=0.2)
        
        # 定义分类层
        self.fc = nn.Linear(128, 40)
        
    def forward(self, x):
        # GRU 输出
        gru_out, _ = self.gru(x)
        
        # 取最后一个时间步的输出
        last_hidden = gru_out[:, -1, :]
        
        # 分类
        output = self.fc(last_hidden)
        output = output.view(-1, 20, 2)
        
        return output