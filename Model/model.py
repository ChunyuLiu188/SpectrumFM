import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from einops import rearrange
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel
import re
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0, bias=True):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

# ---------- LoRA for Conv1d ----------
class LoRAConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=4, alpha=1.0, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

        # 用两个 1x1 卷积做 LoRA 的 A 和 B
        self.lora_A = nn.Conv1d(in_channels, r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv1d(r, out_channels, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.conv(x) + self.lora_B(self.lora_A(x)) * self.scaling

def inject_lora(model, target_modules, r=4, alpha=1.0):
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            parent = get_parent_module(model, name)
            attr_name = name.split(".")[-1]
            orig_module = getattr(parent, attr_name)

            # 替换 Linear
            if isinstance(orig_module, nn.Linear):
                lora_module = LoRALinear(orig_module.in_features, orig_module.out_features, r=r, alpha=alpha, bias=orig_module.bias is not None)
                lora_module.linear.weight.data = orig_module.weight.data.clone()
                if orig_module.bias is not None:
                    lora_module.linear.bias.data = orig_module.bias.data.clone()
                setattr(parent, attr_name, lora_module)

            # 替换 Conv1d，但忽略 depthwise（groups == out_channels）
            elif isinstance(orig_module, nn.Conv1d) and orig_module.groups == 1:
                lora_module = LoRAConv1d(
                    orig_module.in_channels,
                    orig_module.out_channels,
                    orig_module.kernel_size[0],
                    r=r,
                    alpha=alpha,
                    stride=orig_module.stride[0],
                    padding=orig_module.padding[0],
                    dilation=orig_module.dilation[0],
                    bias=orig_module.bias is not None
                )
                lora_module.conv.weight.data = orig_module.weight.data.clone()
                if orig_module.bias is not None:
                    lora_module.conv.bias.data = orig_module.bias.data.clone()
                setattr(parent, attr_name, lora_module)


def get_parent_module(model, full_name):
    parts = full_name.split(".")
    for part in parts[:-1]:
        model = getattr(model, part)
    return model



class ReConstruction(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(ReConstruction, self).__init__() 
        # 用于重构的全连接层
        self.reconstruction_layer = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        
        reconstructed_signal = self.reconstruction_layer(x)
        return reconstructed_signal
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

# 1. 定义位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, model_dim, 2) * -(math.log(10000.0) / model_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # 添加 batch 维度
    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=129):
        super(RotaryPositionEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 创建一个位置索引
        self.register_buffer('position_ids', torch.arange(0, max_len).unsqueeze(1))

    def forward(self, x, seq_len=129):
        """
        seq_len: 当前输入序列的长度
        返回旋转后的相对位置编码
        """
        # 计算角度
        position_ids = self.position_ids[:seq_len]  # 取前 seq_len 个位置
        freqs = torch.pow(10000, -torch.arange(0, self.d_model, 2).float() / self.d_model).to(position_ids.device)
        freqs = freqs.unsqueeze(0)  # shape: (1, d_model//2)
        
        # 计算 sin 和 cos
        angles = position_ids.float() * freqs  # shape: (seq_len, d_model//2)
        sin = torch.sin(angles)
        cos = torch.cos(angles)

        # 合并 sin 和 cos
        position_embeddings = torch.cat([sin, cos], dim=-1)  # shape: (seq_len, d_model)
        
        return x + position_embeddings

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.query_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.key_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.value_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(0.2)
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask  # Apply mask (large negative value for masked positions)
        
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

    def forward(self, x, mask=None):
        batch_size, seq_len, model_dim = x.size()
        
        Q = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, model_dim)
        attention_output = self.out_proj(attention_output)
        attention_output = self.dropout(attention_output)
        return attention_output
class RelativePositionAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_len=512):
        super(RelativePositionAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_len = max_len
        
        # 查询、键、值的线性变换层
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        # 相对位置嵌入（相对位置差的最大范围）
        self.relative_positions = nn.Parameter(torch.randn(2 * max_len - 1, num_heads))  # shape: [2*max_len-1, num_heads]
        
        # 输出线性变换层
        self.output = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x, mask):
        batch_size, seq_len, _ = x.size()

        # 查询、键、值的计算
        Q = self.query(x)  # [batch_size, seq_len, embed_dim]
        K = self.key(x)    # [batch_size, seq_len, embed_dim]
        V = self.value(x)  # [batch_size, seq_len, embed_dim]

        # 将 embed_dim 分配给多个头部
        Q = Q.view(batch_size, seq_len, self.num_heads, self.embed_dim // self.num_heads)
        K = K.view(batch_size, seq_len, self.num_heads, self.embed_dim // self.num_heads)
        V = V.view(batch_size, seq_len, self.num_heads, self.embed_dim // self.num_heads)

        # 计算查询和键的点积
        attention_scores = torch.einsum('bqhd,bkhd->bhqk', Q, K)  # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = attention_scores / (self.embed_dim // self.num_heads) ** 0.5

        # 加上相对位置编码
        position_indices = torch.arange(seq_len, device=x.device).unsqueeze(0) - torch.arange(seq_len, device=x.device).unsqueeze(1)  # [seq_len, seq_len]
        position_indices = position_indices + self.max_len - 1  # 使得位置差从 0 开始
        position_indices = position_indices.clamp(min=0, max=2 * self.max_len - 2)  # 防止越界
        relative_position_embedding = self.relative_positions[position_indices]  # [seq_len, seq_len, num_heads]
        relative_position_embedding = relative_position_embedding.permute(2, 0, 1)  # [num_heads, seq_len, seq_len]
        relative_position_embedding = relative_position_embedding.unsqueeze(0)
        attention_scores += relative_position_embedding  # 将相对位置嵌入加到注意力得分上
        if mask is not None:
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            mask_value = -torch.finfo(attention_scores.dtype).max
            attention_scores = attention_scores.masked_fill(mask == 0, mask_value)
        # 计算注意力权重并应用于值
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
       
        output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V)  # [batch_size, seq_len, num_heads, embed_dim // num_heads]

        # 将多个头部合并并通过输出线性变换层
        output = output.contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.output(output)
        output = self.dropout(output)

        return output



class FeedForward(nn.Module):
    def __init__(self, model_dim, hidden_dim, dropout=0.2):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(model_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU() 
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.2):
        super(DepthwiseConv, self).__init__()
        
        # 1×1 卷积，用于通道变换
        self.pointwise1 = nn.Conv1d(in_channels, out_channels * 2, kernel_size=1)
        
        # 深度可分离卷积：每个通道独立卷积
        self.depthwise = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, 
                                   groups=out_channels, padding=kernel_size // 2)  # groups等于输出通道数
        
        # 第二个 1×1 卷积，用于通道混合
        self.pointwise2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        
        # 可选：激活函数和归一化层
        self.glu = GLU(dim=1)
        self.swish = Swish()
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, in_channels, length)
        x = self.pointwise1(x)  # 第一个1×1卷积
        x = self.glu(x) 
        x = self.depthwise(x)
        x = self.bn(x)  # 归一化层
        x = self.swish(x) # Swish激活函数
        x = self.pointwise2(x)  # 第二个1×1卷积
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # 转换回 (batch_size, length, out_channels)
        return x
    
class ConformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ff_hidden_dim, max_len):
        super(ConformerEncoderLayer, self).__init__()
        self.attention = RelativePositionAttention(model_dim, num_heads, max_len)
        self.feed_forward1 = FeedForward(model_dim, ff_hidden_dim)
        self.feed_forward2 = FeedForward(model_dim, ff_hidden_dim)
        self.conv = DepthwiseConv(model_dim, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.norm4 = nn.LayerNorm(model_dim)       
    def forward(self, x, mask):
        x = x + 0.5 * self.feed_forward1(x)
        x = self.norm1(x)
        x = x + self.attention(x, mask)
        x = self.norm2(x)
        x = x + self.conv(x)
        x = self.norm3(x)
        x = x + 0.5 * self.feed_forward2(x)
        x = self.norm4(x)
        return x
class InputProjection(nn.Module):
    def __init__(self, input_dim, model_dim):
        super(InputProjection, self).__init__()
        self.input_proj = nn.Conv1d(input_dim, model_dim, kernel_size=1)
    def forward(self, x, mask=None):
        x = self.input_proj(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x
    
class ConformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, ff_hidden_dim, max_len=128):
        super(ConformerEncoder, self).__init__()
        self.model_dim = model_dim
        self.layers = nn.ModuleList(
        [InputProjection(input_dim, model_dim)] + 
        [ConformerEncoderLayer(model_dim, num_heads, ff_hidden_dim, max_len) for _ in range(num_layers)]
)


    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        
        return x

# 6. 最终分类器
class ConformerClassifier(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, ff_hidden_dim, num_classes, max_len=1024):
        super(ConformerClassifier, self).__init__()
        self.encoder = ConformerEncoder(input_dim, model_dim, num_heads, num_layers, ff_hidden_dim, max_len)
        state_dict = torch.load("Checkpoint/pretrain_model_full.pt")
        self.encoder.load_state_dict(state_dict)
        inject_lora(self.encoder, target_modules=["query", "key", "value", "output", "fc1", "fc2"], r=16, alpha=32)
        for name, param in self.encoder.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

        self.gru = nn.GRU(model_dim, model_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(model_dim*2, num_classes)
        
        
    def forward(self, x, mask=None):
        # 获取编码器输出
        x = self.encoder(x, mask)
        out, hn = self.gru(x)
        x = out[:, -1, :]
        x = self.dropout(x)
        # x = torch.mean(x, dim=1)
        x = self.classifier(x)
       
        return x

class ConformerPretrain(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, ff_hidden_dim, max_len=1024):
        super(ConformerPretrain, self).__init__()
        self.encoder = ConformerEncoder(input_dim, model_dim, num_heads, num_layers, ff_hidden_dim, max_len)
        self.construction = ReConstruction(model_dim, input_dim)
        self.gru = nn.GRU(model_dim, model_dim, batch_first=True)
        self.fc = nn.Linear(model_dim, 2)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, mask=None):
        x = self.encoder(x, mask)
        constructed = self.construction(x)
        output, _ = self.gru(x[:, :-1, :])
        output = output[:, -1, :]
        output = self.dropout(output)
        predicted = self.fc(output)
        
        
        # 分类器对 CLS 表示进行分类
        return constructed, predicted
    
class ConformerPredict(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, ff_hidden_dim, max_len=1024):
        super(ConformerPredict, self).__init__()
        self.encoder = ConformerEncoder(input_dim, model_dim, num_heads, num_layers, ff_hidden_dim, max_len)
        self.gru = nn.GRU(model_dim, model_dim, batch_first=True)
        self.fc = nn.Linear(model_dim, 40)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, mask=None):
        x = self.encoder(x, mask)
        output, _ = self.gru(x)
        output = output[:, -1, :]
        output = self.dropout(output)
        predicted = self.fc(output)
        predicted = predicted.view(-1, 20, 2)
        
        
        # 分类器对 CLS 表示进行分类
        return predicted
    
# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
        
#         # Fully connected layers to go from the latent space back to the feature map size
#         self.fc3 = nn.Linear(64, 128)
#         self.fc4 = nn.Linear(128, 256 * 128 * 2)

#         # Transposed convolution layers (deconvolutions) to upsample the feature maps
#         self.t_conv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 1), padding=(1, 0))
#         self.t_conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 1), padding=(1, 0))
#         self.t_conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 1), padding=(1, 0))
#         self.t_conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 1), padding=(1, 0))
#         self.t_conv5 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(3, 1), padding=(1, 0))

#         # Batch normalization layers for the transposed convolutions
#         self.batchnorm6 = nn.BatchNorm2d(num_features=128)
#         self.batchnorm7 = nn.BatchNorm2d(num_features=64)
#         self.batchnorm8 = nn.BatchNorm2d(num_features=32)
#         self.batchnorm9 = nn.BatchNorm2d(num_features=16)

#         # Output activation function, adjust according to your input data range
#         self.output_activation = nn.Tanh()  # or nn.Sigmoid() if your data is in [0, 1]

#     def forward(self, x):
#         # Fully connected layers to expand the latent vector
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
        
#         # Reshape to match the expected input shape for the transposed convolutions
#         x = x.view(-1, 256, 128, 2)

#         # Transposed convolution layers with batch normalization and ReLU activations
#         x = F.relu(self.batchnorm6(self.t_conv1(x)))
#         x = F.relu(self.batchnorm7(self.t_conv2(x)))
#         x = F.relu(self.batchnorm8(self.t_conv3(x)))
#         x = F.relu(self.batchnorm9(self.t_conv4(x)))

#         # Final transposed convolution without batch normalization and with output activation
#         x = self.output_activation(self.t_conv5(x))

#         return x  
        
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(256, 2048),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(2048, 1024),
                                 nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(1024, 256),
)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.view(-1, 128, 2)
        return x
        
class OurEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, ff_hidden_dim, max_len):
        super().__init__()
        self.encoder = ConformerEncoder(input_dim, model_dim, num_heads, num_layers, ff_hidden_dim, max_len)
        
    def forward(self, x):
        x = self.encoder(x, mask=None)
        return x
       
class ConformerDetection(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, ff_hidden_dim, max_len=1024):
        super(ConformerDetection, self).__init__()
        self.encoder = OurEncoder(input_dim, model_dim, num_heads, num_layers, ff_hidden_dim, max_len)             
        self.decoder = ReConstruction(model_dim, input_dim)
    def forward(self, x):
        x_ = self.encoder(x)
        x_rec = self.decoder(x_)
        
        return torch.mean(torch.pow(x - x_rec, 2), dim=[1,2])
    
    def continuetrain(self, x):
        x_ = self.encoder(x)
        return x_
    
    def test(self, x):
        x_ = self.encoder(x)
        x_rec = self.decoder(x_).squeeze(1)
        return torch.mean(torch.abs(x - x_rec), dim=[1,2])

class LightweightTransformerBridge(nn.Module):
    def __init__(self, input_dim=256, llama_embed_dim=1536, hidden_dim=1024, num_layers=1, num_heads=4, dropout=0.1):
        super().__init__()

        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),  # 非线性激活
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llama_embed_dim)
        )
        
    def forward(self, x):
        x = self.adapter(x)            # Normalize for LLaMA compatibility
        return x
class GRUAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=False, dropout=0.1, pooling='last'):
        super(GRUAggregator, self).__init__()
        self.pooling = pooling
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 输出维度：单向 = hidden_dim，双向 = hidden_dim * 2
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x, mask=None):
        """
        x: Tensor of shape [B, T, input_dim]
        mask: Optional, Tensor of shape [B, T], 1 for valid token, 0 for padding (not used here)
        """
        gru_out, h_n = self.gru(x)  # gru_out: [B, T, output_dim], h_n: [num_layers * num_directions, B, hidden_dim]

        if self.pooling == 'last':
            # 取最后一层、最后一个 time step 的输出（只取 forward）
            output = h_n[-1]  # [B, hidden_dim]
        elif self.pooling == 'mean':
            output = torch.mean(gru_out, dim=1)  # [B, output_dim]
        elif self.pooling == 'max':
            output, _ = torch.max(gru_out, dim=1)  # [B, output_dim]
        else:
            raise ValueError(f"Unsupported pooling mode: {self.pooling}")
        
        return output  # [B, output_dim]
class LLMClassifier(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, ff_hidden_dim, num_classes, max_len=1024):  # 
        super().__init__()
        self.encoder = ConformerEncoder(input_dim, model_dim, num_heads, num_layers, ff_hidden_dim, max_len)
        self.encoder.load_state_dict(torch.load("Checkpoint/pretrain_model_full.pt"))
        for layer in self.encoder.layers:  
            for param in layer.parameters():
                param.requires_grad = False
        # self.feature_mapper = LightweightTransformerBridge()
        self.feature_mapper = nn.Sequential(nn.Linear(256, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 1536))
        # Load LLaMA3B (make sure the path or repo is correct and model downloaded)
        model_name = "Qwen_2.5_1.5B"

        self.qwen = AutoModel.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
        ).to(torch.float32)
        print(self.qwen)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 关键，指定插入 LoRA 的模块名（根据 Qwen 的模块结构调整）
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        self.qwen = get_peft_model(self.qwen, lora_config)
        self.qwen.print_trainable_parameters() 
        
        self.output_head = nn.Sequential(
            nn.Linear(1536, num_classes)
        )
        # self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        # x: (batch, 128, 2)
        features = self.encoder(x)  
        llama_input = self.feature_mapper(features)  

        outputs = self.qwen(inputs_embeds=llama_input)
        output = outputs.last_hidden_state.mean(dim=1)
        
      
        # Option 1: use projected representation directly
        logits = self.output_head(output)  
        return logits
if __name__ == "__main__":
    model = LLMClassifier(2, 256, 4, 16, 512, 3, 129)
