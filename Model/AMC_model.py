import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import copy

# class ResNet(nn.Module):
#     def __init__(self, args):
#         super(ResNet, self).__init__()
        
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(3, 1), padding=(1, 0))
#         self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), padding=(1, 0))
#         self.conv3 = nn.Conv2d(in_channels=256, out_channels=80, kernel_size=(3, 1), padding=(1, 0))
#         self.conv4 = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=(3, 2), padding=(1, 0))
        
#         # Fully connected layers
#         self.fc1 = nn.Linear(80 * args.max_seq_length, 128)
#         self.fc2 = nn.Linear(128, args.amc_classes)
        
#         # Dropout layers
#         self.dropout = nn.Dropout(p=args.dropout)
        
#         # Initialization
#         nn.init.xavier_uniform_(self.conv1.weight)
#         nn.init.xavier_uniform_(self.conv2.weight)
#         nn.init.xavier_uniform_(self.conv3.weight)
#         nn.init.xavier_uniform_(self.conv4.weight)
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)
    
#     def forward(self, x):
#         x_in = x.unsqueeze(1)
#         # First convolutional block
#         x = F.relu(self.conv1(x_in))
#         x = self.conv2(x)
#         x1 = x + x_in  # Residual connection
#         x1 = F.relu(x1)
#         # Second convolutional block
#         x = F.relu(self.conv3(x1))
#         x = F.relu(self.conv4(x))
#         x = self.dropout(x)
        
#         # Flatten and fully connected layers
#         x = x.view(x.size(0), -1)  # Flatten
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
        
#         return x
class ModulationRecognitionGRU(nn.Module):
    def __init__(self):
        super(ModulationRecognitionGRU, self).__init__()
        
        # 定义 GRU 层
        self.gru = nn.GRU(input_size=2, 
                          hidden_size=128, 
                          num_layers=2, 
                          batch_first=True, 
                          dropout=0.2)
        
        # 定义分类层
        self.classifier = nn.Linear(128, 2)
        
    def forward(self, x):
        # GRU 输出
        gru_out, _ = self.gru(x)
        
        # 取最后一个时间步的输出
        last_hidden = gru_out[:, -1, :]
        
        # 分类
        output = self.classifier(last_hidden)
        
        return output

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.in_c = in_channel
        self.out_c = out_channel

        self.conv_block = nn.Sequential(
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.Conv2d(self.in_c, self.out_c, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c)
        )

    def forward(self, x):
        """
        x: [batchsize, C, H, W]
        """
        x = self.conv_block(x)

        return x


class MultiScaleModule(nn.Module):
    def __init__(self, out_channel):
        super(MultiScaleModule, self).__init__()
        self.out_c = out_channel

        self.conv_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.Conv2d(1, self.out_c // 3, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3)
        )
        self.conv_5 = nn.Sequential(
            nn.ZeroPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, self.out_c // 3, kernel_size=(2, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3)
        )
        self.conv_7 = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(1, self.out_c // 3, kernel_size=(2, 7)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3)
        )

    def forward(self, x):
        y1 = self.conv_3(x)
        y2 = self.conv_5(x)
        y3 = self.conv_7(x)
        x = torch.cat([y1, y2, y3], dim=1)

        return x


class TinyMLP(nn.Module):
    def __init__(self, N):
        super(TinyMLP, self).__init__()
        self.N = N

        self.mlp = nn.Sequential(
            nn.Linear(self.N, self.N // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.N // 4, self.N),
            # nn.Sigmoid()
            nn.Tanh()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class AdaCorrModule(nn.Module):
    def __init__(self, N):
        super(AdaCorrModule, self).__init__()
        self.Im = TinyMLP(N)
        self.Re = TinyMLP(N)

    def forward(self, x):
        # x:[N, C_out, 1, W]
        x_init = copy.deepcopy(x)
        x = torch.fft.fft(x, dim=-1)
        X_re = torch.real(x)
        X_im = torch.imag(x)
        h_re = self.Re(X_re)
        h_im = self.Im(X_im)
        # x:[N, C_out, 1, W]_complex
        x = torch.mul(h_re, X_re) + 1j * torch.mul(h_im, X_im)
        x = torch.real(torch.fft.ifft(x, dim=-1))
#         x = x / x.norm(p=2, dim=-1, keepdim=True)
#         x_init = x_init / x_init.norm(p=2, dim=-1, keepdim=True)
        x = x + x_init
        
        return x


class FeaFusionModule(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(FeaFusionModule, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[: -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value_heads)
        shape = context.size()
        context = context.contiguous().view(shape[0], -1, shape[-1])
        return context


class AMC_Net(nn.Module):
    def __init__(self,
                 num_classes=11,
                 sig_len=128,
                 extend_channel=36,
                 latent_dim=512,
                 num_heads=2,
                 conv_chan_list=None):
        super(AMC_Net, self).__init__()
        self.sig_len = sig_len
        self.extend_channel = extend_channel
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.conv_chan_list = conv_chan_list

        if self.conv_chan_list is None:
            self.conv_chan_list = [36, 64, 128, 256]
        self.stem_layers_num = len(self.conv_chan_list) - 1

        self.ACM = AdaCorrModule(self.sig_len)
        self.MSM = MultiScaleModule(self.extend_channel)
        self.FFM = FeaFusionModule(self.num_heads, self.sig_len, self.sig_len)

        self.Conv_stem = nn.Sequential()

        for t in range(0, self.stem_layers_num):
            self.Conv_stem.add_module(f'conv_stem_{t}',
                                      Conv_Block(
                                          self.conv_chan_list[t],
                                          self.conv_chan_list[t + 1])
                                      )

        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(self.latent_dim, self.num_classes)
        )

    def forward(self, x):
        # x = x / x.norm(p=2, dim=-1, keepdim=True)
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        x = self.ACM(x)
        x = x / x.norm(p=2, dim=-1, keepdim=True)
        x = self.MSM(x)
        x = self.Conv_stem(x)
        x = self.FFM(x.squeeze(2))
        x = self.GAP(x)
        y = self.classifier(x.squeeze(2))
        return y


def ConvBNReLU(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels, momentum=0.99),
        nn.ReLU(inplace=True)
    )


class MultiScale(nn.Module):
    def __init__(self, in_channels, channels=32, stride=(2, 1)):
        super(MultiScale, self).__init__()

        self.conv = ConvBNReLU(in_channels, channels, kernel_size=(3, 1), stride=stride, padding=(1, 0))

        self.branch1 = ConvBNReLU(channels, channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        self.branch2 = nn.Sequential(
            ConvBNReLU(channels, channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            ConvBNReLU(channels, channels, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(channels, channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            ConvBNReLU(channels, channels, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
        )

        self.branch4 = nn.Sequential(
            ConvBNReLU(channels, channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            ConvBNReLU(channels, channels, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0))
        )

    def forward(self, x):
        x = self.conv(x)
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out



class MSNet(nn.Module):
    def __init__(self, in_channels, classes):
        super(MSNet, self).__init__()
        self.net = nn.Sequential(
            MultiScale(in_channels, stride=(2, 1)),
            MultiScale(128, 32, stride=(2, 1)),
            nn.AdaptiveAvgPool2d((4, 1))
        )
        self.embeddings = nn.Sequential(
            nn.Linear(32*4*4, 128),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(128, classes)

    def forward(self, x):
        x = torch.permute(x, (0,2,1)).unsqueeze(-1)
        out = self.net(x)
        # print('out.shape',out.shape)

        out = out.view(out.shape[0], -1)
        # out = F.normalize(out, p=2, dim=-1)
        # self.classifier.weight.data = F.normalize(self.classifier.weight.data, p=2, dim=-1)
        features = self.embeddings(out)
        # print('features.shape', features.shape)
        logits = self.classifier(features)
        # xx
        # print('logits.shape', logits.shape)
        # features = out.div(torch.norm(out, p=2, dim=1, keepdim=True).expand_as(features))
        return logits


def conv_bn_relu_maxpool(in_channels, out_channels, kernel_size,  stride, padding, pool=(2, 1)):
    return [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool, stride=(2, 1))]


class CONV(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  stride, padding, bn=True):
        super(CONV, self).__init__()
        # self.conv = nn.Conv1d
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if bn else False
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.relu(x)
        return x


def make_layer(in_channels):
    conv1 = conv_bn_relu_maxpool(in_channels, 64, kernel_size=(3, 2), stride=(1, 2), padding=(1, 1), pool=(2, 1))
    conv2 = conv_bn_relu_maxpool(64, 64, kernel_size=(1, 2), stride=(1, 2), padding=(1, 1))
    conv3 = conv_bn_relu_maxpool(64, 64, kernel_size=(1, 2), stride=(1, 2), padding=(1, 1))
    conv4 = conv_bn_relu_maxpool(64, 64, kernel_size=(1, 2), stride=(1, 2), padding=(1, 1))
    conv5 = conv_bn_relu_maxpool(64, 64, kernel_size=(1, 2), stride=(1, 2), padding=(1, 1))
    conv6 = conv_bn_relu_maxpool(64, 64, kernel_size=(1, 2), stride=(1, 2), padding=(1, 1))
    conv7 = conv_bn_relu_maxpool(64, 64, kernel_size=(1, 2), stride=(1, 2), padding=(1, 1))
    # return conv1 + conv2 + conv3 + conv4
    return conv1 + conv2 + conv3 + conv4 + conv5 + conv6 + conv7


def fc_dropout(in_channels, out_channels, alpha=0.5):
    return nn.ModuleList([
        nn.Linear(in_channels, out_channels, bias=True),
        nn.SELU(inplace=True),
        nn.AlphaDropout(alpha)
    ])


class VGG(nn.Module):
    def __init__(self, in_channels, classes):
        super(VGG, self).__init__()
        self.base = nn.ModuleList(make_layer(in_channels))
        self.embeddings = nn.Sequential(
            nn.Linear(128, 128),
            nn.SELU(inplace=True),
            nn.AlphaDropout(0.5),
            nn.Linear(128, 128),
            nn.SELU(inplace=True),
            nn.AlphaDropout(0.5)
        )
        self.classifier = nn.Linear(128, classes)

        # self.apply(_weights_init)

    def forward(self, x):
        x = torch.permute(x, (0,2,1)).unsqueeze(-1)
        for l in self.base:
            x = l(x)
        x = x.view(x.shape[0], -1)
        x = self.embeddings(x)
        return self.classifier(x)

class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, downsample=None):
        super(ResidualUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, affine=True, momentum=0.99)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=True, momentum=0.99)
        self.downsample = downsample
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)


        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)
        return out


class Residual_Stack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0,1), pool=(1, 1)):
        super(Residual_Stack, self).__init__()
        self.conv = nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, bias=True)
        self.residual_unit1 = ResidualUnit(32, 32, kernel_size, stride, padding)
        self.residual_unit2 = ResidualUnit(32, 32, kernel_size, stride, padding)
        self.maxpool = nn.MaxPool2d(kernel_size=pool, stride=(1, 1))

    def forward(self, x):
        x = self.conv(x)
        x = self.residual_unit1(x)
        x = self.residual_unit2(x)

        return x


class fc(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(fc, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.selu = nn.SELU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.selu(x)

        return x


class ResNet(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(ResNet, self).__init__()

        self.net = nn.Sequential(
            Residual_Stack(in_channels, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0,1), pool=(2, 1)),
            Residual_Stack(32, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0,1), pool=(2, 1)),
            Residual_Stack(32, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0,1), pool=(2, 1)),
            # Residual_Stack(32, 32, kernel_size=(3, 2), stride=(1, 2), padding=(1, 1), pool=(2, 1)),
            # Residual_Stack(32, 32, kernel_size=(3, 2), stride=(1, 2), padding=(1, 1), pool=(2, 1)),
            # Residual_Stack(32, 32, kernel_size=(3, 2), stride=(1, 2), padding=(1, 1), pool=(2, 1))
        )
        # self.stack1 = Residual_Stack(in_channels, 32, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), pool=(2, 1))
        # self.stack2 = Residual_Stack(32, 32, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), pool=(2, 1))
        # self.stack3 = Residual_Stack(32, 32, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), pool=(2, 1))
        # self.stack4 = Residual_Stack(32, 32, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), pool=(2, 1))
        # self.stack5 = Residual_Stack(32, 32, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), pool=(2, 1))
        # self.stack6 = Residual_Stack(32, 32, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), pool=(2, 1))
        self.embeddings = nn.Sequential(
            nn.Linear(4096, 2048),
            # nn.Linear(4096, 128),
            nn.SELU(inplace=True),
            nn.AlphaDropout(0.5),

            nn.Linear(2048, 512),
            # nn.Linear(4096, 128),
            nn.SELU(inplace=True),
            nn.AlphaDropout(0.5),

            nn.Linear(512, 128),
            nn.SELU(inplace=True),
            nn.AlphaDropout(0.5)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.permute(x, (0,2,1)).unsqueeze(-1)
        out = self.net(x)
        out = out.view(out.shape[0], -1)
        out = self.embeddings(out)
        return self.classifier(out)


class CNN2(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN2, self).__init__()
        
        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=(5, 1), padding='same')
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(5, 1), padding='same')
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(5, 1), padding='same')
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), padding='same')
        
        # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        
        # 定义Dropout层
        self.dropout = nn.Dropout(0.5)
        
        # 定义全连接层
        self.fc1 = nn.Linear(1024, 128) # 需要计算正确的输入尺寸
        self.fc2 = nn.Linear(128, num_classes)
        
        # 初始化权重 - 类似于Glorot uniform
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = x.unsqueeze(1)
        # 卷积层+ReLU激活函数+池化层
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # 展平操作
        x = torch.flatten(x, start_dim=1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class DAE(nn.Module):
    def __init__(self, input_shape=(128, 2), classes=11, dropout_rate=0.0):
        super(DAE, self).__init__()
        
        # LSTM单元
        self.lstm1 = nn.LSTM(input_size=input_shape[1], hidden_size=32, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)
        
        # 分类器
        self.fc1 = nn.Linear(32, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, classes)
        
        # 解码器
        self.time_distributed = nn.Linear(32, input_shape[1])
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
       

    def forward(self, x):
        # LSTM单元
        x, (hn1, cn1) = self.lstm1(x)
        x = self.dropout(x)
        x, (hn2, cn2) = self.lstm2(x)
        
        # 分类器
        xc = self.fc1(hn2[-1])  # 使用最后一个LSTM层的最终状态
        xc = self.bn1(xc)
        xc = F.relu(xc)
        xc = self.dropout(xc)
        
        xc = self.fc2(xc)
        xc = self.bn2(xc)
        xc = F.relu(xc)
        xc = self.dropout(xc)
        
        xc = self.fc3(xc)
        xc = F.softmax(xc, dim=-1)
        
        # 解码器
        xd = self.time_distributed(x)
        
        return xc, xd

class CGDNN(nn.Module):
    def __init__(self, input_channels=1, classes=11, dropout_rate=0.5):
        super(CGDNN, self).__init__()
        
        # 定义卷积层和池化层
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=50, kernel_size=(6, 1))
        # self.pool1 = nn.MaxPool2d(kernel_size=(2, 2),padding=(1, 1))
        self.gaussian_dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(6, 1))
        # self.pool2 = nn.MaxPool2d(kernel_size=(2, 2),padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(6, 1))
        # self.pool3 = nn.MaxPool2d(kernel_size=(2, 2),padding=(1, 1))
        # 定义GRU层
        self.gru = nn.GRU(input_size=472, hidden_size=50, batch_first=True)

        # 定义全连接层
        self.fc1 = nn.Linear(50, 256)
        self.fc2 = nn.Linear(256, classes)

        

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加一个维度
        # 卷积层+ReLU激活函数+池化层
        x1 = F.relu(self.conv1(x))
        # x1 = self.pool1(x1)
        x1 = self.gaussian_dropout(x1)

        x2 = F.relu(self.conv2(x1))
        # x2 = self.pool2(x2)
        x2 = self.gaussian_dropout(x2)

        x3 = F.relu(self.conv3(x2))
        # x3 = self.pool3(x3)
        x3 = self.gaussian_dropout(x3)

        # 连接x1和x3
        x_concat = torch.cat([x1, x3], dim=-2)
        
        # Reshape操作
        x_reshaped = x_concat.view(x_concat.size(0), 50, -1)
        
        # GRU层
        x_gru, _ = self.gru(x_reshaped) 
        
        # 全连接层
        x_fc = F.relu(self.fc1(x_gru[:, -1, :])) 
        x_fc = self.gaussian_dropout(x_fc)
        x_out = self.fc2(x_fc)
        
        return x_out


class GaussianDropout(nn.Module):
    def __init__(self, p=0.5):
        super(GaussianDropout, self).__init__()
        if p <= 0 or p >= 1:
            raise Exception("p value should accomplish 0 < p < 1")
        self.p = p
        
    def forward(self, x):
        if self.training:
            stddev = (self.p / (1.0 - self.p))**0.5
            epsilon = torch.randn_like(x) * stddev
            return x * epsilon
        else:
            return x


class MCNet(nn.Module):
    def __init__(self, num_classes):
        super(MCNet, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3))
        self.pool1_1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        # Preblock
        self.conv2_1 = nn.Conv2d(64, 32, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.pool2_1 = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv2_2 = nn.Conv2d(64, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        # Skip connection
        self.conv111 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 2), padding=(0, 0))
        self.pool2_2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        # Mblockp1
        self.pool3_1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv3_1 = nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv3_2 = nn.Conv2d(32, 48, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.pool3_2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv3_3 = nn.Conv2d(32, 48, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv3_4 = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 2), padding=(0, 0))

        # Mblock2
        self.conv4_1 = nn.Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv4_2 = nn.Conv2d(32, 48, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.conv4_3 = nn.Conv2d(32, 48, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.conv4_4 = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # Mblockp3
        self.conv5_1 = nn.Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv5_2 = nn.Conv2d(32, 48, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.pool5_2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv5_3 = nn.Conv2d(32, 48, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv5_4 = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 2), padding=(0, 0))

        # Mblockp4
        self.conv6_1 = nn.Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv6_2 = nn.Conv2d(32, 48, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.conv6_3 = nn.Conv2d(32, 48, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.conv6_4 = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # Mblockp5
        self.conv7_1 = nn.Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv7_2 = nn.Conv2d(32, 48, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.pool7_2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv7_3 = nn.Conv2d(32, 48, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv7_4 = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 2), padding=(0, 0))

        # Mblockp6
        self.conv8_1 = nn.Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv8_2 = nn.Conv2d(32, 96, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.conv8_3 = nn.Conv2d(32, 96, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.conv8_4 = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # Output layers
        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(24576, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv2D
        x = F.relu(self.conv1_1(x))
        x = self.pool1_1(x)

        x2 = F.relu(self.conv2_1(x))
        x2 = self.pool2_1(x2)
        x22 = F.relu(self.conv2_2(x))
        x222 = torch.cat([x2, x22], dim=1)

        xx1 = F.relu(self.conv111(x222))
        xx1 = self.pool2_2(xx1)

        x3 = self.pool3_1(x222)
        x3 = F.relu(self.conv3_1(x3))
        x31 = F.relu(self.conv3_2(x3))
        x31 = self.pool3_2(x31)
        x32 = F.relu(self.conv3_3(x3))
        x33 = F.relu(self.conv3_4(x3))
        x31 = torch.cat([x31, x32], dim=1)
        x333 = torch.cat([x33, x31], dim=1)

        add1 = x333 + xx1

        x4 = F.relu(self.conv4_1(add1))
        x41 = F.relu(self.conv4_2(x4))
        x42 = F.relu(self.conv4_3(x4))
        x43 = F.relu(self.conv4_4(x4))
        x41 = torch.cat([x41, x42], dim=1)
        x444 = torch.cat([x43, x41], dim=1)

        add2 = x444 + add1

        x5 = F.relu(self.conv5_1(add2))
        x51 = F.relu(self.conv5_2(x5))
        x51 = self.pool5_2(x51)
        x52 = F.relu(self.conv5_3(x5))
        x53 = F.relu(self.conv5_4(x5))
        x51 = torch.cat([x51, x52], dim=1)
        x555 = torch.cat([x53, x51], dim=1)

        ad3 = self.pool5_2(add2)
        add3 = x555 + ad3

        x6 = F.relu(self.conv6_1(add3))
        x61 = F.relu(self.conv6_2(x6))
        x62 = F.relu(self.conv6_3(x6))
        x63 = F.relu(self.conv6_4(x6))
        x61 = torch.cat([x61, x62], dim=1)
        x666 = torch.cat([x63, x61], dim=1)

        add4 = x666 + add3

        x7 = F.relu(self.conv7_1(add4))
        x71 = F.relu(self.conv7_2(x7))
        x71 = self.pool7_2(x71)
        x72 = F.relu(self.conv7_3(x7))
        x73 = F.relu(self.conv7_4(x7))
        x71 = torch.cat([x71, x72], dim=1)
        x777 = torch.cat([x73, x71], dim=1)

        ad5 = self.pool7_2(add4)
        add5 = x777 + ad5

        x8 = F.relu(self.conv8_1(add5))
        x81 = F.relu(self.conv8_2(x8))
        x82 = F.relu(self.conv8_3(x8))
        x83 = F.relu(self.conv8_4(x8))
        x81 = torch.cat([x81, x82], dim=1)
        x888 = torch.cat([x83, x81], dim=1)

        x_con = torch.cat([x888, add5], dim=1)
        xout = self.avg_pool(x_con)
        xout = self.dropout(xout)
        xout = torch.flatten(xout, 1)
        xout = self.fc(xout)

        return xout



class Transformer(nn.Module):
    def __init__(self, input_dim=(128, 1), num_classes=11, n_transformer_layers=2, d_model=256, n_heads=4, dim_feedforward=512, dropout=0.1):
        super(Transformer, self).__init__()

        self.conv = nn.Conv2d(2, d_model, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn = nn.BatchNorm2d(d_model)
        self.flatten_conv = nn.Flatten(start_dim=2)
        
        # Positional Embedding
        self.positional_embedding = nn.Parameter(torch.zeros(1, input_dim[0], d_model))

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        # Fully Connected Classification Layers
        self.fc = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes),
        )
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)

    def forward(self, x):
        # Input shape: B x 128 x 2
        x = x.permute(0,2,1).unsqueeze(-1) 
        x = self.conv(x) 
        x = self.bn(x)  
        x = F.tanh(x)
        x = self.flatten_conv(x)  # Flatten along the time dimension -> B x d_model x 128
        x = x.permute(0, 2, 1)  # Rearrange to B x 128 x d_model for Transformer

        # Add positional embedding
        x = x + self.positional_embedding # B x 128 x d_model

        # Transformer Encoding
        x = self.transformer_encoder(x)  # B x 128 x d_model
        x, _ = self.gru(x)
        x = x[:, -1, :]
        # Classification Head
        logits = self.fc(x)  # B x num_classes

        return logits

class GRU2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, bidirectional=False):
        """
        Args:
            input_size (int): 输入特征维度 F
            hidden_size (int): GRU 隐藏层单元数量
            num_classes (int): 分类类别数量
            num_layers (int): GRU 层数，默认为 2
            bidirectional (bool): 是否使用双向 GRU，默认为 False
        """
        super(GRU2, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        # 两层 GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # 全连接层，用于分类
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)
        
    
    def forward(self, x):
        """
        Args:
            x (Tensor): 输入张量，形状为 (B, T, F)
        
        Returns:
            Tensor: 分类输出，形状为 (B, num_classes)
        """
        # 初始化隐藏状态 (num_layers * num_directions, B, hidden_size)
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
        
        # GRU 前向传播
        out, _ = self.gru(x, h0)  # out: (B, T, hidden_size * num_directions)
        
        # 取最后时间步的输出作为分类输入
        last_output = out[:, -1, :]  # (B, hidden_size * num_directions)
        
        # 全连接层
        logits = self.fc(last_output)  # (B, num_classes)
        
       
        
        return logits