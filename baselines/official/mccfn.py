import torch
import torch.nn as nn

class ComplexConv1d1(nn.Module):
    """
    Complex-valued Convolution (CC)
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, complex_axis=1):

        super(ComplexConv1d1, self).__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if groups == in_channels:
            self.groups = groups // 2
        else:
            self.groups = 1
        self.dilation = dilation
        self.bias = bias
        self.complex_axis = complex_axis
        self.real_conv = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride,
                                   self.padding, self.dilation, groups=self.groups, bias=self.bias)
        self.imag_conv = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride,
                                   self.padding, self.dilation, groups=self.groups, bias=self.bias)

        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)

    def forward(self, inputs):
        inputs = inputs.squeeze(2)
        real, imag = torch.chunk(inputs, 2, self.complex_axis)

        real2real = self.real_conv(real)
        imag2imag = self.imag_conv(imag)

        real2imag = self.imag_conv(real)
        imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real

        return torch.cat([real, imag], self.complex_axis).unsqueeze(2)


class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.in_c = in_channel
        self.out_c = out_channel

        self.conv_block = nn.Sequential(
            nn.ZeroPad2d((1, 1, 0, 0)),
            ComplexConv1d1(self.in_c, self.out_c, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c)
        )

    def forward(self, x):
        """
        x: [batchsize, C, H, W]
        """
        x = self.conv_block(x)

        return x


class ComplexConv1d(nn.Module):
    """
    Complex-valued Convolution (CC)
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, complex_axis=1):

        super(ComplexConv1d, self).__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if groups == in_channels:
            self.groups = groups // 2
        else:
            self.groups = 1
        self.dilation = dilation
        self.bias = bias
        self.complex_axis = complex_axis
        self.real_conv = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride,
                                   self.padding, self.dilation, groups=self.groups, bias=self.bias)
        self.imag_conv = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride,
                                   self.padding, self.dilation, groups=self.groups, bias=self.bias)

        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)

    def forward(self, inputs):
        inputs = inputs.squeeze(1)
        real, imag = torch.chunk(inputs, 2, self.complex_axis)

        real2real = self.real_conv(real)
        imag2imag = self.imag_conv(imag)

        real2imag = self.imag_conv(real)
        imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real

        return torch.cat([real, imag], self.complex_axis).unsqueeze(2)

class CReLU(nn.Module):
    """
    Complex-valued ReLU (CReLU)
    """

    def __init__(self, complex_axis=1, inplace=False):
        super(CReLU, self).__init__()
        self.r_relu = nn.ReLU(inplace=inplace)
        self.i_relu = nn.ReLU(inplace=inplace)
        self.complex_axis = complex_axis

    def forward(self, inputs):

        inputs = inputs.squeeze(2)

        real, imag = torch.chunk(inputs, 2, self.complex_axis)

        real = self.r_relu(real)

        imag = self.i_relu(imag)

        output = torch.cat([real, imag], self.complex_axis)
        output = output.unsqueeze(2)
        return output


class MultiScaleModule(nn.Module):
    def __init__(self, out_channel):
        super(MultiScaleModule, self).__init__()
        self.out_c = out_channel

        self.conv_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 0, 0)),
            ComplexConv1d(2, self.out_c // 3, kernel_size=3),
            nn.LeakyReLU (inplace=True),
            nn.BatchNorm2d(self.out_c // 3)

        )
        self.conv_5 = nn.Sequential(
            nn.ZeroPad2d((2, 2, 0, 0)),
            ComplexConv1d(2, self.out_c // 3, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3)
        )
        self.conv_7 = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            ComplexConv1d(2, self.out_c // 3, kernel_size=7),
            # CReLU(inplace=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3)
        )

    def forward(self, x):

        y1 = self.conv_3(x)
        y2 = self.conv_5(x)
        y3 = self.conv_7(x)
        x = torch.cat([y1, y2, y3], dim=1)



        return x


class FeaFusionModule(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(FeaFusionModule, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"the hidden size {hidden_size} is not a multiple of the number of attention heads {num_attention_heads}"
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = hidden_size

        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads, dropout=0.5)
        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        batch_size, seq_len, _ = x.size()

        # Linear projections
        key = self.key_layer(x)  # [batch_size, seq_len, hidden_size]
        query = self.query_layer(x)  # [batch_size, seq_len, hidden_size]
        value = self.value_layer(x)  # [batch_size, seq_len, hidden_size]

        # MultiheadAttention expects inputs of shape (seq_len, batch_size, hidden_size)
        key = key.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        query = query.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        value = value.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]

        # Apply multi-head attention
        attn_output, _ = self.attention(query, key, value)

        # Dropout layer
        attn_output = self.dropout(attn_output)

        # attn_output shape: [seq_len, batch_size, hidden_size], we need to permute it back to [batch_size, seq_len, hidden_size]
        attn_output = attn_output.permute(1, 0, 2)

        return attn_output


class MCCFN(nn.Module):
    def __init__(self,
                 num_classes=11,
                 sig_len=128,
                 extend_channel=36,
                 latent_dim=512,
                 num_heads=2,
                 conv_chan_list=None):
        super(MCCFN, self).__init__()
        self.sig_len = sig_len
        self.extend_channel = extend_channel
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.conv_chan_list = conv_chan_list
        if self.conv_chan_list is None:
            self.conv_chan_list = [36, 64, 128, 256]
        self.stem_layers_num = len(self.conv_chan_list) - 1
        self.MSM = MultiScaleModule(self.extend_channel)
        self.FFM1 = FeaFusionModule(self.num_heads, self.sig_len, self.sig_len)
        self.Conv_stem = nn.Sequential()
        for t in range(0, self.stem_layers_num):
            self.Conv_stem.add_module(f'conv_stem_{t}',
                                      Conv_Block(
                                          self.conv_chan_list[t],
                                          self.conv_chan_list[t + 1])
                                      )
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, self.latent_dim),
            nn.Dropout(0.),
            nn.PReLU(),
            nn.Linear(self.latent_dim, self.num_classes)
        )

    def forward(self, x):
        regu_sum = []
        x = x.unsqueeze(1)
        x = self.MSM(x)
        x = self.Conv_stem(x)
        x = self.FFM1(x.squeeze(2))
        x = self.GAP(x)
        y = self.classifier(x.squeeze(2))
        return y, regu_sum


if __name__ == '__main__':
    model = MCCFN(11, 128, 36, 512, 2, [36,48,64,128, 256])
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    x = torch.randn((1, 2, 128)).to(device)
    y, regu_sum = model(x)
    print(y)

    print(">>> total params: {:.2f}M".format(
        sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    # 计算 FLOPs
    flops = FlopCountAnalysis(model, x)

    # 将 FLOPs 转换为 Million FLOPs（M）
    flops_in_million = flops.total() / 1_000_000  # 计算百万 FLOPs

    print(f"FLOPs: {flops_in_million:.2f}M")  # 输出以M为单位的FLOPs


    MACs, Params = thop.profile(model, inputs=(x,), verbose=False)
    FLOPs = MACs * 2
    MACs, FLOPs,  Params = thop.clever_format([MACs, FLOPs, Params], "%.3f")

    print(f"MACs: {MACs}")
    print(f"FLOPs: {FLOPs}")
    print(f"Params: {Params}")