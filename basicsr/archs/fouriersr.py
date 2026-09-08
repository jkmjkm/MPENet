import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class ResBlock_fre(nn.Module):
    def __init__(
            self, n_feats, kernel_size,conv=default_conv,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock_fre, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

        self.fre = Frequency_Convolution(n_feats)

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res = self.fre(res)
        res += x

        return res


class Frequency_Convolution(nn.Module):
    """
    channels: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    input shape [B N C]
    """

    def __init__(self, channels, num_blocks=8, sparsity_threshold=0.01):
        super().__init__()
        assert channels % num_blocks == 0, f"channels {channels} should be divisble by num_blocks {num_blocks}"
        # 通道分组
        self.channels = channels
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = channels // self.num_blocks
        self.scale = 0.02
        # 可学习滤波器
        # [num_blocks,block_size,block_size,2] Token mix matrix 矩阵运算通道混合
        self.w = nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size, self.block_size, 2))
        # [2,num_blocks.block_size,1,1] 单频率位置的实数和复数特征调制 缩放 逐元素相乘
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, 1, 1))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, 1, 1))
        # 实部和虚部的bias
        self.b = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape
        # 二维实数傅里叶变换 rfft2 针对实数输入，利用共轭对称性，只保留一半频谱。
        # x[[B, C, H, Wf]] Wf = W // 2 + 1
        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        # 通道分组
        x = x.reshape(B, self.num_blocks, self.block_size, x.shape[2], x.shape[3])
        # 构造复数矩阵 每个通道组都有一个复数矩阵
        weight = torch.view_as_complex(self.w.contiguous())
        # 对每一个频率特征都进行通道分组的组内线性组合  也就是组内的通道混合
        x = torch.einsum('bkihw,kio->bkohw', x, weight)
        # 更新实部虚部  ReLU（原始实部和原始虚部经过缩放过后的加减结果+偏置）
        o1_real = F.relu(
            torch.mul(x.real, self.w1[0].unsqueeze(dim=0)) - \
            torch.mul(x.imag, self.w1[1].unsqueeze(dim=0)) + \
            self.b[0, :, :, None, None]
        )  # [16, 8, 8, 48, 25]  x.imag=[16, 8, 8, 48, 25]

        o1_imag = F.relu(
            torch.mul(x.imag, self.w2[0].unsqueeze(dim=0)) + \
            torch.mul(x.real, self.w2[1].unsqueeze(dim=0)) + \
            self.b[1, :, :, None, None]
        )  # [16, 8, 8, 48, 25] x.real=[16, 8, 8, 48, 25]
        # 将实部和虚部拼接
        x = torch.stack([o1_real, o1_imag], dim=-1)  # [16, 8, 8, 48, 25, 2]
        # 稀疏软筛选
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        # 恢复复数矩阵
        x = torch.view_as_complex(x)  # [16, 8, 8, 48, 25]
        x = x.reshape(B, C, x.shape[3], x.shape[4])
        # 逆变换
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")
        # 转会原始类型
        x = x.type(dtype)

        return x + bias


'''
class AFNO2D_channelfirst(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    input shape [B N C]
    """
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1,
                 hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

        self.fliter = nn.Parameter(self.scale * torch.randn(1, self.hidden_size, 1, 1))

    def forward(self, x, spatial_size=None):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape

        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        origin_ffted = x
        x = x.reshape(B, self.num_blocks, self.block_size, x.shape[2], x.shape[3])

        o1_real = F.relu(
            torch.einsum('bkihw,kio->bkohw', x.real, self.w1[0]) - \
            torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[1]) + \
            self.b1[0, :, :, None, None]
        ) # [16, 8, 8, 48, 25]  x.imag=[16, 8, 8, 48, 25]

        o1_imag = F.relu(
            torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[0]) + \
            torch.einsum('bkihw,kio->bkohw', x.real, self.w1[1]) + \
            self.b1[1, :, :, None, None]
        ) # [16, 8, 8, 48, 25]


        x = torch.stack([o1_real, o1_imag], dim=-1) # [16, 8, 8, 48, 25, 2]
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x) # [16, 8, 8, 48, 25]

        x = x.reshape(B, C, x.shape[3], x.shape[4])

        x = x * self.fliter + origin_ffted 

        # x = x + origin_ffted 

        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")
        x = x.type(dtype)

        return x + bias
'''

'''
class AFNO2D_channelfirst(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    input shape [B N C]
    """
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1,
                 hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, 1, 1))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, 1, 1))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

        self.fliter = nn.Parameter(self.scale * torch.randn(1, hidden_size, 1, 1))

    def forward(self, x, spatial_size=None):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape

        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        origin_ffted = x
        x = x.reshape(B, self.num_blocks, self.block_size, x.shape[2], x.shape[3])

        o1_real = F.relu(
            torch.mul(x.real, self.w1[0].unsqueeze(dim=0)) - \
            torch.mul(x.imag, self.w1[1].unsqueeze(dim=0)) + \
            self.b1[0, :, :, None, None]
        ) # [16, 8, 8, 48, 25]  x.imag=[16, 8, 8, 48, 25]
        # print(x.real.size())
        # print(self.w1[0].size())
        # print(self.b1[0, :, :, None, None].size())
        # print(o1_real.size())

        o1_imag = F.relu(
            torch.mul(x.imag, self.w2[0].unsqueeze(dim=0)) + \
            torch.mul(x.real, self.w2[1].unsqueeze(dim=0)) + \
            self.b1[1, :, :, None, None]
        ) # [16, 8, 8, 48, 25]


        x = torch.stack([o1_real, o1_imag], dim=-1) # [16, 8, 8, 48, 25, 2]
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x) # [16, 8, 8, 48, 25]
        x = x.reshape(B, C, x.shape[3], x.shape[4])


        x = x * self.fliter + origin_ffted
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")
        x = x.type(dtype)

        return x + bias
'''

