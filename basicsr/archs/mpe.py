import math
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from sympy.abc import alpha

from basicsr.data.feature_visual_hook import FeatureHook
# Branch 1
# Frequency Statistics Guided Modulation (FSGM)
class FSGMBranch(nn.Module):

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()

        hidden = max(channels // reduction, 8)

        # strip conv
        self.pre = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(7, 1),
                padding=(3, 0),
                groups=channels,
                bias=True
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, 7),
                padding=(0, 3),
                groups=channels,
                bias=True
            ),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # frequency statistics gate
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):

        feat = self.pre(x)

        # -------------------------------------------------
        # FFT
        # -------------------------------------------------
        fft_feat = torch.fft.rfft2(feat, norm='ortho')
        # amplitude spectrum
        amp = torch.abs(fft_feat)
        # -------------------------------------------------
        # frequency statistics
        # -------------------------------------------------
        # mean spectral response
        freq_mean = amp.mean(dim=(-2, -1), keepdim=True)
        # variance spectral response
        freq_var = amp.var(dim=(-2, -1), keepdim=True)
        # concat statistics
        freq_stat = torch.cat([freq_mean, freq_var], dim=1)
        # -------------------------------------------------
        # generate adaptive gate
        # -------------------------------------------------
        gate = self.gate(freq_stat)
        # -------------------------------------------------
        # modulation
        # -------------------------------------------------
        out = feat * gate
        return out



# Branch 2
# Energy-Structure Aware Enhancement (ESAE)
class EACMBranch(nn.Module):

    def __init__(
        self,
        channels: int,
        reduction: int = 8,
        lam: float = 1e-4
    ):
        super().__init__()

        self.lam = lam

        hidden = max(channels // reduction, 8)

        # large-kernel preprocessing
        self.pre = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=(11, 1),
                padding=(5, 0),
                groups=channels,
                bias=True
            ),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, 11),
                padding=(0, 5),
                groups=channels,
                bias=True
            ),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=True
            ),

            nn.LeakyReLU(0.1, inplace=True),
        )

        # statistics-guided gate generation
        self.gate = nn.Sequential(

            nn.Conv2d(
                channels * 2,
                hidden,
                kernel_size=1,
                bias=True
            ),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(
                hidden,
                channels,
                kernel_size=1,
                bias=True
            ),
            nn.Sigmoid()
        )
    # Energy Function
    def energy_function(self, x):

        # global mean
        mu = x.mean(
            dim=(2, 3),
            keepdim=True
        )

        # global variance
        var = ((x - mu) ** 2).mean(
            dim=(2, 3),
            keepdim=True
        )

        # energy response
        energy = ((x - mu) ** 2) / (
            4.0 * (var + self.lam)
        ) + 0.5

        return energy

    def forward(self, x):

        # preprocessing
        feat = self.pre(x)

        # energy prior
        energy = self.energy_function(feat)


        # mean energy response
        energy_mean = energy.mean(
            dim=(2, 3),
            keepdim=True
        )
        # variance energy response
        energy_var = energy.var(
            dim=(2, 3),
            keepdim=True
        )
        # concat statistics
        descriptor = torch.cat(
            [energy_mean, energy_var],
            dim=1
        )

        # adaptive gate
        gate = self.gate(descriptor)

        # dynamic modulation
        out = feat * gate
        return out

# Branch 3
# Local Texture Refinement (LTR)
class TCSGConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.weight = nn.Parameter(
            torch.randn(channels, 1, 3, 3)
        )

        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):

        w = self.weight

        # kernel mean
        w_centered = w - w.mean(dim=[2,3], keepdim=True)

        return F.conv2d(
            x,
            w_centered,
            self.bias,
            padding=1,
            groups=self.channels
        )

class CSGConv(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True
        )

    def forward(self, x):

        # local mean
        local_mean = F.avg_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # center-surrounding difference
        diff = x - local_mean
        out = self.dwconv(diff)
        return out



class AE(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.body = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=True
            ),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=True
            ),

            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):

        return self.body(x)


# Horizontal Gradient Convolution
class HorizontalGradientConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 1×3 水平差分核
        kernel = torch.tensor([1., 0., -1.]).view(1, 1, 1, 3)
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.diff = nn.Conv2d(
            channels, channels,
            kernel_size=(1, 3),
            padding=(0, 1),
            groups=channels,
            bias=False
        )
        self.diff.weight = nn.Parameter(kernel, requires_grad=False)
        # 3×1 融合
        self.fusion = nn.Conv2d(
            channels, channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=channels,
            bias=True
        )

    def forward(self, x):
        x = self.diff(x)
        x = self.fusion(x)
        return x


# Vertical Gradient Convolution
class VerticalGradientConv(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # 3×1 垂直差分核（替代3×3）
        kernel = torch.tensor([1., 0., -1.]).view(1, 1, 3, 1)
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.diff = nn.Conv2d(
            channels, channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=channels,
            bias=False
        )
        self.diff.weight = nn.Parameter(kernel, requires_grad=False)
        # 1×3 融合
        self.fusion = nn.Conv2d(
            channels, channels,
            kernel_size=(1, 3),
            padding=(0, 1),
            groups=channels,
            bias=True
        )

    def forward(self, x):
        x = self.diff(x)
        x = self.fusion(x)
        return x

class SobelXConv(nn.Module):

    def __init__(self, channels):
        super().__init__()

        kernel = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ]).view(1, 1, 3, 3)


        kernel = kernel.repeat(
            channels, 1, 1, 1
        )


        self.sobel = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False
        )


        self.sobel.weight = nn.Parameter(
            kernel,
            requires_grad=False
        )


    def forward(self, x):

        x = self.sobel(x)

        return x

class SobelYConv(nn.Module):

    def __init__(self, channels):
        super().__init__()

        kernel = torch.tensor([
            [-1., -2., -1.],
            [ 0.,  0.,  0.],
            [ 1.,  2.,  1.]
        ]).view(1, 1, 3, 3)


        kernel = kernel.repeat(
            channels, 1, 1, 1
        )


        self.sobel = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False
        )


        self.sobel.weight = nn.Parameter(
            kernel,
            requires_grad=False
        )


    def forward(self, x):

        x = self.sobel(x)

        return x

class LaplacianConv(nn.Module):

    def __init__(self, channels):
        super().__init__()

        kernel = torch.tensor([
            [-1., -1., -1.],
            [-1.,  8., -1.],
            [-1., -1., -1.]
        ]).view(1, 1, 3, 3)


        kernel = kernel.repeat(
            channels, 1, 1, 1
        )


        self.laplacian = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False
        )


        self.laplacian.weight = nn.Parameter(
            kernel,
            requires_grad=False
        )


    def forward(self, x):

        x = self.laplacian(x)

        return x

class MltiBranchTextureBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        # shared preprocessing
        self.pre = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=True
            ),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # three branches
        self.h_branch = HorizontalGradientConv(channels)

        self.v_branch = VerticalGradientConv(channels)

        self.csg_branch = CSGConv(channels)

        self.dw_branch = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(channels,channels,kernel_size=3,padding=1,groups=channels,bias=True)
        )

        # fusion
        self.fuse = nn.Conv2d(
            channels * 4,
            channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):

        x = self.pre(x)
        h = self.h_branch(x)
        v = self.v_branch(x)
        d = self.dw_branch(x)
        csg = self.csg_branch(x)
        out = torch.cat([h,v,d,csg], dim=1)
        out = self.fuse(out)
        return out

class MultiPriorEnhancement(nn.Module):

    def __init__(self, channels: int = 64):

        super().__init__()

        # shallow preprocessing
        self.pre = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=5,
                padding=2,
                groups=channels,
                bias=True
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=True
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # branch1
        self.fsgm = FSGMBranch(channels)

        # branch2
        self.esae = EACMBranch(channels)

        # branch3
        # self.ltr = TripleBranchTextureBlock(channels)
        self.ltr = MltiBranchTextureBlock(channels)
        # fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=True
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    def forward(self, x):
        identity=x
        feat = self.pre(x)
        out1 = self.fsgm(feat)
        out2 = self.esae(feat)
        out3 = self.ltr(feat)
        # out = out1 + out2
        out = out1 + out2 +out3
        out = self.fuse(out)+0.1*identity
        return out