# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))


# my_model
# my_attention
class ChannelAttention(nn.Module):           # Channel Attention Module
    # 最大和均值池化后分别使用卷积、Relu、卷积最后concat后用sigmoid（但卷积不共享参数）
    def __init__(self, c1, mid_reduction=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(c1, c1 // mid_reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(c1 // mid_reduction, c1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)

        max_out = self.max_pool(x)
        max_out = self.fc1(max_out)
        max_out = self.relu(max_out)
        max_out = self.fc2(max_out)

        out = avg_out + max_out
        out = self.sigmoid(out)
        return out
class SpatialAttention(nn.Module):           # Spatial Attention Module
    def __init__(self, kernel=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel, padding=kernel//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out
class CBAM(nn.Module):
    def __init__(self, c1, reduction=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Apply channel attention
        channel_attention = self.channel_attention(x)
        x = x * channel_attention

        # Apply spatial attention
        spatial_attention = self.spatial_attention(x)
        x = x * spatial_attention

        return x

class Attention(nn.Module):
    # YOLOV11的自注意力Attention模块
    def __init__(self, dim, num_heads=8,
                 attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (
                (q.transpose(-2, -1) @ k) * self.scale
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x
class ChannelAttention2(nn.Module):           # Channel Attention Module
    def __init__(self, c1, mid_reduction=4):
        super(ChannelAttention2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(c1, c1 // mid_reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(c1 // mid_reduction, c1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)

        max_out = self.max_pool(x)
        max_out = self.fc1(max_out)
        max_out = self.relu(max_out)
        max_out = self.fc2(max_out)

        out = avg_out + max_out
        out = self.sigmoid(out)
        return out
class SpatialAttention2(nn.Module):           # Spatial Attention Module
    def __init__(self, kernel=3):
        super(SpatialAttention2, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel, padding=kernel//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out
class LCAM(nn.Module):  # LCAM改进版
    # 在CAM的基础上额外添加一个空洞卷积，以增加感受野（可以只在通道分割后使用）
    def __init__(self, c1, mid_reduction=4):
        super(LCAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 双路径共享的MLP结构
        self.fc1 = nn.Conv2d(c1, c1 // mid_reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(c1 // mid_reduction, c1, kernel_size=1, bias=False)

        # 空洞卷积增强上下文感知
        self.dilated_conv = nn.Conv2d(c1, c1, kernel_size=3, dilation=2, padding=2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 双路径池化+MLP
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))

        # 特征融合后接空洞卷积
        out = self.dilated_conv(avg_out + max_out)
        return self.sigmoid(out)
class LSAM(nn.Module):  # LSAM改进版
    # 将PAM中的7*7卷积改为3*3深度可分离卷积，降低参数量，在减少参数量的同时增强局部空间关系的建模能力
    def __init__(self,kernels = 3):
        super(LSAM, self).__init__()
        # 深度可分离卷积替换普通卷积
        self.depthwise = nn.Conv2d(2, 2, kernel_size=kernels, padding=1, groups=2, bias=False)
        self.pointwise = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 双通道特征拼接
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)

        # 深度可分离卷积处理
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.sigmoid(x)
class SRAM7(nn.Module):
    # 1、均分为4块分别进行最大和平均池化，最后组合回去.
    # 2、将原始的特征图整体进行最大和平均池化.
    # 3、将1和2的结果使用可学习权重进行加权相加.
    # 4、将通道注意力后结果的C均分为4部分，分别应用空间注意力.
    # 5、对通道注意力后的结果整体应用空间注意力.
    # 6、将4和5的结果使用可更新参数进行加权和.
    # 7、借鉴SPA注意力的思路，将空间注意力的分块分别进行3、5、7、9尺度的卷积，提取多尺度特征，再进行空间注意力（乘的是特征提取后的结果）
    # 8、借鉴SPA中的自注意机制，在最开始先进行自注意力
    def __init__(self, c1, reduction=4, kernel_size=3):
        super(SRAM7, self).__init__()
        self.channel_attention = ChannelAttention2(c1, reduction)
        self.spatial_attention = SpatialAttention2(kernel_size)
        # 初始化卷积层
        c_mid = c1 // 4
        c_mid2 = c1-c_mid*3
        self.conv1 = nn.Conv2d(c_mid, c_mid, kernel_size=3, groups=1, padding=1)
        self.conv2 = nn.Conv2d(c_mid, c_mid, kernel_size=5, groups=2, padding=2)
        self.conv3 = nn.Conv2d(c_mid, c_mid, kernel_size=7, groups=4, padding=3)
        self.conv4 = nn.Conv2d(c_mid2, c_mid2, kernel_size=9, groups=4, padding=4)
        self.attn = Attention(c1, attn_ratio=0.5, num_heads=c1 // 64)


        # 定义可学习的权重参数
        self.weight_split_channel = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于split的权重
        self.weight_overall_channel = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于整体通道注意力的权重
        self.weight_split_spatial = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于split的权重
        self.weight_overall_spatial = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于整体通道注意力的权重
    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.attn(x)
        # 计算划分大小
        h_split = H // 2
        w_split = W // 2

        # 生成四个部分
        parts_channel = [
            x[:, :, 0:h_split, 0:w_split],  # Left top
            x[:, :, 0:h_split, w_split:W],  # Right top
            x[:, :, h_split:H, 0:w_split],  # Left bottom
            x[:, :, h_split:H, w_split:W],  # Right bottom
        ]

        # 对每个部分应用通道注意力
        attended_parts = []
        for part in parts_channel:
            channel_attention = self.channel_attention(part)
            attended_part = part * channel_attention
            attended_parts.append(attended_part)

        # 拼接回去
        top_row = torch.cat(attended_parts[:2], dim=3)
        bottom_row = torch.cat(attended_parts[2:], dim=3)
        x_split = torch.cat([top_row, bottom_row], dim=2)

        # 计算整体通道注意力
        overall_channel_attention = self.channel_attention(x)

        # 使用可学习权重进行加权相加，x_split 应用 channel_attention
        combined = self.weight_split_channel * x_split + self.weight_overall_channel * (x * overall_channel_attention)

        # 应用空间注意力
        # 通道划分为4个部分
        c_split = C // 4

        # 生成四个部分 (通道分块)
        parts_spatial = [
            combined[:, :c_split, :, :],  # First quarter of channels
            combined[:, c_split:2 * c_split, :, :],  # Second quarter of channels
            combined[:, 2 * c_split:3 * c_split, :, :],  # Third quarter of channels
            combined[:, 3 * c_split:, :, :]  # Fourth quarter of channels
        ]

        # 1. 对每个部分应用空间注意力
        attended_parts_spatial = []
        for i, part in enumerate(parts_spatial):
            if i == 0:
                part = self.conv1(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention

            elif i == 1:
                part = self.conv2(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention
            elif i == 2:
                part = self.conv3(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention
            elif i == 3:
                part = self.conv4(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention

            attended_parts_spatial.append(attended_part)

            # 拼接回去（通道重组）
        combined_split = torch.cat(attended_parts_spatial, dim=1)

        # 2. 计算整体的空间注意力
        overall_spatial_attention = self.spatial_attention(combined)

        # 3. 使用可学习权重进行加权相加
        output = self.weight_split_spatial * combined_split + self.weight_overall_spatial * (combined * overall_spatial_attention)

        return output
class SRAM8(nn.Module):
    # 1、均分为4块分别进行最大和平均池化，最后组合回去.
    # 2、将原始的特征图整体进行最大和平均池化.
    # 3、将1和2的结果使用可学习权重进行加权相加.
    # 4、将通道注意力后结果的C均分为4部分，分别应用空间注意力.
    # 5、对通道注意力后的结果整体应用空间注意力.
    # 6、将4和5的结果使用可更新参数进行加权和.
    # 7、借鉴SPA注意力的思路，将空间注意力的分块分别进行3、5、7、9尺度的卷积，提取多尺度特征，再进行空间注意力（乘的是特征提取后的结果）
    # 8、借鉴SPA中的自注意机制，在最开始先进行自注意力
    # 9、将通道注意力换为LSAM。
    def __init__(self, c1, reduction=4, kernel_size=3):
        super(SRAM8, self).__init__()
        self.channel_attention = ChannelAttention2(c1, reduction)
        self.spatial_attention = LSAM(kernel_size)
        # 初始化卷积层
        c_mid = c1 // 4
        c_mid2 = c1-c_mid*3
        self.conv1 = nn.Conv2d(c_mid, c_mid, kernel_size=3, groups=1, padding=1)
        self.conv2 = nn.Conv2d(c_mid, c_mid, kernel_size=5, groups=2, padding=2)
        self.conv3 = nn.Conv2d(c_mid, c_mid, kernel_size=7, groups=4, padding=3)
        self.conv4 = nn.Conv2d(c_mid2, c_mid2, kernel_size=9, groups=4, padding=4)
        self.attn = Attention(c1, attn_ratio=0.5, num_heads=c1 // 64)


        # 定义可学习的权重参数
        self.weight_split_channel = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于split的权重
        self.weight_overall_channel = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于整体通道注意力的权重
        self.weight_split_spatial = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于split的权重
        self.weight_overall_spatial = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于整体通道注意力的权重
    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.attn(x)
        # 计算划分大小
        h_split = H // 2
        w_split = W // 2

        # 生成四个部分
        parts_channel = [
            x[:, :, 0:h_split, 0:w_split],  # Left top
            x[:, :, 0:h_split, w_split:W],  # Right top
            x[:, :, h_split:H, 0:w_split],  # Left bottom
            x[:, :, h_split:H, w_split:W],  # Right bottom
        ]

        # 对每个部分应用通道注意力
        attended_parts = []
        for part in parts_channel:
            channel_attention = self.channel_attention(part)
            attended_part = part * channel_attention
            attended_parts.append(attended_part)

        # 拼接回去
        top_row = torch.cat(attended_parts[:2], dim=3)
        bottom_row = torch.cat(attended_parts[2:], dim=3)
        x_split = torch.cat([top_row, bottom_row], dim=2)

        # 计算整体通道注意力
        overall_channel_attention = self.channel_attention(x)

        # 使用可学习权重进行加权相加，x_split 应用 channel_attention
        combined = self.weight_split_channel * x_split + self.weight_overall_channel * (x * overall_channel_attention)

        # 应用空间注意力
        # 通道划分为4个部分
        c_split = C // 4

        # 生成四个部分 (通道分块)
        parts_spatial = [
            combined[:, :c_split, :, :],  # First quarter of channels
            combined[:, c_split:2 * c_split, :, :],  # Second quarter of channels
            combined[:, 2 * c_split:3 * c_split, :, :],  # Third quarter of channels
            combined[:, 3 * c_split:, :, :]  # Fourth quarter of channels
        ]

        # 1. 对每个部分应用空间注意力
        attended_parts_spatial = []
        for i, part in enumerate(parts_spatial):
            if i == 0:
                part = self.conv1(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention

            elif i == 1:
                part = self.conv2(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention
            elif i == 2:
                part = self.conv3(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention
            elif i == 3:
                part = self.conv4(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention

            attended_parts_spatial.append(attended_part)

            # 拼接回去（通道重组）
        combined_split = torch.cat(attended_parts_spatial, dim=1)

        # 2. 计算整体的空间注意力
        overall_spatial_attention = self.spatial_attention(combined)

        # 3. 使用可学习权重进行加权相加
        output = self.weight_split_spatial * combined_split + self.weight_overall_spatial * (combined * overall_spatial_attention)

        return output
class SRAM9(nn.Module):
    # 1、均分为4块分别进行最大和平均池化，最后组合回去.
    # 2、将原始的特征图整体进行最大和平均池化.
    # 3、将1和2的结果使用可学习权重进行加权相加.
    # 4、将通道注意力后结果的C均分为4部分，分别应用空间注意力.
    # 5、对通道注意力后的结果整体应用空间注意力.
    # 6、将4和5的结果使用可更新参数进行加权和.
    # 7、借鉴SPA注意力的思路，将空间注意力的分块分别进行3、5、7、9尺度的卷积，提取多尺度特征，再进行空间注意力（乘的是特征提取后的结果）
    # 8、借鉴SPA中的自注意机制，在最开始先进行自注意力
    # 9、不使用整体特征图进行逐权重concat
    def __init__(self, c1, reduction=4, kernel_size=3):
        super(SRAM9, self).__init__()
        self.channel_attention = ChannelAttention2(c1, reduction)
        self.spatial_attention = SpatialAttention2(kernel_size)
        # 初始化卷积层
        c_mid = c1 // 4
        c_mid2 = c1-c_mid*3
        self.conv1 = nn.Conv2d(c_mid, c_mid, kernel_size=3, groups=1, padding=1)
        self.conv2 = nn.Conv2d(c_mid, c_mid, kernel_size=5, groups=2, padding=2)
        self.conv3 = nn.Conv2d(c_mid, c_mid, kernel_size=7, groups=4, padding=3)
        self.conv4 = nn.Conv2d(c_mid2, c_mid2, kernel_size=9, groups=4, padding=4)
        self.attn = Attention(c1, attn_ratio=0.5, num_heads=c1 // 64)


        # 定义可学习的权重参数
        # self.weight_split_channel = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于split的权重
        # self.weight_overall_channel = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于整体通道注意力的权重
        # self.weight_split_spatial = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于split的权重
        # self.weight_overall_spatial = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于整体通道注意力的权重
    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.attn(x)
        # 计算划分大小
        h_split = H // 2
        w_split = W // 2

        # 生成四个部分
        parts_channel = [
            x[:, :, 0:h_split, 0:w_split],  # Left top
            x[:, :, 0:h_split, w_split:W],  # Right top
            x[:, :, h_split:H, 0:w_split],  # Left bottom
            x[:, :, h_split:H, w_split:W],  # Right bottom
        ]

        # 对每个部分应用通道注意力
        attended_parts = []
        for part in parts_channel:
            channel_attention = self.channel_attention(part)
            attended_part = part * channel_attention
            attended_parts.append(attended_part)

        # 拼接回去
        top_row = torch.cat(attended_parts[:2], dim=3)
        bottom_row = torch.cat(attended_parts[2:], dim=3)
        x_split = torch.cat([top_row, bottom_row], dim=2)

        # # 计算整体通道注意力
        # overall_channel_attention = self.channel_attention(x)
        #
        # # 使用可学习权重进行加权相加，x_split 应用 channel_attention
        # combined = self.weight_split_channel * x_split + self.weight_overall_channel * (x * overall_channel_attention)
        combined = x_split

        # 应用空间注意力
        # 通道划分为4个部分
        c_split = C // 4

        # 生成四个部分 (通道分块)
        parts_spatial = [
            combined[:, :c_split, :, :],  # First quarter of channels
            combined[:, c_split:2 * c_split, :, :],  # Second quarter of channels
            combined[:, 2 * c_split:3 * c_split, :, :],  # Third quarter of channels
            combined[:, 3 * c_split:, :, :]  # Fourth quarter of channels
        ]

        # 1. 对每个部分应用空间注意力
        attended_parts_spatial = []
        for i, part in enumerate(parts_spatial):
            if i == 0:
                part = self.conv1(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention

            elif i == 1:
                part = self.conv2(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention
            elif i == 2:
                part = self.conv3(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention
            elif i == 3:
                part = self.conv4(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention

            attended_parts_spatial.append(attended_part)

            # 拼接回去（通道重组）
        combined_split = torch.cat(attended_parts_spatial, dim=1)

        # # 2. 计算整体的空间注意力
        # overall_spatial_attention = self.spatial_attention(combined)
        #
        # # 3. 使用可学习权重进行加权相加
        # output = self.weight_split_spatial * combined_split + self.weight_overall_spatial * (combined * overall_spatial_attention)
        output = combined_split
        return output
class SRAM10(nn.Module):
    # 1、均分为4块分别进行最大和平均池化，最后组合回去.
    # 2、将原始的特征图整体进行最大和平均池化.
    # 3、将1和2的结果使用可学习权重进行加权相加.
    # 4、将通道注意力后结果的C均分为4部分，分别应用空间注意力.
    # 5、对通道注意力后的结果整体应用空间注意力.
    # 6、将4和5的结果使用可更新参数进行加权和.
    # 7、借鉴SPA注意力的思路，将空间注意力的分块分别进行3、5、7、9尺度的卷积，提取多尺度特征，再进行空间注意力（乘的是特征提取后的结果）
    # 8、借鉴SPA中的自注意机制，在最开始先进行自注意力
    # 9、不使用分组卷积，使用普通卷积
    def __init__(self, c1, reduction=4, kernel_size=3):
        super(SRAM10, self).__init__()
        self.channel_attention = ChannelAttention2(c1, reduction)
        self.spatial_attention = SpatialAttention2(kernel_size)
        # 初始化卷积层
        c_mid = c1 // 4
        c_mid2 = c1-c_mid*3
        self.conv1 = nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c_mid, c_mid, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(c_mid, c_mid, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(c_mid2, c_mid2, kernel_size=9, padding=4)
        self.attn = Attention(c1, attn_ratio=0.5, num_heads=c1 // 64)


        # 定义可学习的权重参数
        self.weight_split_channel = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于split的权重
        self.weight_overall_channel = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于整体通道注意力的权重
        self.weight_split_spatial = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于split的权重
        self.weight_overall_spatial = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于整体通道注意力的权重
    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.attn(x)
        # 计算划分大小
        h_split = H // 2
        w_split = W // 2

        # 生成四个部分
        parts_channel = [
            x[:, :, 0:h_split, 0:w_split],  # Left top
            x[:, :, 0:h_split, w_split:W],  # Right top
            x[:, :, h_split:H, 0:w_split],  # Left bottom
            x[:, :, h_split:H, w_split:W],  # Right bottom
        ]

        # 对每个部分应用通道注意力
        attended_parts = []
        for part in parts_channel:
            channel_attention = self.channel_attention(part)
            attended_part = part * channel_attention
            attended_parts.append(attended_part)

        # 拼接回去
        top_row = torch.cat(attended_parts[:2], dim=3)
        bottom_row = torch.cat(attended_parts[2:], dim=3)
        x_split = torch.cat([top_row, bottom_row], dim=2)

        # 计算整体通道注意力
        overall_channel_attention = self.channel_attention(x)

        # 使用可学习权重进行加权相加，x_split 应用 channel_attention
        combined = self.weight_split_channel * x_split + self.weight_overall_channel * (x * overall_channel_attention)

        # 应用空间注意力
        # 通道划分为4个部分
        c_split = C // 4

        # 生成四个部分 (通道分块)
        parts_spatial = [
            combined[:, :c_split, :, :],  # First quarter of channels
            combined[:, c_split:2 * c_split, :, :],  # Second quarter of channels
            combined[:, 2 * c_split:3 * c_split, :, :],  # Third quarter of channels
            combined[:, 3 * c_split:, :, :]  # Fourth quarter of channels
        ]

        # 1. 对每个部分应用空间注意力
        attended_parts_spatial = []
        for i, part in enumerate(parts_spatial):
            if i == 0:
                part = self.conv1(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention

            elif i == 1:
                part = self.conv2(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention
            elif i == 2:
                part = self.conv3(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention
            elif i == 3:
                part = self.conv4(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention

            attended_parts_spatial.append(attended_part)

            # 拼接回去（通道重组）
        combined_split = torch.cat(attended_parts_spatial, dim=1)

        # 2. 计算整体的空间注意力
        overall_spatial_attention = self.spatial_attention(combined)

        # 3. 使用可学习权重进行加权相加
        output = self.weight_split_spatial * combined_split + self.weight_overall_spatial * (combined * overall_spatial_attention)

        return output
class SRAM11(nn.Module):
    # 1、均分为4块分别进行最大和平均池化，最后组合回去.
    # 2、将原始的特征图整体进行最大和平均池化.
    # 3、将1和2的结果使用可学习权重进行加权相加.
    # 4、将通道注意力后结果的C均分为4部分，分别应用空间注意力.
    # 5、对通道注意力后的结果整体应用空间注意力.
    # 6、将4和5的结果使用可更新参数进行加权和.
    # 7、借鉴SPA注意力的思路，将空间注意力的分块分别进行3、5、7、9尺度的卷积，提取多尺度特征，再进行空间注意力（乘的是特征提取后的结果）
    # 8、借鉴SPA中的自注意机制，在最开始先进行自注意力
    # 9、不使用整体特征图进行逐权重concat
    def __init__(self, c1, reduction=4, kernel_size=3):
        super(SRAM9, self).__init__()
        self.channel_attention = ChannelAttention2(c1, reduction)
        self.spatial_attention = SpatialAttention2(kernel_size)
        # 初始化卷积层
        c_mid = c1 // 4
        c_mid2 = c1-c_mid*3
        self.conv1 = nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c_mid, c_mid, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(c_mid, c_mid, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(c_mid2, c_mid2, kernel_size=9, padding=4)
        self.attn = Attention(c1, attn_ratio=0.5, num_heads=c1 // 64)


        # 定义可学习的权重参数
        # self.weight_split_channel = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于split的权重
        # self.weight_overall_channel = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于整体通道注意力的权重
        # self.weight_split_spatial = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于split的权重
        # self.weight_overall_spatial = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于整体通道注意力的权重
    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.attn(x)
        # 计算划分大小
        h_split = H // 2
        w_split = W // 2

        # 生成四个部分
        parts_channel = [
            x[:, :, 0:h_split, 0:w_split],  # Left top
            x[:, :, 0:h_split, w_split:W],  # Right top
            x[:, :, h_split:H, 0:w_split],  # Left bottom
            x[:, :, h_split:H, w_split:W],  # Right bottom
        ]

        # 对每个部分应用通道注意力
        attended_parts = []
        for part in parts_channel:
            channel_attention = self.channel_attention(part)
            attended_part = part * channel_attention
            attended_parts.append(attended_part)

        # 拼接回去
        top_row = torch.cat(attended_parts[:2], dim=3)
        bottom_row = torch.cat(attended_parts[2:], dim=3)
        x_split = torch.cat([top_row, bottom_row], dim=2)

        # # 计算整体通道注意力
        # overall_channel_attention = self.channel_attention(x)
        #
        # # 使用可学习权重进行加权相加，x_split 应用 channel_attention
        # combined = self.weight_split_channel * x_split + self.weight_overall_channel * (x * overall_channel_attention)
        combined = x_split

        # 应用空间注意力
        # 通道划分为4个部分
        c_split = C // 4

        # 生成四个部分 (通道分块)
        parts_spatial = [
            combined[:, :c_split, :, :],  # First quarter of channels
            combined[:, c_split:2 * c_split, :, :],  # Second quarter of channels
            combined[:, 2 * c_split:3 * c_split, :, :],  # Third quarter of channels
            combined[:, 3 * c_split:, :, :]  # Fourth quarter of channels
        ]

        # 1. 对每个部分应用空间注意力
        attended_parts_spatial = []
        for i, part in enumerate(parts_spatial):
            if i == 0:
                part = self.conv1(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention

            elif i == 1:
                part = self.conv2(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention
            elif i == 2:
                part = self.conv3(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention
            elif i == 3:
                part = self.conv4(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention

            attended_parts_spatial.append(attended_part)

            # 拼接回去（通道重组）
        combined_split = torch.cat(attended_parts_spatial, dim=1)

        # # 2. 计算整体的空间注意力
        # overall_spatial_attention = self.spatial_attention(combined)
        #
        # # 3. 使用可学习权重进行加权相加
        # output = self.weight_split_spatial * combined_split + self.weight_overall_spatial * (combined * overall_spatial_attention)
        output = combined_split
        return output
class CBAMSA(nn.Module):
    def __init__(self, c1, reduction=16, kernel_size=3):
        super(CBAMSA, self).__init__()
        self.channel_attention = ChannelAttention(c1, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.attn = Attention(c1, attn_ratio=0.5, num_heads=c1 // 64)

    def forward(self, x):
        x = x + self.attn(x)
        # Apply channel attention
        channel_attention = self.channel_attention(x)
        x = x * channel_attention

        # Apply spatial attention
        spatial_attention = self.spatial_attention(x)
        x = x * spatial_attention

        return x
class CBAMLG(nn.Module):
    # 1、均分为4块分别进行最大和平均池化，最后组合回去.
    # 2、将原始的特征图整体进行最大和平均池化.
    # 3、将1和2的结果使用可学习权重进行加权相加.
    # 4、将通道注意力后结果的C均分为4部分，分别应用空间注意力.
    # 5、对通道注意力后的结果整体应用空间注意力.
    # 6、将4和5的结果使用可更新参数进行加权和.

    # 9、不使用分组卷积，使用普通卷积
    def __init__(self, c1, reduction=4, kernel_size=3):
        super(CBAMLG, self).__init__()
        self.channel_attention = ChannelAttention2(c1, reduction)
        self.spatial_attention = SpatialAttention2(kernel_size)
        # 初始化卷积层
        c_mid = c1 // 4
        c_mid2 = c1-c_mid*3
        self.conv1 = nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c_mid, c_mid, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(c_mid, c_mid, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(c_mid2, c_mid2, kernel_size=9, padding=4)


        # 定义可学习的权重参数
        self.weight_split_channel = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于split的权重
        self.weight_overall_channel = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于整体通道注意力的权重
        self.weight_split_spatial = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于split的权重
        self.weight_overall_spatial = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 用于整体通道注意力的权重
    def forward(self, x):
        B, C, H, W = x.shape
        # 计算划分大小
        h_split = H // 2
        w_split = W // 2

        # 生成四个部分
        parts_channel = [
            x[:, :, 0:h_split, 0:w_split],  # Left top
            x[:, :, 0:h_split, w_split:W],  # Right top
            x[:, :, h_split:H, 0:w_split],  # Left bottom
            x[:, :, h_split:H, w_split:W],  # Right bottom
        ]

        # 对每个部分应用通道注意力
        attended_parts = []
        for part in parts_channel:
            channel_attention = self.channel_attention(part)
            attended_part = part * channel_attention
            attended_parts.append(attended_part)

        # 拼接回去
        top_row = torch.cat(attended_parts[:2], dim=3)
        bottom_row = torch.cat(attended_parts[2:], dim=3)
        x_split = torch.cat([top_row, bottom_row], dim=2)

        # 计算整体通道注意力
        overall_channel_attention = self.channel_attention(x)

        # 使用可学习权重进行加权相加，x_split 应用 channel_attention
        combined = self.weight_split_channel * x_split + self.weight_overall_channel * (x * overall_channel_attention)

        # 应用空间注意力
        # 通道划分为4个部分
        c_split = C // 4

        # 生成四个部分 (通道分块)
        parts_spatial = [
            combined[:, :c_split, :, :],  # First quarter of channels
            combined[:, c_split:2 * c_split, :, :],  # Second quarter of channels
            combined[:, 2 * c_split:3 * c_split, :, :],  # Third quarter of channels
            combined[:, 3 * c_split:, :, :]  # Fourth quarter of channels
        ]

        # 1. 对每个部分应用空间注意力
        attended_parts_spatial = []
        for i, part in enumerate(parts_spatial):
            if i == 0:
                part = self.conv1(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention

            elif i == 1:
                part = self.conv2(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention
            elif i == 2:
                part = self.conv3(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention
            elif i == 3:
                part = self.conv4(part)
                spatial_attention = self.spatial_attention(part)
                attended_part = part * spatial_attention

            attended_parts_spatial.append(attended_part)

            # 拼接回去（通道重组）
        combined_split = torch.cat(attended_parts_spatial, dim=1)

        # 2. 计算整体的空间注意力
        overall_spatial_attention = self.spatial_attention(combined)

        # 3. 使用可学习权重进行加权相加
        output = self.weight_split_spatial * combined_split + self.weight_overall_spatial * (combined * overall_spatial_attention)

        return output

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class SPDConv(nn.Module):
    """
    SpaceToDepth 类继承自 nn.Module，用于实现空间到深度的转换。
    这种转换通过重排输入张量的元素降低其空间维度，同时增加深度维度，
    另外还会对结果进行卷积操作。

    参数:
    out_channels: 输出通道数，在转换后的张量上执行卷积操作时使用。
    """

    def __init__(self, input_channels, out_channels):
        """
        初始化 SpaceToDepth 模块。

        参数:
        input_channels: 输入的通道数。
        out_channels: 转换后输出的通道数。
        """
        super(SPDConv, self).__init__()  # 调用父类的构造函数进行初始化
        # 定义卷积层，将拼接后的输出通道映射到指定的 out_channels
        self.conv = nn.Conv2d(4 * input_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        前向传播函数，实现输入 x 的空间到深度的转换，并进行卷积操作。

        参数:
        x: 输入张量，需要进行空间到深度转换的数据，形状为 (N, C, H, W)。

        返回:
        转换并经过卷积处理后的张量。
        """
        # 对输入 x 进行空间到深度的转换操作，并在指定维度上进行拼接
        spatial_output = torch.cat([
            x[..., ::2, ::2],  # 取偶行和偶列
            x[..., 1::2, ::2],  # 取奇行和偶列
            x[..., ::2, 1::2],  # 取偶行和奇列
            x[..., 1::2, 1::2]  # 取奇行和奇列
        ], 1)  # 在深度维度上拼接，结果通道数为 4 * input_channels

        # 经过卷积层处理，输出的通道数为 out_channels
        conv_output = self.conv(spatial_output)

        return conv_output
class MBD(nn.Module):
    """MDC模块，处理输入特征图并在多个分支中进行处理"""

    def __init__(self, in_channels, out_channels):
        """初始化MDC模块

        参数:
            in_channels (int): 输入特征图的通道数
            out_channels (int): 输出特征图的通道数
        """
        super(MBD, self).__init__()

        self.intermediate_channels = in_channels // 2  # 中间层通道数，可以根据需求调整

        # 1x1卷积，用于减少通道数
        self.reduce_channels = Conv(in_channels, self.intermediate_channels, k=1, act=True)

        # 分支1: 最大池化下采样
        self.branch1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #Conv(self.intermediate_channels, self.intermediate_channels, k=1, act=True)  # 在 pooling 后的 1x1 卷积
        )

        # 分支2: 深度可分离卷积，3x3卷积核
        self.branch2 = DWConv(self.intermediate_channels, self.intermediate_channels, k=3, s=2, act=True)

        # 分支3: 常规3x3卷积
        self.branch3 = Conv(self.intermediate_channels, self.intermediate_channels, k=3, s=2, act=True)

        # 分支4: SPDConv
        self.branch4 = SPDConv(self.intermediate_channels, self.intermediate_channels)

        # 最后的1x1卷积，用于调整通道数回原始值
        self.final_conv = Conv(self.intermediate_channels * 4, out_channels, k=1, act=True)  # 更新为 *4 以包括新的分支

    def forward(self, x):
        """前向传播，应用所有的分支和连接操作"""
        # 先通过1x1卷积减少通道数
        x = self.reduce_channels(x)

        # 各个分支的输出
        out1 = self.branch1(x)  # 分支1: 最大池化
        out2 = self.branch2(x)  # 分支2: 深度可分离卷积
        out3 = self.branch3(x)  # 分支3: 常规卷积
        out4 = self.branch4(x)  # 新分支4: SPDConv

        # 在通道维度上连接所有的分支输出
        out = torch.cat((out1, out2, out3, out4), dim=1)  # 更新为 *4 以包括新的分支

        # 通过最后的1x1卷积调整通道数
        return self.final_conv(out)
class MBD2(nn.Module):
    """MDC模块，处理输入特征图并在多个分支中进行处理"""
    # SPD output change mid_channel to 4*mid_channel
    def __init__(self, in_channels, out_channels):
        """初始化MDC模块

        参数:
            in_channels (int): 输入特征图的通道数
            out_channels (int): 输出特征图的通道数
        """
        super(MBD2, self).__init__()

        self.intermediate_channels = in_channels // 2  # 中间层通道数，可以根据需求调整

        # 1x1卷积，用于减少通道数
        self.reduce_channels = Conv(in_channels, self.intermediate_channels, k=1, act=True)

        # 分支1: 最大池化下采样
        self.branch1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #Conv(self.intermediate_channels, self.intermediate_channels, k=1, act=True)  # 在 pooling 后的 1x1 卷积
        )

        # 分支2: 深度可分离卷积，3x3卷积核
        self.branch2 = DWConv(self.intermediate_channels, self.intermediate_channels, k=3, s=2, act=True)

        # 分支3: 常规3x3卷积
        self.branch3 = Conv(self.intermediate_channels, self.intermediate_channels, k=3, s=2, act=True)

        # 分支4: SPDConv
        self.branch4 = SPDConv(self.intermediate_channels, 4*self.intermediate_channels)

        # 最后的1x1卷积，用于调整通道数回原始值
        self.final_conv = Conv(self.intermediate_channels * 7, out_channels, k=1, act=True)  # 更新为 *4 以包括新的分支

    def forward(self, x):
        """前向传播，应用所有的分支和连接操作"""
        # 先通过1x1卷积减少通道数
        x = self.reduce_channels(x)

        # 各个分支的输出
        out1 = self.branch1(x)  # 分支1: 最大池化
        out2 = self.branch2(x)  # 分支2: 深度可分离卷积
        out3 = self.branch3(x)  # 分支3: 常规卷积
        out4 = self.branch4(x)  # 新分支4: SPDConv

        # 在通道维度上连接所有的分支输出
        out = torch.cat((out1, out2, out3, out4), dim=1)  # 更新为 *4 以包括新的分支

        # 通过最后的1x1卷积调整通道数
        return self.final_conv(out)
