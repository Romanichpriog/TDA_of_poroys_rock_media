import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.utils import _pair

def lp_convolution(input, out_channels, weight, bias, C, log2p, kernel_size, stride, padding, dilation, groups, constraints=False):
    if log2p is None:
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    x = torch.arange(kernel_size[0]).to(input.device)
    y = torch.arange(kernel_size[1]).to(input.device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    x0 = (x.max() - x.min())/2
    y0 = (y.max() - y.min())/2
    offset = torch.stack([xx - x0, yy - y0], dim=0)
    Z = torch.einsum('cij, jmn -> cimn', C, offset)
    mask = torch.exp( - Z.abs().pow(2**log2p[:, None, None, None]).sum(dim=1, keepdim=True) )
    return F.conv2d(input, weight * mask, bias, stride, padding, dilation, groups)

class LpConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, sigma, log2p, kernel_size, stride=1, padding=0, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        sigma = _pair(sigma)
        C_00 = 1 / (sigma[0] + 1e-4)
        C_11 = 1 / (sigma[1] + 1e-4)
        if log2p is None:
            self.register_buffer('log2p', None)
            self.register_buffer('C', None)
        else:
            self.log2p = nn.Parameter(torch.full((out_channels,), float(log2p)))
            self.C = nn.Parameter(torch.tensor([[C_00, 0.0], [0.0, C_11]]).repeat(out_channels,1,1))
    def forward(self, input):
        return lp_convolution(
            input, self.out_channels, self.weight, self.bias, self.C, self.log2p,
            self.kernel_size, self.stride, self.padding, self.dilation, self.groups
        )

    @classmethod
    def convert(cls, conv2d, log2p, transfer_params=False, set_requires_grad={}):
        in_channels = conv2d.in_channels
        out_channels = conv2d.out_channels
        kernel_size = conv2d.kernel_size
        stride = conv2d.stride
        padding = conv2d.padding
        dilation = conv2d.dilation
        groups = conv2d.groups
        padding_mode = conv2d.padding_mode
        bias = conv2d.bias is not None
        sigma = (kernel_size[0] * 0.5, kernel_size[1] * 0.5)
        new_conv2d = cls(in_channels, out_channels, sigma, log2p, kernel_size, stride=stride, padding=padding, bias=bias)
        if transfer_params:
            new_conv2d.weight.data.copy_(conv2d.weight.data)
            if bias:
                new_conv2d.bias.data.copy_(conv2d.bias.data)
        for param, reqgrad in set_requires_grad.items():
            getattr(new_conv2d, param).requires_grad = reqgrad
        return new_conv2d


def LpConvert(module: nn.Module, log2p: float, transfer_params=True, set_requires_grad={}):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(
                module, name,
                LpConv2d.convert(
                    child, log2p=log2p,
                    transfer_params=transfer_params,
                    set_requires_grad=set_requires_grad
                )
            )
        else:
            LpConvert(child, log2p, transfer_params=transfer_params, set_requires_grad=set_requires_grad)
    return module

class PI_LpCNN(nn.Module):
    def __init__(self, input_channels=1, base_filters=16, num_layers=3, img_size=32):
        super().__init__()
        layers = []
        in_ch = input_channels
        for i in range(num_layers):
            out_ch = base_filters * (2**i)
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        # Adaptive Pool
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_ch * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        x = self.cnn(x)
        x = self.global_pool(x)
        x = self.regressor(x)
        return x

def make_model_with_lpconv(
        input_channels=1, base_filters=32, num_layers=4, img_size=32,
        log2p=1, lp_learnable=True):
    model = PI_LpCNN(input_channels, base_filters, num_layers=4, img_size=32)
    set_requires_grad = dict(log2p=lp_learnable, C=lp_learnable)
    model = LpConvert(
        model,
        log2p=log2p,
        transfer_params=True,
        set_requires_grad=set_requires_grad
    )
    return model
