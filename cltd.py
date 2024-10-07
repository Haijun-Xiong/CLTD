import torch
import torch.nn as nn
from einops import rearrange, reduce
import torch.fft as fft


class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv3d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias, **kwargs)

    def forward(self, ipts):
        '''
            ipts: [n, c, s]
            outs: [n, c, s]
        '''
        outs = self.conv3d(ipts)
        return outs


class CPAG(nn.Module):
    def __init__(self, in_c, h_in, h_out, w):
        super(CPAG, self).__init__()
        self.w = w
        self.softmax = nn.Softmax(dim=-1)
        self.conv_q = BasicConv1d(in_c, in_c)
        self.conv_k = BasicConv1d(in_c, in_c)
        self.gamma = nn.Parameter(torch.ones(1))


        self.fc3 = nn.Sequential(
            BasicConv1d(h_in * w, h_in * w, 1, stride=1, padding=0, groups=h_in),
            nn.LeakyReLU(inplace=True)
        )
        self.fc4 = nn.Sequential(
            BasicConv1d(h_in * w, h_out * w, 1, stride=1, padding=0, groups=w),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        g = reduce(x, 'n c s h w -> n c s', 'max') + reduce(x, 'n c s h w -> n c s', 'mean')
        q = self.conv_q(g)
        k = self.conv_k(g)
        Mc = torch.einsum('nci,ncj->nij', q, k) / (torch.abs(self.gamma) + 1.0e-6)
        Mc = self.softmax(Mc)

        v = reduce(x, 'n c s h w -> n (h w) s', 'max') + reduce(x, 'n c s h w -> n (h w) s', 'mean')
        v = self.fc3(v)
        g = torch.einsum('bik,bjk->bij', Mc, v) # n s (h w)
        g = rearrange(g, 'n s (h w) -> n (w h) s', w=self.w)
        g = self.fc4(g)
        g = rearrange(g, 'n (w h) s -> n 1 s h w', w=self.w)
        return g
    
class FPH(nn.Module):
    def __init__(self, k, in_c=128, out_c=128):
        super(FPH, self).__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(in_c * 2, out_c)
        )

    def forward(self, x):
        n, c, s, h, w = x.size()
        x = rearrange(x, 'n c s h w -> (n s) c h w')
        x = fft.fftshift(fft.fft2(x, dim=(-2, -1), s=None))
        x = torch.cat([x.real, x.imag], dim=1)
        new_h = int((h - self.k) / 2)
        x = x[..., new_h:new_h + self.k, new_h:new_h + self.k]
        x = rearrange(x, '(n s) c h w -> n s (h w) c', n=n)
        x = self.mlp(x)
        x = reduce(x, 'n s k c -> n c k', 'max') + reduce(x, 'n s k c -> n c k', 'mean')
        return x