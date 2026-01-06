# dcenet_arch.py
"""
DCENet: Dual-domain Collaborative Enhancement Network for Digital Core Super-Resolution
Modified version: Replaced SparseAttentionLayerBlock with DualBlock and removed IDynamicLayerBlock
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

try:
    from basicsr.utils.registry import ARCH_REGISTRY
except Exception:
    # Minimal fallback registry if basicsr not available
    class _DummyReg:
        def register(self, cls):
            return cls
    ARCH_REGISTRY = _DummyReg()

# ------------------ utility functions ------------------
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

# ------------------ LayerNorm utilities ------------------
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        if LayerNorm_type =='BiasFree':
            raise NotImplementedError('BiasFree not used in this variant')
        else:
            self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# ------------------ Shallow extractor & helpers ------------------
class ShallowFeatureExtractor(nn.Module):
    def __init__(self, in_c=3, embed_dim=48):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False),
        )
    def forward(self, x):
        return self.layer(x)

class ResidualScale(nn.Module):
    def __init__(self, init_scale=0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_scale))
    def forward(self, x):
        return x * self.scale

# ------------------ CSAM ------------------
class CSAM(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        mid = max(1, dim // reduction)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, dim, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        ca_w = self.ca(x)
        sa_w = self.sa(x)
        return x * ca_w * sa_w + x

# ------------------ DFEM ------------------
class DFEM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False)
        )
        self.freq_scale = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.fuse = nn.Conv2d(dim*2, dim, 1, bias=False)
    def forward(self, x):
        B, C, H, W = x.shape
        try:
            freq = torch.fft.rfft2(x, norm='ortho')
            scale = self.freq_scale
            freq = freq * scale
            freq_ifft = torch.fft.irfft2(freq, s=(H, W), norm='ortho')
        except Exception:
            freq_ifft = x
        spa = self.spatial(x)
        out = self.fuse(torch.cat([freq_ifft, spa], dim=1))
        return out + x


# ------------------ PCFN / FeedForward ------------------
class PCFN(nn.Module):
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1)
        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        self.p_dim = p_dim
        self.hidden_dim = hidden_dim
    def forward(self, x):
        if self.training:
            x = self.act(self.conv_0(x))
            x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)
            x1 = self.act(self.conv_1(x1))
            x = self.conv_2(torch.cat([x1, x2], dim=1))
        else:
            x = self.act(self.conv_0(x))
            x[:, :self.p_dim, :, :] = self.act(self.conv_1(x[:, :self.p_dim, :, :]))
            x = self.conv_2(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, input_resolution=None):
        super(FeedForward, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.ffn_expansion_factor = ffn_expansion_factor
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class SparseAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, tlc_flag=True, tlc_kernel=48, activation='relu', input_resolution=None):
        super(SparseAttention, self).__init__()
        self.tlc_flag = tlc_flag    # TLC flag for validation and test

        self.dim = dim
        self.input_resolution = input_resolution

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.act = nn.Identity()

        # ['gelu', 'sigmoid'] is for ablation study
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()

        # [x2, x3, x4] -> [96, 72, 48]
        self.kernel_size = [tlc_kernel, tlc_kernel]

    def _forward(self, qkv):
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        # 核心区别：使用激活函数替代softmax实现稀疏注意力
        attn = self.act(attn)     # Sparse Attention due to ReLU's property
        out = (attn @ v)

        return out

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))

        if self.training or not self.tlc_flag:
            out = self._forward(qkv)
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

            out = self.project_out(out)
            return out

        # 测试模式下使用TLC分块处理
        qkv = self.grids(qkv)  # convert to local windows
        out = self._forward(qkv)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=qkv.shape[-2], w=qkv.shape[-1])
        out = self.grids_inverse(out)  # reverse

        out = self.project_out(out)
        return out

    # Code from [megvii-research/TLC] https://github.com/megvii-research/TLC
    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c // 3, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

    def flops(self):
        # calculate flops for window with token length of N
        h, w = self.input_resolution
        N = h * w

        flops = 0
        # x = self.qkv(x)
        flops += N * self.dim * self.dim * 3
        # x = self.qkv_dwconv(x)
        flops += N * self.dim * 3 * 9

        # qkv
        # CxC
        N_k = self.kernel_size[0] * self.kernel_size[1]
        N_num = ((h - 1)//self.kernel_size[0] + 1) * ((w - 1) // self.kernel_size[1] + 1)

        flops += N_num * self.num_heads * self.dim // self.num_heads * N_k * self.dim // self.num_heads
        # CxN CxC
        flops += N_num * self.num_heads * self.dim // self.num_heads * self.dim // self.num_heads * N_k

        # x = self.project_out(x)
        flops += N * self.dim * self.dim
        return flops
# ------------------ 新的 DualBlock (双路径块版本) ------------------

class DSABlock(nn.Module):
    def __init__(self, dim, sparse_attn, ffn=None):
        super().__init__()
        self.dim = dim

        # -------- 串联路径 --------
        # 第二阶段：PCFN
        self.norm2 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.ffn = ffn if ffn is not None else PCFN(dim)
        
        # 第三阶段：SparseAttention
        self.norm3 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.sparse_attn = sparse_attn

    def forward(self, x):
        # ===== 第二阶段：LayerNorm → PCFN → 残差 =====
        residual2 = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual2
        
        # ===== 第三阶段：LayerNorm → SparseAttention → 残差 =====
        residual3 = x
        x = self.norm3(x)
        x = self.sparse_attn(x)
        x = x + residual3
        
        return x


class SparseAttentionLayerBlock(nn.Module):
    def __init__(self, dim, restormer_ffn_type='PCFN', restormer_ffn_expansion_factor=2., 
                 tlc_flag=True, tlc_kernel=48, activation='relu', input_resolution=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
       
        # 创建SparseAttention实例
        sparse_attn = SparseAttention(
            dim=dim,
            num_heads=6,
            bias=False,
            tlc_flag=tlc_flag,
            tlc_kernel=tlc_kernel,
            activation=activation,
            input_resolution=input_resolution
        )
        
        # 创建FFN
        ffn = PCFN(dim)
        
        # 使用更新后的DualBlock（不再需要lite_glka参数）
        self.dsa_block = DSABlock(
            dim=dim,
            sparse_attn=sparse_attn,
            ffn=ffn
        )
        
    def forward(self, x):
        return self.dsa_block(x)
# ------------------MAA: Multi-scale Aggregate Attention ------------------

class MAA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)

        self.dw3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.dw5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.dw7 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.fuse = nn.Conv2d(dim * 3, dim, 1, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = self.pw(x)

        b1 = self.dw3(x)
        b2 = self.dw5(x)
        b3 = self.dw7(x)

        out = torch.cat([b1, b2, b3], dim=1)
        out = self.fuse(out)
        out = self.act(out)

        return identity * out



# ------------------ BuildBlock (修改后，只使用DualBlock) ------------------
class BuildBlock(nn.Module):
    def __init__(self, dim, blocks=3, buildblock_type='dual',
                 window_size=7, idynamic_num_heads=6, idynamic_ffn_type='PCFN', idynamic_ffn_expansion_factor=2., idynamic=True,
                 restormer_num_heads=6, restormer_ffn_type='PCFN', restormer_ffn_expansion_factor=2., tlc_flag=True, tlc_kernel=48,
                 activation='relu', input_resolution=None):
        super(BuildBlock, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.blocks = blocks
        self.buildblock_type = buildblock_type
        
        body = []
        for _ in range(blocks):
            body.append(SparseAttentionLayerBlock(
                dim, 
                restormer_ffn_type, 
                restormer_ffn_expansion_factor, 
                tlc_flag, 
                tlc_kernel, 
                activation, 
                input_resolution=input_resolution
            ))
        
        body.append(nn.Conv2d(dim, dim, 3, 1, 1))
        self.body = nn.Sequential(*body)
        
        # additions: MAA-> CSAM -> DFEM -> ResidualScale
        self.csam = CSAM(dim)
        self.maa = MAA(dim)
        self.dfem = DFEM(dim)
        self.res_scale = ResidualScale(0.1)
        
    def forward(self, x):
        out = self.body(x)
        out = self.csam(out)
        out = self.maa(out)
        out = self.dfem(out)
        out = self.res_scale(out)
        return out + x
    
    def flops(self):
        return 0
# ------------------ Upsample modules ------------------
class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)
    def flops(self):
        return 0

class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported.')
        super(Upsample, self).__init__(*m)

# ------------------ DCENet main ------------------
@ARCH_REGISTRY.register
class DCENetSR(nn.Module):
    def __init__(self,
                 in_chans=3,
                 dim=60,
                 groups=4,
                 blocks=2,
                 buildblock_type='dual',  # 默认使用dual类型
                 window_size=7, idynamic_num_heads=6, idynamic_ffn_type='PCFN', idynamic_ffn_expansion_factor=2.,
                 idynamic=True,
                 restormer_num_heads=6, restormer_ffn_type='PCFN', restormer_ffn_expansion_factor=2., tlc_flag=True, tlc_kernel=48, activation='relu',
                 upscale=4, img_range=1., upsampler='pixelshuffledirect', body_norm=False, input_resolution=None,
                 use_pam=True,  # 这个参数可以保留但不再使用
                 **kwargs):
        super().__init__()
        print(f'Using DCENetSR -> dim={dim}, groups={groups}, blocks={blocks}, buildblock_type={buildblock_type}')
        self.dim = dim
        self.input_resolution = input_resolution
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        # shallow feature extractor
        self.overlap_embed = ShallowFeatureExtractor(in_chans, dim)
        # body
        m_body = []
        if body_norm:
            m_body.append(LayerNorm(dim, LayerNorm_type='WithBias'))
        for i in range(groups):
            m_body.append(BuildBlock(dim, blocks, buildblock_type,
                                     window_size, idynamic_num_heads, idynamic_ffn_type, idynamic_ffn_expansion_factor, idynamic,
                                     restormer_num_heads, restormer_ffn_type, restormer_ffn_expansion_factor, tlc_flag, tlc_kernel, activation, input_resolution=input_resolution))
        if body_norm:
            m_body.append(LayerNorm(dim, LayerNorm_type='WithBias'))
        m_body.append(nn.Conv2d(dim, dim, 3, 1, 1))
        self.deep_feature_extraction = nn.Sequential(*m_body)
        # upsample & recon
        self.upsample = Upsample(upscale, dim)
        self.reconstruction = nn.Conv2d(dim, in_chans, 3, 1, 1)
        # 移除PAM，使用Identity占位符
        self.pam = nn.Identity()

    def forward(self, x):
        x = (x - self.mean.to(x.device))
        shallow = self.overlap_embed(x)
        deep = self.deep_feature_extraction(shallow)
        deep = deep + shallow
        out = self.upsample(deep)
        out = self.reconstruction(out)
        out = out + F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out = out + self.mean.to(out.device)
        return out