import os, sys
from turtle import forward
sys.path.append(os.getcwd())
# print(sys.path)
import torch
import torch.nn as nn
import torch.nn.functional as F 

from mmcv.cnn import ConvModule
from collections import OrderedDict
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from logging import raiseExceptions
from einops import rearrange
from thop import profile
from torchstat import stat


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(_init_weights)



    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(_init_weights)


    def forward(self, x, H, W):
        # 需要H和W是因为efficient self-atten需要用conv对序列长度缩减
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1: # sequence reduce，原来是H*W，reduce后变成H/sr * W/sr
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_) # (b, h*w, c)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module): # 一个Transformer层
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(_init_weights)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(_init_weights)


    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x) # (b, H*W, c)
        return x, H, W


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), 
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace = True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace = True)  
            )
    def forward(self, x):
        return self.conv(x)


class PatchExpand(nn.Module):
    """ Image to Patch Expanding
        (b, c, h, w) -> (b, c', 2h, 2w) 
        -> (b, 2h*2w, c')
    """
    def __init__(self, in_chans, out_chans, cat_chans, mode="bilinear", dbc=False):
        super().__init__()

        self.norm = nn.Identity()
        
        if mode == "bilinear":
            self.up = nn.Upsample(scale_factor=2)
            mid_chans = in_chans
        elif mode == "convt":
            self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2,)
            self.norm = nn.BatchNorm2d(out_chans)
            mid_chans = out_chans
        if dbc:
            self.db = DoubleConv(out_chans, out_chans)
        else:
            self.db = nn.Identity()
        self.linear = nn.Linear(cat_chans + mid_chans, out_chans)
        self.apply(_init_weights)

class PatchExpandLiCatDown(nn.Module): 
    # 用PatchExpandLi扩大空间缩小通道
    # 与前面的cat起来，然后再用linear降c的维
    def __init__(self, dim, catx_dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.catx_dim = catx_dim
        self.expand = PatchExpandLi(dim, dim_scale) # 通道dim减半
        self.ch_down = nn.Linear(dim//2 + catx_dim, dim//2) # 保持通道dim减半

    def forward(self, x, resolution, catx):
        """
        空间扩大一倍，通道缩小一倍
        x: (B, H*W, C) -> (B, H*W, 2C) -> (B, 2H, 2W, C/2) -> (B, 2H*2W, C/2)
        cat后，通道是放大了一倍，故再缩小一倍
        """
        x = self.expand(x, resolution)

        x = torch.cat([x, catx], dim=-1) # (B, 2H*2W, C+C')
        x = self.ch_down(x) # (B, 2H*2W, C)
        return x # (B, 2H*2W, C)


class PatchExpandLi(nn.Module):
    # 用Linear将通道升维，reshape到宽高，同时channel减半
    def __init__(self, dim, dim_scale=2, reduce_chan=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        if reduce_chan: # 通道减半
            self.expand = nn.Linear(dim, int(dim_scale**2/2)*dim, bias=False)
            self.output_dim = dim // 2
        else: # 通道不变
            self.expand = nn.Linear(dim, dim_scale**2*dim, bias=False)
            self.output_dim = dim
        
        self.norm = norm_layer(self.output_dim)

    def forward(self, x, resolution):
        """
        x: B, H*W, C
        """
        H, W = resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                      p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B, -1, C//(self.dim_scale**2))
        x = self.norm(x)
        return x


class SegEncoder(nn.Module):
    """
        embed_dims：每个stage的输出channel
        num_heads：每个stage中transformer的头数
        mlp_ratios：四个stage的transformerBlock中MLP中放大的倍数
        drop_rate：Attention内最后proj-drop和MLP内的drop
        attn_drop_rate：Attention中的drop，默认为0
        drop_path_rate：transformBlock中Atten和MLP间的dropPath
        depths：每个stage中的Transformer Block的个数
        sr_ratio: Attention中的序列降维，[8, 4, 2, 1]对应[64, 16, 4, 1]，因为是在h和w方向分别降sr倍
    """
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[2, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.1,
                 attn_drop_rate=0.1, drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], dec_dims=[64, 128, 256, 512], **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims # 添加
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule 
        cur = 0
        self.pat_emb = nn.ModuleList([])
        self.trans_blocks = nn.ModuleList([])
        for j in range(len(depths)):
            if j > 0:
                cur += depths[j-1]
                patch_embed = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[j-1], embed_dim=embed_dims[j])
            else: # j==0
                if patch_size == 2:
                    patch_embed = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=in_chans, embed_dim=embed_dims[j])
                elif patch_size == 4:
                    patch_embed = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[j])
                else:
                    raiseExceptions("patch_size error")
                pass
            self.pat_emb.append(patch_embed)
            # transformer encoder
            block = nn.ModuleList([Block(
                dim=embed_dims[j], num_heads=num_heads[j], mlp_ratio=mlp_ratios[j], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[j])
                for i in range(depths[j])])
            norm = norm_layer(embed_dims[j])
            block.append(norm)
            self.trans_blocks.append(block)
        self.apply(_init_weights)


    def forward(self, x):
        B = x.shape[0]
        imtermediates = []
        resolutions = []
        for j in range(len(self.depths)):
            # stage j
            x, H1, W1 = self.pat_emb[j](x) # (b, c0, h, w) -> (b, h/2*w/2, c1)
            resolutions.append((H1, W1))
            for i, blk in enumerate(self.trans_blocks[j]):
                if isinstance(blk, Block):
                    x = blk(x, H1, W1)
                else: # norm
                    x = blk(x)
            imtermediates.append(x)
            x = x.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
            # (b, c1, h/2, w/2) 为了下一个patch embedding
        return x, imtermediates, resolutions


class UDecoder(nn.Module):
    """
        embed_dims：每个stage的输出channel
        num_heads：每个stage中transformer的头数
        mlp_ratios：四个stage的transformerBlock中MLP中放大的倍数
        drop_rate：Attention内最后proj-drop和MLP内的drop
        attn_drop_rate：Attention中的drop，默认为0
        drop_path_rate：transformBlock中Atten和MLP间的dropPath
        depths：每个stage中的Transformer Block的个数
        sr_ratio: Attention中的序列降维，[8, 4, 2, 1]对应[64, 16, 4, 1]，因为是在h和w方向分别降sr倍
    """
    def __init__(self, patch_size=2, num_classes=6, embed_dims=[64, 128, 256, 512],
                 num_heads=[2, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.1,
                 attn_drop_rate=0.1, drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims # 添加
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule 
        cur = sum(depths)
        self.pat_exp = nn.ModuleList([])
        self.trans_blocks = nn.ModuleList([])
        for j in range(len(depths)-2, -1, -1):
            # j: 2, 1, 0
            patch_exp = PatchExpandLiCatDown(dim=embed_dims[j+1], catx_dim=embed_dims[j])
            self.pat_exp.append(patch_exp)
            # transformer encoder
            cur -= depths[j] # up3
            block = nn.ModuleList([Block(
                dim=embed_dims[j], num_heads=num_heads[j], mlp_ratio=mlp_ratios[j], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur - i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[j])
                for i in range(depths[j])])
            norm = norm_layer(embed_dims[j])
            block.append(norm)
            self.trans_blocks.append(block)
        
        self.final_up = PatchExpandLi(embed_dims[0], dim_scale=patch_size, reduce_chan=False)
        self.cls = nn.Conv2d(embed_dims[0], num_classes, 1, bias=False)
        self.apply(_init_weights)

    def forward(self, x, imtermediates, resolutions):
        B = x.size(0)
        nl = len(self.depths)
        # for j in range(len(self.depths)-2, -1, -1): # j: 2, 1, 0
        for j in range(len(self.pat_exp)): # j: 0, 1, 2
            x = self.pat_exp[j](x, resolutions[nl-1-j], imtermediates[nl-2-j]) 
            # (b, c, h, w) -> (b, 2h*2w, c/2) + (b, 2h*2w, c/2) -> (b, 2h*2w, c/2)
            H, W = resolutions[nl-1-j-1]
            for i, blk in enumerate(self.trans_blocks[j]):
                if isinstance(blk, Block):
                    x = blk(x, H, W)
                else:
                    x = blk(x)

        x = self.final_up(x, resolutions[0]) # (b, h*w, c1)
        H, W = resolutions[0]
        H, W = self.patch_size*H, self.patch_size*W
        x = x.view(B, H, W, self.embed_dims[0]).permute(0, 3, 1, 2) # (b, c, h, w)
        x = self.cls(x) # (b, nc, h, w)
        return x


class USegformerLi(nn.Module):
    """
        embed_dims：每个stage的输出channel
        num_heads：每个stage中transformer的头数
        mlp_ratios：四个stage的transformerBlock中MLP中放大的倍数
        drop_rate：Attention内最后proj-drop和MLP内的drop
        attn_drop_rate：Attention中的drop，默认为0
        drop_path_rate：transformBlock中Atten和MLP间的dropPath
        depths：每个stage中的Transformer Block的个数
        sr_ratio: Attention中的序列降维，[8, 4, 2, 1]对应[64, 16, 4, 1]，因为是在h和w方向分别降sr倍
    """
    def __init__(self, patch_size=2, in_chans=1, num_classes=6, embed_dims=[64, 128, 256, 512],
                 num_heads=[2, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.1,
                 attn_drop_rate=0.1, drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims # 添加
        self.enc = SegEncoder(patch_size, in_chans, num_classes, embed_dims,
                 num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate,
                 attn_drop_rate, drop_path_rate, norm_layer,
                 depths, sr_ratios, **kwargs)
        self.dec = UDecoder(patch_size, num_classes, embed_dims,
                 num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate,
                 attn_drop_rate, drop_path_rate, norm_layer,
                 depths, sr_ratios, **kwargs)
    
    def forward(self, x):
        x, intermediates, resolutions = self.enc(x)
        # x = (b, c, h, w)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, -1, C).contiguous() # (b, hw, c)
        x = self.dec(x, intermediates, resolutions)
        return x


class USegformer(nn.Module):
    """
        embed_dims：每个stage的输出channel
        num_heads：每个stage中transformer的头数
        mlp_ratios：四个stage的transformerBlock中MLP中放大的倍数
        drop_rate：Attention内最后proj-drop和MLP内的drop
        attn_drop_rate：Attention中的drop，默认为0
        drop_path_rate：transformBlock中Atten和MLP间的dropPath
        depths：每个stage中的Transformer Block的个数
        sr_ratio: Attention中的序列降维，[8, 4, 2, 1]对应[64, 16, 4, 1]，因为是在h和w方向分别降sr倍
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[2, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.1,
                 attn_drop_rate=0.1, drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], dec_dims=[64, 128, 256, 512], **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims # 添加

        # patch_embed    # img_size没用 
        if patch_size == 2:
            self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=3, stride=2, in_chans=in_chans, embed_dim=embed_dims[0])
        elif patch_size == 4:
            self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        else:
            raiseExceptions("patch_size error")
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size//4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size//8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size//16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2] # center
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # up
        self.patch_exp3 = PatchExpand(embed_dims[3], dec_dims[2], embed_dims[2]) # (b, c4, h/16, w/16) -> (b, h/8*w/8, c3')
        cur -= depths[2] # up3
        self.block3a = nn.ModuleList([Block(
            dim=dec_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur - i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3a = norm_layer(dec_dims[2]) # 8

        cur -= depths[1] # up2
        self.patch_exp2 = PatchExpand(dec_dims[2], dec_dims[1], embed_dims[1])
        self.block2a = nn.ModuleList([Block(
            dim=dec_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur - i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2a = norm_layer(dec_dims[1]) # 4

        cur -= depths[0] # up1
        self.patch_exp1 = PatchExpand(dec_dims[1], dec_dims[0], embed_dims[0])
        self.block1a = nn.ModuleList([Block(
            dim=dec_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur - i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1a = norm_layer(dec_dims[0]) # 2

        self.cls = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(dec_dims[0], dec_dims[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(dec_dims[0], num_classes, kernel_size=1)
        )
        self.apply(_init_weights)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H1, W1 = self.patch_embed1(x) # (b, c0, h, w) -> (b, h/2*w/2, c1)
        for i, blk in enumerate(self.block1):
            x = blk(x, H1, W1)
        x = self.norm1(x) # (b, h/2*w/2, c1)
        c1 = x
        x = x.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        # outs.append(x) # (b, c1, h/2, w/2)

        # stage 2
        x, H2, W2 = self.patch_embed2(x) # (b, c1, h/2, w/2) -> (b, h/4*w/4, c2)
        for i, blk in enumerate(self.block2):
            x = blk(x, H2, W2)
        x = self.norm2(x) # (b, h/4*w/4, c2)
        c2 = x # (b, h/4*w/4, c2)
        x = x.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
        # outs.append(x) # (b, c2, h/4, w/4)

        # stage 3
        x, H3, W3 = self.patch_embed3(x) # (b, c2, h/4, w/4) -> (b, h/8*w/8, c3)
        for i, blk in enumerate(self.block3):
            x = blk(x, H3, W3)
        x = self.norm3(x)
        c3 = x # (b, h/8*w/8, c3)
        x = x.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
        # outs.append(x) # (b, c3, h/8, w/8)

        # center
        x, H4, W4 = self.patch_embed4(x) # (b, c3, h/8, w/8) -> (b, h/16*w/16, c4)
        for i, blk in enumerate(self.block4):
            x = blk(x, H4, W4)
        x = self.norm4(x)
        x = x.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous() # (b, c4, h/16, w/16)


        # dec
        # stage 3
        x = self.patch_exp3(x, c3) # # (b, c4, h/16, w/16) -> (b, h/8*w/8, c3')
        for i, blk in enumerate(self.block3a):
            x = blk(x, H3, W3)
        x = self.norm3a(x) # (b, h/8*w/8, c3')
        x = x.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous() # (b, c3, h/8, w/8)

        # stage 2
        x = self.patch_exp2(x, c2) # (b, c3', h/8, w/8) -> (b, h/4*w/4, c2')
        for i, blk in enumerate(self.block2a):
            x = blk(x, H2, W2)
        x = self.norm2a(x) # (b, h/4*w/4, c2')
        x = x.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous() # (b, c2, h/4, w/4)

        # stage 1
        x = self.patch_exp1(x, c1) # # (b, c2', h/4, w/4) -> (b, h/2*w/2, c1')
        for i, blk in enumerate(self.block1a):
            x = blk(x, H1, W1)
        x = self.norm1a(x) # (b, h/2*w/2, c1')
        x = x.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous() # (b, c1, h/2, w/2)

        x = self.cls(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)

        return x



class DWConv(nn.Module):
    # 原文说mlp里加这个depth-wise conv足够提供transformer的position information，
    # 0填充以泄露位置信息？
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x): # 先将hw维度展平，然后把通道放到最后
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x # (b, h*w, c')

class MLP2(nn.Module):
    """
    Linear Embedding (b, c, h, w) -> (b, h*w, c')
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        # self.up = nn.ConvTranspose2d(input_dim, input_dim, kernel_size=3, stride=2)
        # self.bn = nn.BatchNorm2d(input_dim)
        # self.act = nn.CELU()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x): # (b, c, h, w)先将hw维度展平，然后把通道放到最后
        # x = self.act(self.bn(self.up(x)))
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x # (b, h*w, c')


class Decoder(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides=None, in_channels=128, embedding_dim=256, num_classes=20, **kwargs):
        super(Decoder, self).__init__() 
        self.in_channels = in_channels # 添加
        self.num_classes = num_classes
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP2(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP2(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP2(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP2(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.dropout = nn.Dropout2d(0.1)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            # norm_cfg=dict(type='SyncBN', requires_grad=True),
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        # self.up = nn.Sequential(
        #     nn.ConvTranspose2d(embedding_dim*2, embedding_dim, 2, 2),
        #     nn.BatchNorm2d(embedding_dim),
        #     nn.ReLU(),
        # )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        # x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        x = inputs
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape # (b, c, h, w)

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        # _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        # 以上每个都是(b, emb_dim, h/ps, w/ps)
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        # (b, 4*emb_dim, h/ps, w/ps) -> (b, emb_dim, h/ps, w/ps)
        # _c = self.up(_c)
        x = self.dropout(_c)
        x = self.linear_pred(x) # (b, 6, h/ps, w/ps)

        return x


class oriDecoder(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, intermediate_dims=[64, 128, 256, 512], embedding_dim=256, num_classes=20, **kwargs):
        super(oriDecoder, self).__init__() 
        self.intermediate_dims = intermediate_dims # 添加
        self.num_classes = num_classes
        linears = []
        for i in range(len(intermediate_dims)):
            li = nn.Linear(intermediate_dims[i], embedding_dim)
            linears.append(li)
        self.linears = nn.ModuleList(linears)
        self.dropout = nn.Dropout2d(0.1)
        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, intermediates, resolutions):
        # len=4, 1/4,1/8,1/16,1/32
        # x = intermediate_dims  每个为(b, h*w, c)

        projs = []
        b = intermediates[0].size(0)
        for i in range(len(self.intermediate_dims)):
            t = self.linears[i](intermediates[i]).permute(0, 2, 1).reshape(b, -1, *resolutions[i])
            t = F.interpolate(t, size=resolutions[0], mode='bilinear', align_corners=False)
            projs.append(t)
        # 以上每个都是(b, emb_dim, h/ps, w/ps)
        fuse = self.linear_fuse(torch.cat(projs, dim=1))
        x = self.dropout(fuse)
        x = self.linear_pred(x) # (b, 6, h/ps, w/ps)
        return x


class oriSegformer(nn.Module):
    """
        embed_dims：每个stage的输出channel
        num_heads：每个stage中transformer的头数
        mlp_ratios：四个stage的transformerBlock中MLP中放大的倍数
        drop_rate：Attention内最后proj-drop和MLP内的drop
        attn_drop_rate：Attention中的drop，默认为0
        drop_path_rate：transformBlock中Atten和MLP间的dropPath
        depths：每个stage中的Transformer Block的个数
        sr_ratio: Attention中的序列降维，[8, 4, 2, 1]对应[64, 16, 4, 1]，因为是在h和w方向分别降sr倍
    """
    def __init__(self, patch_size=2, in_chans=1, num_classes=6, embed_dims=[64, 128, 256, 512], dec_dim=256,
                 num_heads=[2, 4, 8, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.1,
                 attn_drop_rate=0.1, drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims # 添加
        self.enc = SegEncoder(patch_size, in_chans, num_classes, embed_dims,
                 num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate,
                 attn_drop_rate, drop_path_rate, norm_layer,
                 depths, sr_ratios, **kwargs)
        self.dec = oriDecoder(intermediate_dims=embed_dims, num_classes=num_classes, embed_dims=dec_dim, **kwargs)
    
    def forward(self, x):
        x, intermediates, resolutions = self.enc(x)
        # x=(b, c, h, w)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, -1, C).contiguous() # (b, hw, c)
        x = self.dec(intermediates, resolutions)
        return x


class upDecoder(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, intermediate_dims=[64, 128, 256, 512], embedding_dim=256, num_classes=20, **kwargs):
        super().__init__() 
        self.intermediate_dims = intermediate_dims # 添加
        self.num_classes = num_classes
        linears = []
        for i in range(len(intermediate_dims)):
            li = nn.Linear(intermediate_dims[i], embedding_dim)
            linears.append(li)
        self.linears = nn.ModuleList(linears)
        self.dropout = nn.Dropout2d(0.1)
        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.up = PatchExpandLi(embedding_dim, reduce_chan=False)
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.embedding_dim = embedding_dim

    def forward(self, intermediates, resolutions):
        # len=4, 1/4,1/8,1/16,1/32
        # x = intermediate_dims  每个为(b, h*w, c)

        projs = []
        b = intermediates[0].size(0)
        for i in range(len(self.intermediate_dims)):
            t = self.linears[i](intermediates[i]).permute(0, 2, 1).reshape(b, -1, *resolutions[i])
            t = F.interpolate(t, size=resolutions[0], mode='bilinear', align_corners=False)
            projs.append(t)
        # 以上每个都是(b, emb_dim, h/ps, w/ps)
        fuse = self.linear_fuse(torch.cat(projs, dim=1)) # (b, 2*emb_dim, h/ps, w/ps)
        x = self.dropout(fuse)
        x = x.permute(0, 2, 3, 1).view(b, -1, self.embedding_dim).contiguous()
        #print(x.shape)
        x = self.up(x, resolutions[0])
        # print(resolutions[0])
        #print(x.shape)
        x = x.permute(0, 2, 1).view(b, self.embedding_dim, 2*resolutions[0][0], 2*resolutions[0][1]).contiguous()
        x = self.linear_pred(x) # (b, 6, h/ps, w/ps)
        return x

class upSegformer(nn.Module):
    """
        embed_dims：每个stage的输出channel
        num_heads：每个stage中transformer的头数
        mlp_ratios：四个stage的transformerBlock中MLP中放大的倍数
        drop_rate：Attention内最后proj-drop和MLP内的drop
        attn_drop_rate：Attention中的drop，默认为0
        drop_path_rate：transformBlock中Atten和MLP间的dropPath
        depths：每个stage中的Transformer Block的个数
        sr_ratio: Attention中的序列降维，[8, 4, 2, 1]对应[64, 16, 4, 1]，因为是在h和w方向分别降sr倍
    """
    def __init__(self, patch_size=2, in_chans=1, num_classes=6, embed_dims=[64, 128, 256, 512], dec_dim=256,
                 num_heads=[2, 4, 8, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.1,
                 attn_drop_rate=0.1, drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims # 添加
        self.enc = SegEncoder(patch_size, in_chans, num_classes, embed_dims,
                 num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate,
                 attn_drop_rate, drop_path_rate, norm_layer,
                 depths, sr_ratios, **kwargs)
        self.dec = upDecoder(intermediate_dims=embed_dims, num_classes=num_classes, embed_dims=dec_dim, **kwargs)
    
    def forward(self, x):
        x, intermediates, resolutions = self.enc(x)
        # x=(b, c, h, w)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, -1, C).contiguous() # (b, hw, c)
        x = self.dec(intermediates, resolutions)
        return x


class UDecoderHyper(nn.Module):
    """
        embed_dims：每个stage的输出channel
        num_heads：每个stage中transformer的头数
        mlp_ratios：四个stage的transformerBlock中MLP中放大的倍数
        drop_rate：Attention内最后proj-drop和MLP内的drop
        attn_drop_rate：Attention中的drop，默认为0
        drop_path_rate：transformBlock中Atten和MLP间的dropPath
        depths：每个stage中的Transformer Block的个数
        sr_ratio: Attention中的序列降维，[8, 4, 2, 1]对应[64, 16, 4, 1]，因为是在h和w方向分别降sr倍
    """
    def __init__(self, patch_size=2, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[2, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.1,
                 attn_drop_rate=0.1, drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims # 添加
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule 
        cur = sum(depths)
        self.pat_exp = nn.ModuleList([])
        self.trans_blocks = nn.ModuleList([])
        for j in range(len(depths)-2, -1, -1):
            # j: 2, 1, 0
            patch_exp = PatchExpandLiCatDown(dim=embed_dims[j+1], catx_dim=embed_dims[j])
            self.pat_exp.append(patch_exp)
            # transformer encoder
            cur -= depths[j] # up3
            block = nn.ModuleList([Block(
                dim=embed_dims[j], num_heads=num_heads[j], mlp_ratio=mlp_ratios[j], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur - i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[j])
                for i in range(depths[j])])
            norm = norm_layer(embed_dims[j])
            block.append(norm)
            self.trans_blocks.append(block)
        
        self.apply(_init_weights)

    def forward(self, x, imtermediates, resolutions):
        B = x.size(0)
        nl = len(self.depths)
        outs = [x]
        outs_resolutions = [resolutions[-1]]
        # for j in range(len(self.depths)-2, -1, -1): # j: 2, 1, 0
        for j in range(len(self.pat_exp)): # j: 0, 1, 2
            x = self.pat_exp[j](x, resolutions[nl-1-j], imtermediates[nl-2-j]) 
            # (b, c, h, w) -> (b, 2h*2w, c/2) + (b, 2h*2w, c/2) -> (b, 2h*2w, c/2)
            H, W = resolutions[nl-1-j-1]
            for i, blk in enumerate(self.trans_blocks[j]):
                if isinstance(blk, Block):
                    x = blk(x, H, W)
                else:
                    x = blk(x)
                    outs.append(x)
                    outs_resolutions.append((H, W))

        return x, outs, outs_resolutions


class USegformerHyper(nn.Module):
    """
        embed_dims：每个stage的输出channel
        num_heads：每个stage中transformer的头数
        mlp_ratios：四个stage的transformerBlock中MLP中放大的倍数
        drop_rate：Attention内最后proj-drop和MLP内的drop
        attn_drop_rate：Attention中的drop，默认为0
        drop_path_rate：transformBlock中Atten和MLP间的dropPath
        depths：每个stage中的Transformer Block的个数
        sr_ratio: Attention中的序列降维，[8, 4, 2, 1]对应[64, 16, 4, 1]，因为是在h和w方向分别降sr倍
    """
    def __init__(self, patch_size=2, in_chans=1, num_classes=6, embed_dims=[64, 128, 256, 512],
                 num_heads=[2, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.1,
                 attn_drop_rate=0.1, drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims # 添加
        self.enc = SegEncoder(patch_size, in_chans, num_classes, embed_dims,
                 num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate,
                 attn_drop_rate, drop_path_rate, norm_layer,
                 depths, sr_ratios, **kwargs)
        self.dec1 = UDecoderHyper(patch_size, num_classes, embed_dims,
                 num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate,
                 attn_drop_rate, drop_path_rate, norm_layer,
                 depths, sr_ratios, **kwargs)
        self.dec2 = oriDecoder(intermediate_dims=embed_dims, num_classes=num_classes, embed_dims=256, **kwargs)
    
    def forward(self, x):
        x, intermediates, resolutions = self.enc(x)
        # x = (b, c, h, w)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, -1, C).contiguous() # (b, hw, c)
        x, intermediates, resolutions = self.dec1(x, intermediates, resolutions)
        x = self.dec2(intermediates[::-1], resolutions[::-1])
        return x


def get_parameter_number( model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__=="__main__":

    
    x = torch.rand(2, 3, 704, 256)
    model = USegformerHyper(
            patch_size=2, in_chans=3, num_classes=6, embed_dims=[64, 128, 256, 512], dec_dim=256, num_heads=[2, 2, 4, 8],
            qkv_bias=True, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1)

    print(model)
    y = model(x)
    print(y.shape)
    # print(F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False).shape)

    # x = torch.rand(1, 1, 128, 128)
    # ######可训练参数量
    # model = oriSegformer()
    # total = sum([param.nelement() for param in model.parameters()])
    # print("Number of parameter: %.2fM" % (total/1e6))

    # ######计算量(FLOPs)和参数量(Params)

    # input = torch.randn(1, 1, 128, 128)
    # flops, params = profile(model, inputs=(input,))
    # print('flops: ', flops, 'params: ', params)
