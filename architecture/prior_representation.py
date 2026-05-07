import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class FANLayer(nn.Module):

    def __init__(self, input_dim, output_dim, p_ratio=0.25, activation='gelu', use_p_bias=True):
        super(FANLayer, self).__init__()
        assert 0 < p_ratio < 0.5, "p_ratio must be between 0 and 0.5"
        self.p_ratio = p_ratio
        p_output_dim = int(output_dim * p_ratio)
        g_output_dim = output_dim - 2 * p_output_dim
        self.input_linear_p = nn.Linear(input_dim, p_output_dim, bias=use_p_bias)
        self.input_linear_g = nn.Linear(input_dim, g_output_dim)
        self.activation = getattr(F, activation) if isinstance(activation, str) else activation

    def forward(self, src):
        # src: (B, input_dim)
        g = self.activation(self.input_linear_g(src))        # → (B, g_output_dim)
        p = self.input_linear_p(src)                         # → (B, p_output_dim)
        return torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)  # → (B, output_dim)


class DDFF(nn.Module):
    def __init__(self, num_patches, embed_dims, p_ratio=0.25, activation='gelu', use_p_bias=True):
        super().__init__()
        #spatial FAN：input_dim=N → output_dim=N
        self.space_fan = FANLayer(input_dim=num_patches,
                                  output_dim=num_patches,
                                  p_ratio=p_ratio,
                                  activation=activation,
                                  use_p_bias=use_p_bias)
        # channel FAN：input_dim=C → output_dim=C
        self.channel_fan = FANLayer(input_dim=embed_dims,
                                    output_dim=embed_dims,
                                    p_ratio=p_ratio,
                                    activation=activation,
                                    use_p_bias=use_p_bias)
        # LayerNorm
        self.norm_space = nn.LayerNorm(num_patches)
        self.norm_chan  = nn.LayerNorm(embed_dims)

    def forward(self, x):
        # x: (B, C, N)
        B, C, N = x.shape

        x_space = self.norm_space(x.transpose(1,2))    # (B, N, C)
        x_space = self.space_fan(x_space)
        x_space = x_space.transpose(1,2)
        x = x + x_space
        
        x_chan = self.norm_chan(x)
       
        x_chan = self.channel_fan(x_chan)
        
        x = x + x_chan

        return x
    

from .lightweight_blocks import LFB
class PRM(nn.Module):

    def __init__(self, in_chans=120, embed_dim=64, block_num=2, stage=1, group=4, patch_expansion=0.5, channel_expansion=4):
        super(PRM, self).__init__()

        self.group = group

        self.pixel_unshuffle = nn.PixelUnshuffle(4) # [B, C, H, W]>[B, C * 4², H/4, W/4]
        self.conv1 = LFB(3,in_chans*16,in_chans*4,embed_dim,nn.GELU,True,1)
        
        self.blocks = nn.ModuleList()
        for i in range(block_num):
            block = nn.Sequential(LFB(3,embed_dim,embed_dim*4,embed_dim,nn.GELU,True,1))
                
                
            self.blocks.append(block)

        self.conv2 = LFB(3,embed_dim,embed_dim*4,embed_dim,nn.GELU,True,1) 
        self.pool = nn.AdaptiveAvgPool2d((group, group))

        self.FAN = DDFF(num_patches=group*group, embed_dims=embed_dim,
                                   p_ratio=0.25, activation='gelu', use_p_bias=True)
        self.end = nn.Sequential(
                nn.Linear(embed_dim, embed_dim*4),
                nn.GELU(),)
        

    def forward(self, inp_img, gt=None):
        if gt is not None:
            x = torch.cat([gt, inp_img], dim=1)
        else:
            x = inp_img

        x = self.pixel_unshuffle(x)
        
        x = self.conv1(x)
        
        for block in self.blocks:
            
            x = block(x) + x
                   
        x = self.conv2(x)
    
        x = self.pool(x)
        

        x = rearrange(x, 'b c h w-> b (h w) c') # [2, 64, 4, 4] 
        
        x = self.FAN(x)
        x = self.end(x)
        
        return x # [2, wh, 64*4]