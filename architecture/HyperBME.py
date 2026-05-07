from collections import defaultdict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from .dun_utils import *
from .lightweight_blocks import LFB

class ChannelAtt(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(ChannelAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpaCNN(nn.Module):
    def __init__(self, in_channels, SK_size = 3, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=SK_size, padding=int(SK_size/2), stride=strides)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=SK_size, padding=int(SK_size/2), stride=strides)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        return F.relu(Y)

class SpatialCNNAtt(nn.Module):
    def __init__(self,in_channels = 64, SK_size = 3, kernel_size=3):
        super(SpatialCNNAtt, self).__init__()
        self.scnn = SpaCNN(in_channels=in_channels, SK_size=SK_size)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.scnn(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SAFFM_Flexible(nn.Module):
    def __init__(self, in_channel, out_channel, SK_size = 3):
        super(SAFFM_Flexible, self).__init__()
        self.spaCnnAtt = SpatialCNNAtt(in_channels=in_channel, SK_size=SK_size)
        self.chaAtt = ChannelAtt(in_channels=in_channel)
        self.conv1 = nn.Conv2d(in_channel*2, out_channel, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
    def forward(self, x ,y):
        f = self.chaAtt(x) * self.spaCnnAtt(y)
        z = torch.cat([f*x, f*y], dim=1)
        return F.relu(self.bn1(self.conv1(z)))
    
class HSIB(nn.Module):
    def __init__(self, opt, dim, num_heads, bias=False, LayerNorm_type="WithBias", use_prior=False):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.norm = LayerNorm(dim, LayerNorm_type=LayerNorm_type)

        self.qk = DWPWConvL(dim, dim * 4, 2, 2, 0)
        self.spatial_down = nn.Conv2d(1, 1, 2, 2, 0)
        self.qkv_dwconv = nn.Conv2d(
            dim * 4, dim * 4,
            kernel_size=3, stride=1, padding=1,
            groups=dim * 2, bias=bias
        )

        # Base value branch
        self.qkv_v_base = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1, groups=dim),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=(3, 5), padding=(1, 2), bias=False, groups=dim)
        )

        # Enhanced value branch for spectral feature refinement
        self.qkv_v_enhance = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1, groups=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, groups=1, bias=bias)
        )


        self.beta = nn.Parameter(torch.full((1, dim * 2, 1, 1), 0.1))

        self.qkv_v_down = DWPWConvL(dim * 2, dim * 2, 2, 2, 0)
        self.project_out = DWPWConvTL(dim * 2, dim, 4, 2, 1)

    def forward(self, x, spatial_interaction=None, prior=None):
        b, c, h, w = x.shape
        x_norm = self.norm(x)

        # -------- Q/K  --------
        qk = self.qkv_dwconv(self.qk(x_norm))
        q, k = qk.chunk(2, dim=1)

        # -------- V --------
        v_base = self.qkv_v_base(x_norm)
        v_enhance = self.qkv_v_enhance(x_norm)
        
        # Residual Enhancement Injection
        v = v_base + torch.tanh(self.beta) * v_enhance

        if spatial_interaction is not None:
            spatial_interaction_down = self.spatial_down(spatial_interaction)
            q = q * spatial_interaction_down
            k = k * spatial_interaction_down
            v = v * spatial_interaction

        v_down = self.qkv_v_down(v)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_down = rearrange(v_down, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v_down)
        out = rearrange(
            out,
            'b head c (h w) -> b (head c) h w',
            head=self.num_heads, h=h // 2, w=w // 2
        )
        out = self.project_out(out)

        return out, v

class PCB(nn.Module):
    def __init__(self, opt, dim, num_heads, bias=False, LayerNorm_type="WithBias", use_prior=True):
        super().__init__()
        self.opt = opt
        self.dim = dim
        self.use_prior = use_prior
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_zl = nn.Linear(256, dim * 4) 
        self.project_out = DWPWConvTL(dim * 2, dim, 4, 2, 1)

        # Global Uncertainty Estimator
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 4, dim), 
            nn.ReLU(),
            nn.Linear(dim, dim),     
            nn.Sigmoid()
        )

    def forward(self, x, q, spatial_interaction=None, prior=None):
        # x: unused, just for interface compatibility
        b, _, h, w = x.shape

        kv = self.qkv_zl(prior) 
        k_prior, v_prior = kv.chunk(2, dim=-1) 

        q_global = torch.mean(q, dim=[2, 3])          # [b, 2*dim]
        k_prior_global = torch.mean(k_prior, dim=1)   # [b, 2*dim]
        
        
        q_global = F.normalize(q_global, dim=-1)
        k_prior_global = F.normalize(k_prior_global, dim=-1)
        
        gate_input = torch.cat([q_global, k_prior_global], dim=1) # [b, 4*dim]
        b = gate_input.shape[0]
        trust_gate = self.gate_mlp(gate_input).view(b, self.dim, 1, 1)
        # =================================

        # Attention Interaction
        q_reshaped = rearrange(q, 'b (head c) w h -> b head (w h) c', head=self.num_heads)
        k_reshaped = rearrange(k_prior, 'b n (head c) -> b head n c', head=self.num_heads)
        v_reshaped = rearrange(v_prior, 'b n (head c) -> b head n c', head=self.num_heads)

        k_reshaped = torch.nn.functional.normalize(k_reshaped, dim=-1)

        attn = (q_reshaped @ k_reshaped.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v_reshaped)
        out = rearrange(out, 'b head (h w) c-> b (head c) h w', head=self.num_heads, h=h//2, w=w//2)
        out = self.project_out(out) 

        out = out * trust_gate

        return out


class MBlocks(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim,
                 bias=False
    ):
        super(MBlocks, self).__init__()
        self.mb = LFB(3,out_dim,out_dim*2,out_dim,nn.GELU,True,1)

    def forward(self, x):
        out = self.mb(x)
        return out


class DSRB(nn.Module):
    def __init__(self,input_channels ,reduction_N = 32):
        super(DSRB, self).__init__()
        self.point_wise = nn.Conv2d(input_channels,reduction_N,kernel_size=1,padding=0,bias=False)    
        self.depth_wise = nn.Sequential(nn.Conv2d(reduction_N, reduction_N, kernel_size=(3, 3),padding=1),nn.BatchNorm2d(reduction_N),nn.ReLU(),)

        self.conv3D = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1,1,3),padding=(0,0,1),stride=(1,1,1),bias=False)
        self.bn = nn.BatchNorm2d(reduction_N)
        self.relu = nn.ReLU()
        
    def forward(self,x):
            x_1 = self.point_wise(x)  
            x_2 = self.depth_wise(x_1)       
            x_2=x_1+x_2
            
            #DSC
            x_3 = x_1.unsqueeze(1)
            x_3 = self.conv3D(x_3)
            x_3 = x_3.squeeze(1)
            x = torch.cat((x_2,x_3),dim=1)
            
            return x


class MIB(nn.Module):
    def __init__(self, opt, dim, num_heads, use_prior=True, use_ch_split=False):
        super().__init__()
        self.opt = opt
        self.use_prior = use_prior
        self.split_ch  = 2 if use_ch_split else 1
        dw_channel = dim//self.split_ch * opt.DW_Expand
        dw_channel_1 = dim//self.split_ch
       
        
        self.spatial_branch = DSRB(dw_channel_1, dw_channel_1 // 2)

        self.spatial_gelu = nn.GELU()
        self.spatial_conv = nn.Conv2d(in_channels=dw_channel, out_channels=dim//self.split_ch, kernel_size=1, padding=0, stride=1, groups=1, bias=opt.bias)
        self.spatial_interaction = nn.Conv2d(dw_channel, 1, kernel_size=1, bias=opt.bias)

        self.spectral_branch = HSIB(
            opt,
            dim//self.split_ch, 
            num_heads=num_heads, 
            bias=opt.bias,
            LayerNorm_type=opt.LayerNorm_type,
            use_prior=use_prior,
        )
        
        if use_prior:
            self.HIM = PCB(
                opt,
                dim//self.split_ch, 
                num_heads=num_heads, 
                bias=opt.bias,
                LayerNorm_type=opt.LayerNorm_type,
                use_prior=use_prior,
            )

        self.spectral_interaction = nn.Sequential(
            nn.Conv2d(dim//self.split_ch, dim // 8, kernel_size=1, bias=opt.bias),
            LayerNorm(dim // 8, opt.LayerNorm_type),
            nn.GELU(),
            nn.Conv2d(dim // 8, dw_channel, kernel_size=1, bias=opt.bias),
        )
        self.spec_to_prior_inter = DWPWConvL(2*dim//self.split_ch, 2*dim//self.split_ch,2,2,0)
        
        self.add_ss = nn.Sequential(nn.Conv2d(dim,dim,1,1,0))  
        
        if self.split_ch>1:
            in_dim = 3*dim//self.split_ch
        else:
            in_dim = 2*dim
        self.add_ssp = nn.Sequential(nn.Conv2d(in_dim,dim,1,1,0))
        
        self.ffn = Residual(
            FFN_FN(
                dim=dim, 
                ffn_name=opt.ffn_name,
                ffn_expansion_factor=opt.FFN_Expand, 
                bias=opt.bias,
                LayerNorm_type=opt.LayerNorm_type
            )
        )

    def forward(self, x, prior=None):
        log_dict = defaultdict(list)
        b, c, h, w = x.shape
        if self.split_ch>1:
            x_spa, x_spec = x.chunk(2, dim=1)
        else:
            x_spa = x
            x_spec = x

        # --- Spatial Branch ---
        spatial_identity = x_spa
        spatial_fea = self.spatial_branch(x_spa)
        
        spatial_interaction = self.spatial_interaction(spatial_fea)
        log_dict['block_spatial_interaction'] = spatial_interaction
        spatial_fea = self.spatial_gelu(spatial_fea)

        # --- Spectral Branch ---
        spectral_identity = x_spec
        
        spectral_delta, qkv = self.spectral_branch(x_spec, spatial_interaction) 

        # Interaction
        spectral_interaction_map = self.spectral_interaction(
            F.adaptive_avg_pool2d(spectral_delta, output_size=1)) 
        spectral_interaction_map = torch.sigmoid(spectral_interaction_map).tile((1, 1, h, w))
        spatial_fea = spectral_interaction_map * spatial_fea
            
        # --- Prior Branch 
        z_fea = 0
        if self.use_prior:
            qkv = self.spec_to_prior_inter(qkv)
            z_fea = self.HIM(x_spec, q=qkv, prior=prior)

        # --- Residuals & Fusion ---
        spatial_fea = self.spatial_conv(spatial_fea)
        spatial_fea = spatial_identity + spatial_fea # Spatial Residual
        log_dict['block_spatial_fea'] = spatial_fea

    
        spectral_fea = spectral_identity + spectral_delta # Spectral Residual
        log_dict['block_spectral_fea'] = spectral_fea
        
        if self.split_ch > 1:
            fea_ss = torch.concat([spatial_fea, spectral_fea], dim=1)
        else:
            fea_ss = spatial_fea + spectral_fea
            
        fea_ss = self.add_ss(fea_ss)
        
        fea = torch.concat([fea_ss, z_fea], dim=1)
        fea = self.add_ssp(fea)
        
        out = self.ffn(fea)

        return out, log_dict



class MRN(nn.Module):
    def __init__(self, opt,use_prior=False):
        super().__init__()
        self.use_prior = use_prior
 
        self.opt = opt
        self.embedding = nn.Conv2d(opt.in_dim, opt.dim, kernel_size=1, stride=1, padding=0, bias=opt.bias)
        

                
        self.Encoder = nn.ModuleList([
            
        MIB(opt = opt, dim = opt.dim * 2 ** 0, num_heads = 2 ** 0,
                          use_prior=self.use_prior,use_ch_split=False),
        MIB(opt = opt, dim = opt.dim * 2 ** 1, num_heads = 2 ** 1,
                          use_prior=self.use_prior,use_ch_split=True),
                      
        ])

        self.BottleNeck = MIB(opt = opt, dim = opt.dim * 2 ** 2, num_heads = 2 ** 2,
                                            use_prior=self.use_prior,use_ch_split=True)
                      
        self.Decoder = nn.ModuleList([

                    MIB(opt = opt, dim = opt.dim * 2 ** 1, num_heads = 2 ** 1,
                                      use_prior=self.use_prior,use_ch_split=True),
              MIB(opt = opt, dim = opt.dim * 2 ** 0, num_heads = 2 ** 0,
                          use_prior=self.use_prior,use_ch_split=False)
                      
        
        ])
                


        self.BlockInteractions = nn.ModuleList([
            BlockInteraction(opt.dim * 7, opt.dim * 1),
            BlockInteraction(opt.dim * 7, opt.dim * 2)
        ])

        self.Downs = nn.ModuleList([
            DownSample(opt.dim * 2 ** 0, bias=opt.bias),
            DownSample(opt.dim * 2 ** 1, bias=opt.bias)
        ])

        self.Ups = nn.ModuleList([
            UpSample(opt.dim * 2 ** 2, bias=opt.bias),
            UpSample(opt.dim * 2 ** 1, bias=opt.bias)
        ])

        self.fusions = nn.ModuleList([
            nn.Conv2d(
                in_channels = opt.dim * 2 ** 2,
                out_channels = opt.dim * 2 ** 1,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                bias = opt.bias
            ),
            nn.Conv2d(
                in_channels = opt.dim * 2 ** 1,
                out_channels = opt.dim * 2 ** 0,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                bias = opt.bias
            )
        ])

 
        self.stage_interactions = nn.ModuleList([
            StageInteraction(dim = opt.dim * 2 ** 0, act_fn_name=opt.act_fn_name, bias=opt.bias),
            StageInteraction(dim = opt.dim * 2 ** 1, act_fn_name=opt.act_fn_name, bias=opt.bias),
            StageInteraction(dim = opt.dim * 2 ** 2, act_fn_name=opt.act_fn_name, bias=opt.bias),
            StageInteraction(dim = opt.dim * 2 ** 1, act_fn_name=opt.act_fn_name, bias=opt.bias),
            StageInteraction(dim = opt.dim * 2 ** 0, act_fn_name=opt.act_fn_name, bias=opt.bias)
        ])


        self.mapping = nn.Conv2d(opt.dim, opt.out_dim, kernel_size=1, stride=1, padding=0, bias=opt.bias)

    def forward(self, x, enc_outputs=None, bottleneck_out=None, dec_outputs=None
                ,prior=None):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        enc_outputs_l = []
        dec_outputs_l = []
        x1 = self.embedding(x)
        
        res1, log_dict1 = self.Encoder[0](x1,prior=prior[0])
        
        
        if (enc_outputs is not None) and (dec_outputs is not None):
            res1 = self.stage_interactions[0](res1, enc_outputs[0], dec_outputs[0])
        res12 = F.interpolate(res1, scale_factor=0.5, mode='bilinear') 

        x2 = self.Downs[0](res1)
        
        res2, log_dict2 = self.Encoder[1](x2,prior=prior[1])
        
        if (enc_outputs is not None) and (dec_outputs is not None):
            res2 = self.stage_interactions[1](res2, enc_outputs[1], dec_outputs[1])
        res21 = F.interpolate(res2, scale_factor=2, mode='bilinear') 


        x4 = self.Downs[1](res2)
        
        res4, log_dict3 = self.BottleNeck(x4,prior=prior[2])
        
        
        
        
        if bottleneck_out is not None:
            res4 = self.stage_interactions[2](res4, bottleneck_out, bottleneck_out)

        res42 = F.interpolate(res4, scale_factor=2, mode='bilinear') 
        res41 = F.interpolate(res42, scale_factor=2, mode='bilinear') 
       
   
        res1 = self.BlockInteractions[0](res1, res21, res41) 
        res2 = self.BlockInteractions[1](res12, res2, res42) 
        enc_outputs_l.append(res1)
        enc_outputs_l.append(res2)
        
        
        dec_res2 = self.Ups[0](res4) # dim * 2 ** 2 -> dim * 2 ** 1
        dec_res2 = torch.cat([dec_res2, res2], dim=1) # dim * 2 ** 2
        dec_res2 = self.fusions[0](dec_res2) # dim * 2 ** 2 -> dim * 2 ** 1
        
        
        dec_res2, log_dict4 = self.Decoder[0](dec_res2,prior=prior[1])
        
        
        
        
        if (enc_outputs is not None) and (dec_outputs is not None):
            dec_res2 = self.stage_interactions[3](dec_res2, enc_outputs[1], dec_outputs[1])
        
        
        dec_res1 = self.Ups[1](dec_res2) # dim * 2 ** 1 -> dim * 2 ** 0
        dec_res1 = torch.cat([dec_res1, res1], dim=1) # dim * 2 ** 1 
        dec_res1 = self.fusions[1](dec_res1) # dim * 2 ** 1 -> dim * 2 ** 0
        
        
        dec_res1, log_dict5 = self.Decoder[1](dec_res1,prior=prior[0])
        
        
        if (enc_outputs is not None) and (dec_outputs is not None):
            dec_res1 = self.stage_interactions[4](dec_res1, enc_outputs[0], dec_outputs[0])

        dec_outputs_l.append(dec_res1)
        dec_outputs_l.append(dec_res2)

        out = self.mapping(dec_res1) + x

        return out[:, :, :h_inp, :w_inp], enc_outputs_l, res4, dec_outputs_l


class RBP(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.DL = nn.Sequential(
            PWDWPWConv(61, 60, opt.bias, act_fn_name=opt.act_fn_name),
            PWDWPWConv(60, 60, opt.bias, act_fn_name=opt.act_fn_name),
        )

        self.minus = nn.Sequential(
            nn.Conv2d(3,1,1,1,0),
        )
        self.DL_grad = nn.Sequential(
            DWPWConvL(120,60,3,1,1),
            DWPWConv(60, 60, 3,1,1,opt.bias, act_fn_name=opt.act_fn_name),
        )


    def forward(self, y, xk_1, Phi):
        """
        Residual back-projection module.

        Args:
            y: Compressed measurement with shape (B, H, W).
            xk_1: Current spectral estimate with shape (B, C, H, W).
            Phi: Broadband mask with shape (B, C, H, W).
        """
        DL_Phi = self.DL(torch.cat([y.unsqueeze(1), Phi], axis=1))# Fuse measurement and mask.
        Phi = DL_Phi + Phi
        phi = A(xk_1, Phi) # (B, 256, 310) 
        phixs_init = phi - y # Measurement residual.
        phixsy = self.minus(torch.concat([phixs_init.unsqueeze(1),phi.unsqueeze(1),y.unsqueeze(1)],axis=1)).squeeze(1)
        phit = At(phixsy, Phi) # Back-projection to the spectral domain.

        xk_1_phit = torch.concat([xk_1, phit], axis=1) 
        vk = self.DL_grad(xk_1_phit)

        return vk



from .prior_representation import PRM
from .ddpm_denoising_simple_arch import simple_denoise
from .ddpm import DSM,DDPM_Func
from einops.layers.torch import Rearrange

class HyperBME(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.train_phase = opt.train_phase
        use_prior_flag = True
        if opt.test_mode==True:
            self.gt_le = None
        else:
            self.gt_le = PRM()
        
        if opt.train_phase ==2:
            self.net_le_dm = PRM(in_chans=60)
            self.net_d = simple_denoise(64,4,timesteps=opt.timesteps) 
            self.ddpm_func = DDPM_Func()
            self.diffusion = DSM(denoise=self.net_d, 
                                    condition=self.net_le_dm, 
                                    n_feats=64, 
                                    group=4,
                                    linear_start= opt.linear_start,
                                    linear_end= opt.linear_end, 
                                    timesteps = opt.timesteps)
    
        self.head_GD = RBP(opt)
        self.head_PM = MRN(opt,use_prior=use_prior_flag)

        self.body_GD = nn.ModuleList([
            RBP(opt) for _ in range(opt.stage - 2)
        ]) if not opt.body_share_params else RBP(opt)
        self.body_PM = nn.ModuleList([
            MRN(opt,use_prior=use_prior_flag) for _ in range(opt.stage - 2)
        ]) if not opt.body_share_params else MRN(opt,use_prior=use_prior_flag)
        self.tail_GD = RBP(opt)
        self.tail_PM = MRN(opt,use_prior=use_prior_flag)

        group = 4
        embed_dim = 64
        self.down_1 = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear(group*group, (group*group)//4),
            Rearrange('b c n -> b n c'),
            nn.Linear(embed_dim*4, embed_dim*4)
        )
        self.down_2 = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear((group*group)//4, 1),
            Rearrange('b c n -> b n c'),
            nn.Linear(embed_dim*4, embed_dim*4)
        )
    def forward(self, y, Phi,gt=None):

        log_dict = defaultdict(list)
        B, C, H, W = Phi.shape
        x0 = y.unsqueeze(1).repeat((1, C, 1, 1))
        phi_s = torch.sum(Phi,1,keepdim=True)       
        inp_img = (x0/phi_s)*Phi
        
        if gt is not None:
            prior_z = self.gt_le(inp_img,gt)
            
        else:
            prior_z = None

        if self.train_phase==1:
            prior = prior_z
        elif self.train_phase==2:
            prior, _=self.diffusion(inp_img,prior_z)
            log_dict['prior'] = prior
            log_dict['prior_z'] = prior_z
            
        if self.train_phase>0:
            prior_att = []
            prior_1 = prior # [2, 16, 256]
            prior_2 = self.down_1(prior_1)# [2, 4, 256]
            prior_3 = self.down_2(prior_2) # [2, 256]
            prior_att.append(prior_1)
            prior_att.append(prior_2) 
            prior_att.append(prior_3) #112 64 64
        elif self.train_phase==0:
            prior_att = [None,None,None]
    
        v_pre = self.head_GD(y, x0, Phi)
        
        x_pre, enc_outputs, bottolenect_out, dec_outputs = self.head_PM(v_pre,prior=prior_att,)
        log_dict['stage0_x'] = x_pre
        
        
       

        for i in range(self.opt.stage-2):
            v_pre = self.body_GD[i](y, x_pre, Phi) if not self.opt.body_share_params else self.body_GD(y, x_pre, Phi)
            
   
            x_pre, enc_outputs, bottolenect_out, dec_outputs = \
                self.body_PM[i](v_pre, enc_outputs, bottolenect_out, dec_outputs,prior=prior_att,) if not self.opt.body_share_params else self.body_PM(v_pre, enc_outputs, bottolenect_out, dec_outputs,prior=prior_att,)
            log_dict[f'stage{i+1}_x'] = x_pre
            
            
        
        v_pre = self.tail_GD(y, x_pre, Phi)
        

        out, enc_outputs, bottolenect_out, dec_outputs = self.tail_PM(v_pre, 
                                                                      enc_outputs, bottolenect_out, dec_outputs,prior=prior_att,)

    
        out = out[:, :, :, :]
        log_dict = dict(log_dict)
        return out, log_dict