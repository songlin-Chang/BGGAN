import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from lib.utils import get_GTI_in_out_chs
from collections import OrderedDict
from typing import Tuple
from torch import nn, Tensor
import torchvision
from torch.nn import MultiheadAttention



class echoCLIP_IMG_ENCODER(nn.Module):
    def __init__(self, echo_clip):
        super(echoCLIP_IMG_ENCODER, self).__init__()
        model = echo_clip.visual
        self.define_module(model)
        self.dtype = self.conv1.weight.dtype
        for param in self.parameters():
            param.requires_grad = False

    def define_module(self, model):
        self.conv1 = model.conv1
        self.class_embedding = model.class_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_pre = model.ln_pre
        self.transformer = model.transformer
        self.ln_post = model.ln_post
        self.proj = model.proj
        

    def transf_to_CLIP_input(self,inputs):
        device = inputs.device
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])\
            .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
        var = torch.tensor([0.26862954, 0.26130258, 0.27577711])\
            .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
        inputs = F.interpolate(inputs*0.5+0.5, size=(224, 224))
        inputs = ((inputs+1)*0.5-mean)/var
        return inputs

    def forward(self, img: torch.Tensor):
        x = self.transf_to_CLIP_input(img)
        x = x.type(self.dtype)
        x = self.conv1(x) 
        grid =  x.size(-1)
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1)  
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) 
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        selected = [1,4,8]
        local_features = []
        for i in range(12):
            x = self.transformer.resblocks[i](x)
            if i in selected:
                local_features.append(x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(-1, 768, grid, grid).contiguous().type(img.dtype))
        x = x.permute(1, 0, 2) 
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return torch.stack(local_features, dim=1), x.type(img.dtype)


class NetGTI(nn.Module):
    def __init__(self, ngf, nz, cond_dim, imsize, ch_size):
        super(NetGTI, self).__init__()
        self.ngf = ngf
        self.fc = nn.Linear(nz, ngf*8*4*4)
        self.GIBlocks = nn.ModuleList([])
        in_out_pairs = get_GTI_in_out_chs(ngf, imsize)
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            self.GIBlocks.append(GI_Block(cond_dim+nz, in_ch, out_ch, upsample=True))
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(out_ch, ch_size, 3, 1, 1),
            nn.Tanh(),
            )

    def forward(self, noise, c): 
        out = self.fc(noise)
        out = out.view(noise.size(0), 8*self.ngf, 4, 4)
        cond = torch.cat((noise, c), dim=1)
        for GIBlock in self.GIBlocks:
            out = GIBlock(out, cond)
        out = self.to_rgb(out)
        return out


class GI_Block(nn.Module):
    def __init__(self, cond_dim, in_ch, out_ch, upsample):
        super(GI_Block, self).__init__()
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch 
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.fuse1 = DFBLK(cond_dim, in_ch)
        self.fuse2 = DFBLK(cond_dim, out_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, y):
        h = self.fuse1(h, y)
        h = self.c1(h)
        h = self.fuse2(h, y)
        h = self.c2(h)
        return h

    def forward(self, x, y):
        if self.upsample==True:
            x = F.interpolate(x, scale_factor=2)
        return self.shortcut(x) + self.residual(x, y)

class NetGIT(nn.Module):
    def __init__(self, ndf):
        super(NetGIT, self).__init__()
        self.conv_img = nn.Conv2d(3, ndf, 3, 1, 1)
        self.block0 = resGIT(ndf * 1, ndf * 2)
        self.block1 = resGIT(ndf * 2, ndf * 4)
        self.block2 = resGIT(ndf * 4, ndf * 8)
        self.block3 = resGIT(ndf * 8, ndf * 16)
        self.block4 = resGIT(ndf * 16, ndf * 16)
        self.block5 = resGIT(ndf * 16, ndf * 16)


    def forward(self,x):
        out = self.conv_img(x)
        out = self.block0(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        return out




class resGIT(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_s = nn.Conv2d(fin,fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, c=None):
        return self.shortcut(x)+self.gamma*self.residual(x)

    def shortcut(self, x):
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        return self.conv_r(x)


class D_GET_SOCRE(nn.Module):
    def __init__(self, ndf):
        super(D_GET_SOCRE, self).__init__()
        self.df_dim = ndf
        self.fc = nn.Linear(512,256)
        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf * 16+256, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )

    def forward(self, out, y):
       
        y = y.view(-1, 256, 1, 1)
        y = y.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((out, y), 1)
        out = self.joint_conv(h_c_code)

        return out







class DFBLK(nn.Module):
    def __init__(self, cond_dim, in_ch):
        super(DFBLK, self).__init__()
        self.affine0 = Affine(cond_dim, in_ch)
        self.affine1 = Affine(cond_dim, in_ch)

    def forward(self, x, y=None):
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        return h


class ImageEncoder(nn.Module):

    def __init__(self, encode_size=14, embed_dim=512):
        super(ImageEncoder, self).__init__()

        self.embed_dim = embed_dim
        resnet = torchvision.models.resnet101(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        self.downsampling = nn.Conv2d(in_channels=2048,
                                      out_channels=embed_dim,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(512, 2048, kernel_size=1)
        self.conv1 = nn.Conv2d(512, 2048, kernel_size=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.adaptive_resize = nn.AdaptiveAvgPool2d(encode_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images: Tensor, image_encoder, netGIT):
        # batch_size = B
        # image_size = [B, 3, h, w]
        B = images.size()[0]
        out = self.resnet(images)
        # [B, 3, h, w] -> [B, 2048, h/32=8, w/32=8]
        # out = self.resnet(images)  # type: Tensor
        with torch.no_grad():
            feature = netGIT(images)
            
            # images = images.bfloat16()
            feature1 = image_encoder(images)
            # feature1 = F.normalize(image_encoder.encode_image(images), dim=-1)
            # feature1 = feature1.unsqueeze(0)
            # feature1 = feature1.float()
            # shapef = feature1.shape
            # if shapef[1] == 16:
            #     feature1 = feature1.reshape(16, 512, 1, 1)
            # elif shapef[1] ==1 :
            #     feature1 = feature1.reshape(1, 512, 1, 1)
            # else:
            #     feature1 = feature1.reshape(6, 512, 1, 1)

        # Downsampling: resnet features size (2048) -> embed_size (512)
        # [B, 2048, 8, 8] -> [B, embed_size=512, 8, 8]
        out = self.relu(self.bn(self.downsampling(out)))
        outD = self.adaptive_pool(self.conv(feature))
        # outI = self.adaptive_pool(self.conv1(feature1))
        outI = feature1
        outD = self.relu(self.bn(self.downsampling(outD)))
        outI = self.relu(self.bn(self.downsampling(outI)))

        out_a = (outD+outI)/2.0
        out_f = (out+out_a)/2.0
        # out_f = (out+out_a)/2.0
        # Adaptive image resize: resnet output size (8,8) -> encode_size (14,14)
        #   [B, embed_size=512, 8, 8] ->
        #       [B, embed_size=512, encode_size=14, encode_size=14] ->
        #           [B, 512, 196] -> [B, 196, 512]
        out = self.adaptive_resize(out_f)
        # out = self.adaptive_resize(out)
        out = out.view(B, self.embed_dim, -1).permute(0, 2, 1)
        return out

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False

        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class Affine(nn.Module):
    def __init__(self, cond_dim, num_features):
        super(Affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)        

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias

class CNNFeedForward(nn.Module):
    def __init__(self, encode_size: int, embed_dim: int, feedforward_dim: int,
                 dropout: float):
        super(CNNFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=encode_size,
                               out_channels=feedforward_dim,
                               kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=feedforward_dim,
                               out_channels=encode_size,
                               kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.conv2(self.relu(self.conv1(inputs.permute(1, 0, 2))))
        output = self.dropout(output)  # type: Tensor
        return self.layer_norm(output.permute(1, 0, 2) + inputs)


class EncSelfAttension(nn.Module):

    def __init__(self, img_embed_dim: int, num_heads: int, dropout: float):
        super(EncSelfAttension, self).__init__()
        self.multi_head_attn = MultiheadAttention(embed_dim=img_embed_dim,
                                                  num_heads=num_heads,
                                                  dropout=dropout)
        self.layer_norm = nn.LayerNorm(img_embed_dim)

    def forward(self, enc_inputs: Tensor) -> Tensor:
        enc_outputs, _ = self.multi_head_attn(enc_inputs, enc_inputs,
                                              enc_inputs)
        enc_outputs = enc_outputs + enc_inputs
        enc_outputs = self.layer_norm(enc_outputs)

        return enc_outputs


class EncoderLayer(nn.Module):

    def __init__(self, img_encode_size: int, img_embed_dim: int,
                 feedforward_dim: int, num_heads: int, dropout: float):
        super(EncoderLayer, self).__init__()

        self.enc_self_attn = EncSelfAttension(img_embed_dim=img_embed_dim,
                                              num_heads=num_heads,
                                              dropout=dropout)
        self.cnn_ff = CNNFeedForward(encode_size=img_encode_size,
                                     embed_dim=img_embed_dim,
                                     feedforward_dim=feedforward_dim,
                                     dropout=dropout)

    def forward(self, enc_inputs: Tensor) -> Tensor:
        enc_outputs = self.enc_self_attn(enc_inputs)
        enc_outputs = self.cnn_ff(enc_outputs)
        return enc_outputs

class DecoderLayer(nn.Module):
    #512,8,2048
    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int,
                 dropout: float):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiheadAttention(d_model,
                                                num_heads,
                                                dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model,
                                                 num_heads,
                                                 dropout=dropout)

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.multihead_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.multihead_dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                nn.ReLU(inplace=True), nn.Dropout(p=dropout),
                                nn.Linear(feedforward_dim, d_model))

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, dec_inputs: Tensor, enc_outputs: Tensor,
                tgt_mask: Tensor,
                tgt_pad_mask: Tensor) -> Tuple[Tensor, Tensor]:
        output, _ = self.dec_self_attn(dec_inputs,
                                       dec_inputs,
                                       dec_inputs,
                                       attn_mask=tgt_mask,
                                       key_padding_mask=tgt_pad_mask)
        output = dec_inputs + self.self_attn_dropout(output)
        output = self.self_attn_norm(output)
        output2, attns = self.multihead_attn(output, enc_outputs, enc_outputs)
        output = output + self.multihead_dropout(output2)
        output = self.multihead_norm(output)
        output2 = self.ff(output)
        output = self.ff_norm(output + self.ff_dropout(output2))
        return output, attns
