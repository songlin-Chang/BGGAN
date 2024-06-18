import torch
import torch.nn.functional as F
from collections import OrderedDict
from torch import nn, Tensor
import torchvision
from lib.utils import get_GTI_in_out_chs
from lib.utils import get_GIT_in_out_chs
from typing import Tuple
from torch.nn import MultiheadAttention
from copy import deepcopy
from typing import Tuple

import torch
from torch import nn, Tensor
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        param:
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        self.d_model = d_model

        # create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # not a parameter, but should be part of the modules state.
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1)]
        return x




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
    # 512,8,2048
    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int,
                 dropout: float):
        super(DecoderLayer, self).__init__()
        """
        param:
        d_model:    features size.
                    int

        num_heads:  number of heads in the multiheadattention model.
                    int

        dropout:    dropout value
                    float
        """
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
        # self attention + resedual summation + norm
        output, _ = self.dec_self_attn(dec_inputs,
                                       dec_inputs,
                                       dec_inputs,
                                       attn_mask=tgt_mask,
                                       key_padding_mask=tgt_pad_mask)
        output = dec_inputs + self.self_attn_dropout(output)
        output = self.self_attn_norm(output)  # type: Tensor

        # # self attention + residual + norm + FF
        output2, attns = self.multihead_attn(output, enc_outputs, enc_outputs)
        output = output + self.multihead_dropout(output2)
        output = self.multihead_norm(output)

        output2 = self.ff(output)  # type: Tensor
        output = self.ff_norm(output + self.ff_dropout(output2))

        return output, attns


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, num_layers: int):
        super().__init__()
        # Make copies of the encoder layer
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)])

    def forward(self, x: Tensor) -> Tensor:


        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):


    def __init__(self,
                 layer: DecoderLayer,
                 vocab_size: int,
                 d_model: int,
                 num_layers: int,
                 max_len: int,
                 dropout: float,
                 pad_id: int):
        super().__init__()

        self.pad_id = pad_id

        # Embedding layer + pos encoding
        self.cptn_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = PositionalEncoding(d_model, max_len)

        # Make copies of the decoder layer
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)])

        self.dropout = nn.Dropout(p=dropout)

    def get_attn_subsequent_mask(self, sz: int) -> Tensor:

        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, tgt_cptn: Tensor,
                src_img: Tensor) -> Tuple[Tensor, Tensor]:


        # create masks, then pass to decoder
        tgt_pad_mask = (tgt_cptn == self.pad_id) #2x192

        tgt_mask = self.get_attn_subsequent_mask(tgt_cptn.size()[1])
        tgt_mask = tgt_mask.to(tgt_cptn.device) # [192,192]

        # encode captions + pos enc
        # (B, max_len) -> (B, max_len, d_model) -> (max_len, B, d_model)
        tgt_cptn = self.cptn_emb(tgt_cptn)  # type: Tensor
        tgt_cptn = self.dropout(self.pos_emb(tgt_cptn.permute(1, 0, 2))) # 192,2,512
        attns_all = []
        for layer in self.layers:
            tgt_cptn, attns = layer(tgt_cptn, src_img, tgt_mask, tgt_pad_mask)

            attns_all.append(attns)
        # [layer_num, batch_size, head_num, max_len, encode_size**2]
        attns_all = torch.stack(attns_all)


        return tgt_cptn, attns_all


class Transformer(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 img_encode_size: int,
                 enc_ff_dim: int,
                 dec_ff_dim: int,
                 enc_n_layers: int,
                 dec_n_layers: int,
                 enc_n_heads: int,
                 dec_n_heads: int,
                 max_len: int,
                 dropout: float = 0.1,
                 pad_id: int = 0):
        super(Transformer, self).__init__()
        encoder_layer = EncoderLayer(img_encode_size=img_encode_size,
                                     img_embed_dim=d_model,
                                     feedforward_dim=enc_ff_dim,
                                     num_heads=enc_n_heads,
                                     dropout=dropout)
        decoder_layer = DecoderLayer(d_model=d_model,
                                     num_heads=dec_n_heads,
                                     feedforward_dim=dec_ff_dim,
                                     dropout=dropout)
        self.encoder = Encoder(layer=encoder_layer, num_layers=enc_n_layers)
        self.decoder = Decoder(layer=decoder_layer,
                               vocab_size=vocab_size,
                               d_model=d_model,
                               num_layers=dec_n_layers,
                               max_len=max_len,
                               dropout=dropout,
                               pad_id=pad_id)

        self.predictor = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, images: Tensor,
                captions: Tensor) -> Tuple[Tensor, Tensor]:
        # encode, decode, predict
        images_encoded = self.encoder(images.permute(1, 0, 2))  # type: Tensor
        tgt_cptn, attns = self.decoder(captions, images_encoded)
        predictions = self.predictor(tgt_cptn).permute(1, 0, 2)  # type: Tensor

        return predictions.contiguous(), attns.contiguous()




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
        self.conv = nn.Conv2d(256, 2048, kernel_size=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.adaptive_resize = nn.AdaptiveAvgPool2d(encode_size)
        # Softmax to compute attention weights
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images: Tensor, image_encoder, netD):
        """
        param:
        images: Input images.
                Tensor [batch_size, 3, h, w]

        output: encoded images.
                Tensor [batch_size, encode_size * encode_size, embed_dim]
        """
        # batch_size = B
        # image_size = [B, 3, h, w]
        B = images.size()[0]
        out = self.resnet(images)
        # [B, 3, h, w] -> [B, 2048, h/32=8, w/32=8]
        # out = self.resnet(images)  # type: Tensor
        with torch.no_grad():
            feature = netD(images)
            feature1 = image_encoder(images)
        # Downsampling: resnet features size (2048) -> embed_size (512)
        # [B, 2048, 8, 8] -> [B, embed_size=512, 8, 8]
        out = self.relu(self.bn(self.downsampling(out)))
        outD = self.adaptive_pool(self.conv(feature))
        outI = self.adaptive_pool(feature1)
        outD = self.relu(self.bn(self.downsampling(outD)))
        outI = self.relu(self.bn(self.downsampling(outI)))

        out_a = (outD + outI) / 2.0
        out_f = (out + out_a) / 2.0
        out_f = (out + out_a) / 2.0
        # Adaptive image resize: resnet output size (8,8) -> encode_size (14,14)
        #   [B, embed_size=512, 8, 8] ->
        #       [B, embed_size=512, encode_size=14, encode_size=14] ->
        #           [B, 512, 196] -> [B, 196, 512]
        out = self.adaptive_resize(out_f)
        # out = self.adaptive_resize(out)
        out = out.view(B, self.embed_dim, -1).permute(0, 2, 1)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the tuning for blocks 2 through 4.
        """

        for p in self.resnet.parameters():
            p.requires_grad = False

        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class NetGTI(nn.Module):
                         #32，100，256,256,3
    def __init__(self, ngf, nz, cond_dim, imsize, ch_size):
        super(NetGTI, self).__init__()
        self.ngf = ngf
        # input noise (batch_size, 100)
        self.fc = nn.Linear(nz, ngf*8*4*4)
        self.GBlocks = nn.ModuleList([])
        in_out_pairs = get_GTI_in_out_chs(ngf, imsize)
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            self.GBlocks.append(G_Block(cond_dim+nz, in_ch, out_ch, upsample=True))
        # to RGB image
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(out_ch, ch_size, 3, 1, 1),
            nn.Tanh(),
            )

    def forward(self, noise, c): # x=noise, c=sent_emb
        # concat noise and sentence
        #out.shape batch_size X 4096
        out = self.fc(noise)
        #reshape batch_size 256 ,4 ,4
        out = out.view(noise.size(0), 8*self.ngf, 4, 4)
        cond = torch.cat((noise, c), dim=1)
        # fuse text and visual features
        for GBlock in self.GBlocks:
            out = GBlock(out, cond)
        # convert to RGB image
        out = self.to_rgb(out)
        return out

class NetGIT(nn.Module):
    def __init__(self, ndf, imsize=128, ch_size=3):
        super(NetGIT, self).__init__()
        self.conv_img = nn.Conv2d(ch_size, ndf, 3, 1, 1)
        # build DBlocks
        self.DBlocks = nn.ModuleList([])
        in_out_pairs = get_GIT_in_out_chs(ndf, imsize)
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            self.DBlocks.append(D_Block(in_ch, out_ch))

    def forward(self,x):
        out = self.conv_img(x)
        for DBlock in self.DBlocks:
            out = DBlock(out)
        return out


class NetD(nn.Module):
    def __init__(self, ndf, cond_dim=256):
        super(NetD, self).__init__()
        self.cond_dim = cond_dim
        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf*8+cond_dim, ndf*2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*2, 1, 4, 1, 0, bias=False)
        )
    def forward(self, out, y):
        y = y.view(-1, self.cond_dim, 1, 1)
        y = y.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((out, y), 1)
        out = self.joint_conv(h_c_code)
        return out 


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





class G_Block(nn.Module):
    def __init__(self, cond_dim, in_ch, out_ch, upsample):
        super(G_Block, self).__init__()
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


class D_Block(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super(D_Block, self).__init__()
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

    def forward(self, x):
        res = self.conv_r(x)
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)        
        return x + self.gamma*res


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



