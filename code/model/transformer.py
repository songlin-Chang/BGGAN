from copy import deepcopy
from typing import Tuple
import math
import torch
from torch import nn, Tensor

from models.model import EncoderLayer
from models.model import DecoderLayer



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
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=dropout)

    def get_attn_subsequent_mask(self, sz: int) -> Tensor:

        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, tgt_cptn: Tensor,
                src_img: Tensor) -> Tuple[Tensor, Tensor]:

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
        images_encoded = self.encoder(images.permute(1, 0, 2))

        tgt_cptn, attns = self.decoder(captions, images_encoded)
        predictions = self.predictor(tgt_cptn).permute(1, 0, 2)

        return predictions.contiguous(), attns.contiguous()


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
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
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1)]
        return x