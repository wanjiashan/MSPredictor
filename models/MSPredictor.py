import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.MSGBlock import GraphBlock, Attention_Block
from models.KAN import KANLinear


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class ScaleGraphBlock(nn.Module):
    def __init__(self, configs):
        super(ScaleGraphBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        self.att0 = Attention_Block(configs.d_model, configs.d_ff,
                                    n_heads=configs.n_heads, dropout=configs.dropout, activation="gelu")
        self.norm = nn.LayerNorm(configs.d_model)
        self.gelu = nn.GELU()
        self.gconv = nn.ModuleList([
            GraphBlock(configs.c_out, configs.d_model, configs.conv_channel, configs.skip_channel,
                       configs.gcn_depth, configs.dropout, configs.propalpha, configs.seq_len,
                       configs.node_dim)
            for _ in range(self.k)
        ])
        self.kan_layer = KANLinear(
            in_features=configs.d_model,
            out_features=configs.d_model,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0
        )

    def forward(self, x):
        B, T, N = x.size()
        scale_list, scale_weight = FFT_for_Period(x, self.k)
        res = []

        for i in range(self.k):
            scale = scale_list[i]
            x_gconv = self.gconv[i](x)

            if (self.seq_len) % scale != 0:
                length = (((self.seq_len) // scale) + 1) * scale
                padding = torch.zeros([x.shape[0], (length - self.seq_len), x.shape[2]]).to(x.device)
                out = torch.cat([x_gconv, padding], dim=1)
            else:
                length = self.seq_len
                out = x_gconv

            out = out.reshape(B, length // scale, scale, N)
            out = out.reshape(-1, scale, N)
            out = self.norm(self.att0(out))
            out = self.gelu(out)

            # print(f"Before KAN, out shape: {out.shape}")
            out = self.kan_layer(out)
            # print(f"After KAN, out shape: {out.shape}")

            out = out.reshape(B, -1, scale, N).reshape(B, -1, N)
            res.append(out[:, :self.seq_len, :])

        res = torch.stack(res, dim=-1)
        scale_weight = F.softmax(scale_weight, dim=1).unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * scale_weight, -1)

        return res + x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len


        self.model = nn.ModuleList([ScaleGraphBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)


        self.projection = KANLinear(
            in_features=configs.d_model,
            out_features=configs.c_out,
            grid_size=5,
            spline_order=3
        )
        # 将整个时间轴的特征展平，再映射为 pred_len * c_out 的输出
        self.seq2pred = KANLinear(
            in_features=configs.seq_len * configs.c_out,
            out_features=configs.pred_len * configs.c_out,
            grid_size=5,
            spline_order=3
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        means = x_enc.mean(1, keepdim=True).detach()  # (B, 1, enc_in)
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True) + 1e-5)  # (B, 1, enc_in)
        x_enc = (x_enc - means) / std_enc

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        for layer in self.model:
            enc_out = layer(enc_out)
        # enc_out 的形状为 (B, seq_len, d_model)

        proj_out = self.projection(enc_out)  # 形状 (B, seq_len, c_out)
        # print(f"proj_out shape: {proj_out.shape}")
        B, T, C = proj_out.shape  # T 应等于 seq_len，C = c_out
        proj_out_flat = proj_out.reshape(B, -1)  # 形状 (B, seq_len * c_out)


        dec_out = self.seq2pred(proj_out_flat)  # 形状 (B, pred_len * c_out)
        dec_out = dec_out.view(B, self.pred_len, C)
        # print(f"dec_out shape after seq2pred: {dec_out.shape}")
        dec_out = dec_out * std_enc + means
        return dec_out
