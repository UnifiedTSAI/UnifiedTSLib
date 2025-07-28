import math
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.fft
import torch.nn as nn
from torch.nn import Softmax

from transformers.utils import ModelOutput
from transformers import PreTrainedModel

from .configuration_timemixerpp import TimeMixerppConfig

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class RowAttention(nn.Module):
    def __init__(self, in_dim, q_k_dim):
        super(RowAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        b, _, h, w = x.size()
        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)

        Q = Q.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)
        K = K.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
        V = V.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        row_attn = torch.bmm(Q, K)
        row_attn = self.softmax(row_attn)
        out = torch.bmm(V, row_attn.permute(0, 2, 1))
        out = out.view(b, h, -1, w).permute(0, 2, 1, 3)
        out = self.gamma * out + x
        return out


class ColAttention(nn.Module):
    def __init__(self, in_dim, q_k_dim):
        super(ColAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, _, h, w = x.size()
        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)

        Q = Q.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)  # size = (b*w,h,c2)
        K = K.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # size = (b*w,c2,h)
        V = V.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # size = (b*w,c1,h)

        col_attn = torch.bmm(Q, K)
        col_attn = self.softmax(col_attn)
        out = torch.bmm(V, col_attn.permute(0, 2, 1))
        out = out.view(b, w, -1, h).permute(0, 2, 3, 1)
        out = self.gamma * out + x
        return out



class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.stride = stride
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i, stride=stride))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Trans_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_kernels=6, init_weight=True):
        super(Inception_Trans_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.stride = stride

        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i, stride=stride))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, output_size):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x, output_size=output_size))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res




class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)




class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)



class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x





def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    if len(frequency_list) < k:
        k = len(frequency_list)
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    index = np.where(period > 0)
    top_list = top_list[index]
    period = period[period > 0]
    return period, abs(xf).mean(-1)[:, top_list], top_list


class MultiScaleSeasonCross(nn.Module):
    def __init__(self, configs):
        super(MultiScaleSeasonCross, self).__init__()
        self.cross_conv_season = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels, stride=(configs.down_sampling_window, 1)),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels))

    def forward(self, season_list):
        B, N, _, _ = season_list[0].size()
        # cross high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = []
        out_season_list.append(out_high.permute(0, 2, 3, 1).reshape(B, -1, N))
        for i in range(len(season_list) - 1):
            out_low_res = self.cross_conv_season(out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 3, 1).reshape(B, -1, N))
        return out_season_list


class MultiScaleTrendCross(nn.Module):
    def __init__(self, configs):
        super(MultiScaleTrendCross, self).__init__()

        self.cross_trans_conv_season = Inception_Trans_Block_V1(configs.d_model, configs.d_ff,
                                                                num_kernels=configs.num_kernels,
                                                                stride=(configs.down_sampling_window, 1))
        self.cross_trans_conv_season_restore = nn.Sequential(
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, trend_list):
        B, N, _, _ = trend_list[0].size()
        # cross low->high
        trend_list.reverse()
        out_low = trend_list[0]
        out_high = trend_list[1]
        out_trend_list = []
        out_trend_list.append(out_low.permute(0, 2, 3, 1).reshape(B, -1, N))

        for i in range(len(trend_list) - 1):
            out_high_res = self.cross_trans_conv_season(out_low, output_size=out_high.size())
            out_high_res = self.cross_trans_conv_season_restore(out_high_res)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list) - 1:
                out_high = trend_list[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 3, 1).reshape(B, -1, N))

        out_trend_list.reverse()
        return out_trend_list


class MixerBlock(nn.Module):
    def __init__(self, configs):
        super(MixerBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.down_sampling_window = configs.down_sampling_window
        self.k = configs.top_k
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.row_attn_2d_trend = RowAttention(configs.d_model, configs.d_ff)
        self.col_attn_2d_season = ColAttention(configs.d_model, configs.d_ff)
        self.multi_scale_season_conv = MultiScaleSeasonCross(configs)
        self.multi_scale_trend_conv = MultiScaleTrendCross(configs)

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)
        period_list, period_weight, top_list = FFT_for_Period(x_list[-1], self.k)

        res_list = []
        for i in range(len(period_list)):
            period = period_list[i]
            season_list = []
            trend_list = []
            for x in x_list:
                out = self.time_imaging(x, period)
                season, trend = self.dual_axis_attn(out)
                season_list.append(season)
                trend_list.append(trend)

            out_list = self.multi_scale_mixing(season_list, trend_list, length_list)
            res_list.append(out_list)

        res_list_new = []
        for i in range(len(x_list)):
            list = []
            for j in range(len(period_list)):
                list.append(res_list[j][i])
            res = torch.stack(list, dim=-1)
            res_list_new.append(res)

        res_list_agg = []
        for x, res in zip(x_list, res_list_new):
            res = self.multi_reso_mixing(period_weight, x, res)
            res = self.layer_norm(res)
            res_list_agg.append(res)
        return res_list_agg

    def time_imaging(self, x, period):
        B, T, N = x.size()
        out, length = self.__conv_padding(x, period)
        out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
        return out

    def dual_axis_attn(self, out):
        trend = self.row_attn_2d_trend(out)
        season = self.col_attn_2d_season(out)
        return season, trend

    def multi_scale_mixing(self, season_list, trend_list, length_list):
        out_season_list = self.multi_scale_season_conv(season_list)
        out_trend_list = self.multi_scale_trend_conv(trend_list)
        out_list = []
        for out_season, out_trend, length in zip(out_season_list, out_trend_list, length_list):
            out = out_season + out_trend
            out_list.append(out[:, :length, :])
        return out_list

    def __conv_padding(self, x, period, down_sampling_window=1):
        B, T, N = x.size()

        if T % (period * down_sampling_window) != 0:
            length = ((T // (period * down_sampling_window)) + 1) * period * down_sampling_window
            padding = torch.zeros([B, (length - T), N]).to(x.device)
            out = torch.cat([x, padding], dim=1)
        else:
            length = T
            out = x
        return out, length

    def multi_reso_mixing(self, period_weight, x, res):
        B, T, N = x.size()
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimeMixerppPreTrainedModel(PreTrainedModel):
    config_class = TimeMixerppConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # _no_split_modules = ["TimeMoeDecoderLayer"]
    # _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class TimeMixerppModel(TimeMixerppPreTrainedModel):
    config_class = TimeMixerppConfig
    def __init__(self, configs:TimeMixerppConfig):
        super().__init__(configs)
        self.configs = configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.encoder_model = nn.ModuleList([MixerBlock(configs)
                                            for _ in range(configs.e_layers)])

        if self.channel_independence:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(self.configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        self.revin_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in,
                          affine=False, subtract_last=False
                          )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        self.layer = configs.e_layers

        if self.configs.channel_mixing:
            d_time_model = configs.seq_len // (configs.down_sampling_window ** configs.down_sampling_layers)
            self.channel_mixing_attention = AttentionLayer(FullAttention(False, attention_dropout=self.configs.dropout,
                                                                         output_attention=self.configs.output_attention),
                                                           d_time_model, self.configs.n_heads)

        if self.configs.down_sampling_method == 'max':
            self.down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            self.down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            if self.channel_independence:
                in_channels = 1
            else:
                in_channels = self.configs.enc_in
            self.down_pool = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                       kernel_size=3, padding=padding,
                                       stride=self.configs.down_sampling_window,
                                       padding_mode='circular',
                                       bias=False)
        else:
            raise ValueError('Downsampling method is error,only supporting the max, avg, conv1D')


        self.predict_layer = nn.Linear(self.seq_len, max(configs.pred_len))

        self.projection_layer = nn.Linear(
            configs.d_model,
            1 if configs.channel_independence else configs.c_out,
            bias=True
        )
        # Initialize weights
        self.post_init()

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        # B,T,C -> B,C,T
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc
        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            if self.configs.down_sampling_method == 'conv' and i == 0 and self.channel_independence:
                x_enc_ori = x_enc_ori.contiguous().reshape(B * N, T, 1).permute(0, 2, 1).contiguous()
            x_enc_sampling = self.down_pool(x_enc_ori)

            if self.configs.down_sampling_method == 'conv':
                x_enc_sampling_list.append(
                    x_enc_sampling.reshape(B, N, T // (self.down_sampling_window ** (i + 1))).permute(0, 2, 1))
            else:
                x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))

            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forward(self,  input_ids: torch.FloatTensor = None,x_mark_enc: torch.FloatTensor = None):
        x_enc = input_ids
        B, T, N = x_enc.size()
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc=x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()

                x = self.revin_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.revin_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        if self.configs.channel_mixing and self.channel_independence == 1:
            _, T, D = x_list[-1].size()

            coarse_scale_enc_out = x_list[-1].reshape(B, N, T * D)
            coarse_scale_enc_out, _ = self.channel_mixing_attention(coarse_scale_enc_out, coarse_scale_enc_out,
                                                                    coarse_scale_enc_out, None)
            x_list[-1] = coarse_scale_enc_out.reshape(B * N, T, D) + x_list[-1]

        enc_out_list = []
        if x_mark_enc is not None:
            for x, x_mark in zip(x_list, x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for x in x_list:
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        for i in range(self.layer):
            enc_out_list = self.encoder_model[i](enc_out_list)

        enc_out = enc_out_list[0]
        preds = []

     
        dec_out = self.predict_layer(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

    
        dec_out = self.projection_layer(dec_out)
        dec_out = dec_out.reshape(B, N, -1).permute(0, 2, 1).contiguous()

        pred = self.revin_layers[0](dec_out, 'denorm')
        for i in range(len(self.pred_len)):
            preds.append(pred[:,:self.pred_len[i],:])

        return preds



@dataclass
class TimeMixerppPredictionOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    # hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class TimeMixerppForPrediction(TimeMixerppPreTrainedModel):
    def __init__(self, configs):

        super().__init__(configs)
        self.criterion = nn.MSELoss(reduction='none')
        self.model = TimeMixerppModel(configs)
        self.pred_len = configs.pred_len


    def forward(
            self,
            input_ids: torch.FloatTensor = None,
            labels: Optional[torch.FloatTensor] = None,
            loss_masks: Optional[torch.FloatTensor] = None,
            prediction_length: Optional[torch.IntTensor] = None,
            context_length: Optional[torch.IntTensor] = None,
            inner_batchsize: Optional[torch.IntTensor] = None,
            num_channel: Optional[torch.IntTensor] = None,
            dataset_idx: Optional[torch.IntTensor] = None
    ) -> Union[Tuple, TimeMixerppPredictionOutput]:

        dataset_identifier = False
        if dataset_idx is not None:

            dataset_identifier = (dataset_idx == dataset_idx[0]).all()


        if inner_batchsize is None:
            if len(input_ids.shape)==2:
                input_ids = input_ids.unsqueeze(-1)
                labels = labels.unsqueeze(-1)
            preds = self.model(input_ids=input_ids)
            loss = self.calc_loss(preds, labels,loss_masks)

            return TimeMixerppPredictionOutput(loss=loss, logits=preds)
        elif not dataset_identifier:
            total_loss = 0
            for batch_idx in range(input_ids.size(0)):
                flatten_seq_len = inner_batchsize[batch_idx]*(context_length[batch_idx]+prediction_length[batch_idx])*num_channel[batch_idx]
                input_batch = input_ids[batch_idx][:flatten_seq_len]
                input_batch = input_batch.reshape(inner_batchsize[batch_idx],num_channel[batch_idx],context_length[batch_idx]+prediction_length[batch_idx]).permute(0,2,1)
                input_context = input_batch[:,:context_length[batch_idx],:]
                input_labels = input_batch[:,context_length[batch_idx]:,:]
                preds = self.model(input_ids=input_context)
                loss = self.calc_loss(preds, input_labels, loss_masks)
                total_loss += loss

            total_loss = total_loss / input_ids.size(0)
            return TimeMixerppPredictionOutput(loss=total_loss)
        else:
            input_contexts = []
            input_labels = []
            for batch_idx in range(input_ids.size(0)):
                flatten_seq_len = inner_batchsize[batch_idx]*(context_length[batch_idx]+prediction_length[batch_idx])*num_channel[batch_idx]
                input_batch = input_ids[batch_idx][:flatten_seq_len]
                input_batch = input_batch.reshape(inner_batchsize[batch_idx],num_channel[batch_idx],context_length[batch_idx]+prediction_length[batch_idx]).permute(0,2,1)
                input_context = input_batch[:,:context_length[batch_idx],:]
                input_label = input_batch[:,context_length[batch_idx]:,:]
                input_contexts.append(input_context)
                input_labels.append(input_label)
         
            input_context = torch.cat(input_contexts, dim=0)
            input_label = torch.cat(input_labels, dim=0)
            preds = self.model(input_ids=input_context)
            loss = self.calc_loss(preds, input_label, loss_masks)
            return TimeMixerppPredictionOutput(loss=loss)



    def calc_loss(self, preds,trues,loss_masks):

        if not isinstance(preds, list):
            preds = [preds]

        total_loss = 0
        for idx, pred in enumerate(preds):
            losses = self.criterion(pred[:,:min(self.pred_len[idx],trues.shape[1]),:], trues[:,:min(self.pred_len[idx],trues.shape[1]),:])
            loss_feq = (torch.fft.rfft(pred[:,:min(self.pred_len[idx],trues.shape[1]),:], dim=1) - torch.fft.rfft(trues[:,:min(self.pred_len[idx],trues.shape[1]),:], dim=1)).abs().mean()

            if loss_masks is not None:
                losses = losses * loss_masks
                loss = losses.sum() / (loss_masks.sum() + 1e-6)  
            else:
                loss = torch.mean(losses)
            total_loss += loss
        total_loss = total_loss / len(preds)
        return total_loss




if __name__ == '__main__':
    config = TimeMixerppConfig()

    model = TimeMixerppForPrediction(config)

    input_ids = torch.randn(16,96,7)
    labels = torch.randn(16,24,7)
    loss_masks = torch.ones(16,24,7)
    item = {
        'input_ids':input_ids,
        'labels': labels,
        'loss_masks': loss_masks,
    }
    num_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {num_total_params}")
    output = model(**item)
    print(output)
