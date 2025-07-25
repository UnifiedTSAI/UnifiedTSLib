import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput
from transformers import PreTrainedModel
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import math
from .configuration_timesnet import TimesNetConfig

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
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


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res
    
@dataclass
class TimesNetOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
class TimesNetPreTrainedModel(PreTrainedModel):
    config_class = TimesNetConfig
    base_model_prefix = "model"
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
class TimesNetModel(TimesNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.task_name = config.task_name
        self.seq_len = config.seq_len
        self.label_len = config.label_len
        self.pred_len = config.pred_len
        self.max_pred_len = max(config.pred_len) if isinstance(config.pred_len, list) else config.pred_len
        self.enc_embedding = DataEmbedding(
            config.enc_in, config.d_model, config.embed, config.freq, config.dropout
        )
        self.model = nn.ModuleList([TimesBlock(config) for _ in range(config.e_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(config.d_model, config.c_out, bias=True)
        elif self.task_name in ["imputation", "anomaly_detection"]:
            self.projection = nn.Linear(config.d_model, config.c_out, bias=True)
        elif self.task_name == "classification":
            self.act = nn.GELU()
            self.dropout = nn.Dropout(config.dropout)
            self.projection = nn.Linear(config.d_model * config.seq_len, config.num_class)
        self.post_init()
    def forward(
        self,
        input_ids: torch.FloatTensor = None,
        x_mark_enc: Optional[torch.FloatTensor] = None,
        x_dec: Optional[torch.FloatTensor] = None,
        x_mark_dec: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
    ):
        x_enc = input_ids
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            # Normalization
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
            # Embedding
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
            for i in range(self.config.e_layers):
                enc_out = self.layer_norm(self.model[i](enc_out))
            dec_out = self.projection(enc_out)
            # De-Normalization
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
            out = dec_out[:, -self.pred_len:, :]
        
            preds = []
            if isinstance(self.pred_len, list):
                for plen in self.pred_len:
                    preds.append(out[:, :plen, :])
            else:
                preds.append(out)
            return preds
        
class TimesNetForPrediction(TimesNetPreTrainedModel):
    def __init__(self, configs):
        super().__init__(configs)
        self.criterion = nn.MSELoss(reduction='none')
        self.model = TimesNetModel(configs)
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
    ) -> Union[Tuple, TimesNetOutput]:
        dataset_identifier = False
        if dataset_idx is not None:
            dataset_identifier = (dataset_idx == dataset_idx[0]).all()
        if inner_batchsize is None:
            if len(input_ids.shape) == 2:
                input_ids = input_ids.unsqueeze(-1)
                labels = labels.unsqueeze(-1)
            preds = self.model(input_ids=input_ids)
            loss = self.calc_loss(preds, labels, loss_masks)
            return TimesNetOutput(loss=loss, logits=preds)
        elif not dataset_identifier:
            total_loss = 0
            for batch_idx in range(input_ids.size(0)):
                flatten_seq_len = inner_batchsize[batch_idx] * (
                    context_length[batch_idx] + prediction_length[batch_idx]) * num_channel[batch_idx]
                input_batch = input_ids[batch_idx][:flatten_seq_len]
                input_batch = input_batch.reshape(inner_batchsize[batch_idx], num_channel[batch_idx],
                                                  context_length[batch_idx] + prediction_length[batch_idx]).permute(0, 2, 1)
                input_context = input_batch[:, :context_length[batch_idx], :]
                input_labels = input_batch[:, context_length[batch_idx]:, :]
                preds = self.model(input_ids=input_context)
                loss = self.calc_loss(preds, input_labels, loss_masks)
                total_loss += loss
            total_loss = total_loss / input_ids.size(0)
            return TimesNetOutput(loss=total_loss)
        else:
            input_contexts = []
            input_labels = []
            for batch_idx in range(input_ids.size(0)):
                flatten_seq_len = inner_batchsize[batch_idx] * (
                    context_length[batch_idx] + prediction_length[batch_idx]) * num_channel[batch_idx]
                input_batch = input_ids[batch_idx][:flatten_seq_len]
                input_batch = input_batch.reshape(inner_batchsize[batch_idx], num_channel[batch_idx],
                                                  context_length[batch_idx] + prediction_length[batch_idx]).permute(0, 2, 1)
                input_context = input_batch[:, :context_length[batch_idx], :]
                input_label = input_batch[:, context_length[batch_idx]:, :]
                input_contexts.append(input_context)
                input_labels.append(input_label)
            input_context = torch.cat(input_contexts, dim=0)
            input_label = torch.cat(input_labels, dim=0)
            preds = self.model(input_ids=input_context)
            loss = self.calc_loss(preds, input_label, loss_masks)
            return TimesNetOutput(loss=loss)
    def calc_loss(self, preds, trues, loss_masks):
        if not isinstance(preds, list):
            preds = [preds]
        total_loss = 0
        for idx, pred in enumerate(preds):
            losses = self.criterion(pred[:, :min(self.pred_len[idx], trues.shape[1]), :],
                                    trues[:, :min(self.pred_len[idx], trues.shape[1]), :])
            loss_feq = (torch.fft.rfft(pred[:, :min(self.pred_len[idx], trues.shape[1]), :], dim=1) -
                        torch.fft.rfft(trues[:, :min(self.pred_len[idx], trues.shape[1]), :], dim=1)).abs().mean()
            if loss_masks is not None:
                losses = losses * loss_masks
                loss = losses.sum() / (loss_masks.sum() + 1e-6)
            else:
                loss = torch.mean(losses)
            total_loss += loss
        total_loss = total_loss / len(preds)
        return total_loss
