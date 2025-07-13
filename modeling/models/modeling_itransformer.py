# modeling_itransformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers.utils import ModelOutput
from transformers import PreTrainedModel
from .configuration_itransformer import iTransformerConfig
from typing import Optional, Tuple, List,Union
import math
import numpy as np

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

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

class DataEmbedding_inverted(nn.Module):
    def __init__(self, seq_len, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        x = x.permute(0, 2, 1)  # [B, N, T]
        x = self.value_embedding(x)  # [B, N, D]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = AttentionLayer(
            FullAttention(False, config.factor, attention_dropout=config.dropout, output_attention=False),
            config.d_model, config.n_heads
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model)
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attn_mask=None):
        # Attention
        x = x + self.dropout(self.attention(x, x, x, attn_mask)[0])
        x = self.norm1(x)

        # FFN
        x = x + self.dropout(self.ffn(x))
        x = self.norm2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x



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
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, L, H, -1)
        values = self.value_projection(values).view(B, L, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        return self.out_projection(out.view(B, L, -1))


@dataclass
class iTransformerOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class iTransformerPreTrainedModel(PreTrainedModel):
    config_class = iTransformerConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class iTransformerModel(iTransformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.task_name = config.task_name
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.max_pred_len = max(config.pred_len) if isinstance(config.pred_len, list) else config.pred_len

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(config.seq_len, config.d_model,
                                                    config.embed, config.freq, config.dropout)

        # Encoder
        self.encoder = Encoder(
            [EncoderLayer(config) for _ in range(config.e_layers)],
            norm_layer=nn.LayerNorm(config.d_model)
        )

        # Prediction
        self.predict_layer = nn.Linear(config.d_model, self.max_pred_len)
        self.revin_layer =  Normalize(self.config.enc_in,
                          affine=False, subtract_last=False
                          )

        # Initialize weights
        self.post_init()

    def forward(self,input_ids: torch.FloatTensor = None,x_mark_enc: torch.FloatTensor = None):
        x_enc = input_ids
        # Normalization
        x_enc = self.revin_layer(x_enc,'norm')

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out)

        # Prediction
        dec_out = self.predict_layer(enc_out).permute(0, 2, 1)

        # De-normalization
        dec_out = self.revin_layer(dec_out, 'denorm')

        # Split predictions
        preds = []
        if isinstance(self.pred_len, list):
            for plen in self.pred_len:
                preds.append(dec_out[:, :plen, :])
        else:
            preds.append(dec_out)

        return preds


class iTransformerForPrediction(iTransformerPreTrainedModel):
    def __init__(self, configs):
        super().__init__(configs)
        self.criterion = nn.MSELoss(reduction='none')
        self.model = iTransformerModel(configs)
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
    ) -> Union[Tuple, iTransformerOutput]:
      
        dataset_identifier = False
        if dataset_idx is not None:

            dataset_identifier = (dataset_idx == dataset_idx[0]).all()

        if inner_batchsize is None:
            if len(input_ids.shape) == 2:
                input_ids = input_ids.unsqueeze(-1)
                labels = labels.unsqueeze(-1)
            preds = self.model(input_ids=input_ids)
            loss = self.calc_loss(preds, labels, loss_masks)

            return iTransformerOutput(loss=loss, logits=preds)
        elif not dataset_identifier:
            total_loss = 0
            for batch_idx in range(input_ids.size(0)):
                flatten_seq_len = inner_batchsize[batch_idx] * (
                            context_length[batch_idx] + prediction_length[batch_idx]) * num_channel[batch_idx]
                input_batch = input_ids[batch_idx][:flatten_seq_len]
                input_batch = input_batch.reshape(inner_batchsize[batch_idx], num_channel[batch_idx],
                                                  context_length[batch_idx] + prediction_length[batch_idx]).permute(0,
                                                                                                                    2,
                                                                                                                    1)
                input_context = input_batch[:, :context_length[batch_idx], :]
                input_labels = input_batch[:, context_length[batch_idx]:, :]
                preds = self.model(input_ids=input_context)
                loss = self.calc_loss(preds, input_labels, loss_masks)
                total_loss += loss

            total_loss = total_loss / input_ids.size(0)
            return iTransformerOutput(loss=total_loss)
        else:
            input_contexts = []
            input_labels = []
            for batch_idx in range(input_ids.size(0)):
                flatten_seq_len = inner_batchsize[batch_idx] * (
                            context_length[batch_idx] + prediction_length[batch_idx]) * num_channel[batch_idx]
                input_batch = input_ids[batch_idx][:flatten_seq_len]
                input_batch = input_batch.reshape(inner_batchsize[batch_idx], num_channel[batch_idx],
                                                  context_length[batch_idx] + prediction_length[batch_idx]).permute(0,
                                                                                                                    2,
                                                                                                                    1)
                input_context = input_batch[:, :context_length[batch_idx], :]
                input_label = input_batch[:, context_length[batch_idx]:, :]
                input_contexts.append(input_context)
                input_labels.append(input_label)

            input_context = torch.cat(input_contexts, dim=0)
            input_label = torch.cat(input_labels, dim=0)
            preds = self.model(input_ids=input_context)
            loss = self.calc_loss(preds, input_label, loss_masks)
            return iTransformerOutput(loss=loss)

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
    config = iTransformerConfig()

    model = iTransformerForPrediction(config).cpu()

    input_ids = torch.randn(16,96,7).cpu()
    labels = torch.randn(16,24,7).cpu()
    loss_masks = torch.ones(16,24,7).cpu()
    item = {
        'input_ids':input_ids,
        'labels': labels,
        'loss_masks': loss_masks,
    }
    num_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {num_total_params}")
    output = model(**item)
    print(output)
