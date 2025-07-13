# modeling_itransformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers.utils import ModelOutput
from transformers import PreTrainedModel
from typing import Optional, Tuple, List,Union
import math
import numpy as np
from .modeling_itransformer import iTransformerModel,iTransformerConfig
from .modeling_timemixerpp import TimeMixerppModel,TimeMixerppConfig
from .modeling_timesnet import TimesNetModel,TimesNetConfig
from .modeling_autoformer import AutoformerModel,AutoformerConfig
from transformers import PretrainedConfig
from typing import Optional, Tuple, List, Union


class UnifiedTSConfig(PretrainedConfig):
    model_type = "UnifiedTS"

    def __init__(
            self,
            model_name: str='iTransformer',
            seq_len: int = 96,
            pred_len: List[int] = [96, 192, 336, 720],
            channel_mixing:bool=False,
            **kwargs
    ):

        self.model_name = model_name
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channel_mixing = channel_mixing

        super().__init__(**kwargs)


@dataclass
class UnifiedTSPredictionOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    # hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class UnifiedTSForPrediction(PreTrainedModel):
    config_class = TimeMixerppConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # _no_split_modules = ["TimeMoeDecoderLayer"]
    # _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = True

    def __init__(self, configs: UnifiedTSConfig, **kwargs):
        super().__init__(configs)
        self.criterion = nn.MSELoss(reduction='none')
        self.pred_len = configs.pred_len
        self.model_dict = {
            'iTransformer': iTransformerModel,
            'timemixerpp': TimeMixerppModel,
            'timesnet': TimesNetModel,
            'autoformer': AutoformerModel,
        }
        self.config_dict = {
            'iTransformer': iTransformerConfig,
            'timemixerpp': TimeMixerppConfig,
            'timesnet': TimesNetConfig,
            'autofomer': AutoformerConfig,
        }
        self.config = configs
        model_config = self.config_dict[configs.model_name]()
        model_config.pred_len = configs.pred_len
        model_config.channel_mixing = configs.channel_mixing
        self.model = self.model_dict[configs.model_name](model_config)

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
    ) -> Union[Tuple, UnifiedTSPredictionOutput]:


        dataset_identifier = False
        if dataset_idx is not None:
            dataset_identifier = (dataset_idx == dataset_idx[0]).all()

        if inner_batchsize is None:
            if len(input_ids.shape) == 2:
                input_ids = input_ids.unsqueeze(-1)
                labels = labels.unsqueeze(-1)
            preds = self.model(input_ids=input_ids)
            loss = self.calc_loss(preds, labels, loss_masks)

            return UnifiedTSPredictionOutput(loss=loss, logits=preds)
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
            return UnifiedTSPredictionOutput(loss=total_loss)
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
            return UnifiedTSPredictionOutput(loss=loss)

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
    config = UnifiedTSConfig()
    # model = TimeMixerppModel(config)
    model = UnifiedTSForPrediction.from_pretrained("tmp/")
    # model = UnifiedTSForPrediction(config).cpu()
    # model.save_pretrained("tmp/")
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
