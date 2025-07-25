#!/usr/bin/env python
# -*- coding:utf-8 _*-
import math
from dataclasses import field, dataclass
from functools import partial

import inspect

import transformers
import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_scheduler
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from typing import Optional, Tuple, List, Union,Dict,Any

class UnifiedTrainer(transformers.Trainer):
    epsilon = 1e-8

    def __init__(self, label_column: str = 'labels', loss_mask_column: str = 'loss_mask', *positional_args, **kwargs):
        super().__init__(*positional_args, **kwargs)
        self.tokenizer = kwargs.get("tokenizer", None)
        self.label_column = label_column
        self.loss_mask_column = loss_mask_column

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        optimizer = self.optimizer if optimizer is None else optimizer
        min_lr_ratio = self.args.min_learning_rate / self.args.learning_rate
        if self.lr_scheduler is None:
            if self.args.lr_scheduler_type == 'cosine':
                self.lr_scheduler = get_cosine_schedule_with_warmup_min_lr(
                    optimizer=optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    min_lr_ratio=min_lr_ratio,
                )
            else:
                self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            params = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns = list(set(
                params + self.label_names + [
                    "label",
                    "label_ids",
                    self.label_column,
                    self.loss_mask_column
                ]
            ))

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
            else:
                logits = outputs

        return (logits[0], None, None)



@dataclass
class UnifiedTrainingArguments(transformers.TrainingArguments):
    min_learning_rate: float = field(
        default=0, metadata={"help": "Minimum learning rate for cosine_schedule"}
    )


def _get_cosine_schedule_with_warmup_and_min_lr_lambda(
        current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr_ratio: float,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))

    return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_ratio)


def get_cosine_schedule_with_warmup_min_lr(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        min_lr_ratio: float = 0,
        last_epoch: int = -1
):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_and_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
