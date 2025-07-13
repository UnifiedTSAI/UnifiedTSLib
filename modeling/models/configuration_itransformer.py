# configuration_itransformer.py
from transformers import PretrainedConfig
from typing import List, Optional


class iTransformerConfig(PretrainedConfig):
    model_type = "itransformer"

    def __init__(
            self,

            d_model: int = 512,
            e_layers: int = 4,
            n_heads: int = 8,
            d_ff: int = 2048,
            dropout: float = 0.1,
            activation: str = 'gelu',


            seq_len: int = 96,
            pred_len: List[int] = [96, 192, 336, 720],
            label_len: int = 48,
            enc_in: int = 7,
            dec_in: int = 7,
            c_out: int = 7,


            task_name: str = 'long_term_forecast',
            num_class: int = 1,


            embed: str = 'timeF',
            freq: str = 'h',
            factor: int = 5,


            initializer_range: float = 0.02,
            **kwargs
    ):

        self.d_model = d_model
        self.e_layers = e_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation


        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out


        self.task_name = task_name
        self.num_class = num_class


        self.embed = embed
        self.freq = freq
        self.factor = factor


        self.initializer_range = initializer_range

        super().__init__(**kwargs)