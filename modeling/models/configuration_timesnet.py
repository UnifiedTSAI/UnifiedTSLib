from transformers import PretrainedConfig
from typing import Optional

class TimesNetConfig(PretrainedConfig):
    model_type = "timesnet"

    def __init__(
            self,
            # 基础参数
            d_model: int = 16,
            d_ff: int = 32,
            e_layers: int = 2,
            seq_len: int = 96,
            label_len: int = 0,
            pred_len: int = 96,
            enc_in: int = 7,
            c_out: int = 7,
            dropout=0.1,
            embed="timeF",
            freq="h",
            top_k=2,
            num_kernels=6,

            # 下采样相关
            down_sampling_layers: int = 3,
            down_sampling_method: str = "avg",
            down_sampling_window: int = 2,


            initializer_range: float = 0.02,
            **kwargs
    ):

        self.d_model = d_model
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.freq = freq
        self.top_k = top_k
        self.dropout = dropout
        self.embed = embed
        self.top_k = top_k
        self.num_kernels = num_kernels

        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_method = down_sampling_method
        self.down_sampling_window = down_sampling_window


        self.initializer_range = initializer_range

        super().__init__(**kwargs)
