from transformers import PretrainedConfig
from typing import Optional, Tuple, List, Union

class TimeMixerppConfig(PretrainedConfig):
    model_type = "time_mixerpp"
    
    def __init__(
        self,

        d_model: int = 64,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 256,

        down_sampling_window: int = 2,
        down_sampling_layers: int = 2,
        down_sampling_method: str = "conv",  # ["max", "avg", "conv"]

        seq_len: int = 96,
        pred_len: List[int] = [96,192,336,720],
        label_len: int = 48,
        top_k: int = 3,

        enc_in: int = 1,
        c_out: int = 1,
        channel_independence: bool = True,
        channel_mixing: bool = False,
        

        dropout: float = 0.1,
        output_attention: bool = False,
        

        embed: str = "timeF",
        freq: str = "h",
        num_kernels: int = 6,
        initializer_range: float = 0.02,
        **kwargs
    ):

        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        

        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_method = down_sampling_method
        

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.top_k = top_k
        

        self.enc_in = enc_in
        self.c_out = c_out
        self.channel_independence = channel_independence
        self.channel_mixing = channel_mixing
        
 
        self.dropout = dropout
        self.output_attention = output_attention
        

        self.embed = embed
        self.freq = freq
        self.num_kernels = num_kernels
        self.initializer_range = initializer_range
        
        super().__init__(**kwargs)