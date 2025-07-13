from transformers import PretrainedConfig

class AutoformerConfig(PretrainedConfig):
    model_type = "autoformer"

    def __init__(
        self,
        seq_len=96,
        label_len=48,
        pred_len=96,
        e_layers=2,
        d_layers=1,
        d_model=512,
        d_ff=2048,
        n_heads=8,
        enc_in=7,
        dec_in=7,
        c_out=7,
        moving_avg=25,
        factor=3,
        dropout=0.1,
        activation="gelu",
        embed="timeF",
        freq="h",
        task_name="long_term_forecast",
        num_class=1,
        initializer_range=0.02,
        **kwargs
    ):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.moving_avg = moving_avg
        self.factor = factor
        self.dropout = dropout
        self.activation = activation
        self.embed = embed
        self.freq = freq
        self.task_name = task_name
        self.num_class = num_class
        self.initializer_range = initializer_range
        super().__init__(**kwargs)
