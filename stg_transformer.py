from STG_model import (
    AttnPool,
    DistributionHead,
    IsoFormerSTGTransformer,
    LowRankActivityBias,
    MultiModalFusionLayer,
    PositionalEncoding,
    SpatioTemporalAttention,
    STGTransformerLayer,
    TimeTokenEncoder,
)


class SingleTaskSTGTransformer(IsoFormerSTGTransformer):
    """
    Backward-compatible alias around IsoFormerSTGTransformer.
    Keeps existing import paths and call signatures used by training scripts.
    """

    def __init__(
        self,
        num_activities,
        num_resources,
        d_model=128,
        num_heads=8,
        num_layers=4,
        model_type="full",
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(
            num_activities=num_activities,
            num_resources=num_resources,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            model_type=model_type,
            **kwargs,
        )

    def forward(
        self,
        act_seq,
        res_seq,
        time_features,
        time_matrix=None,
        graph_matrix=None,
        padding_mask=None,
        return_dist=False,
    ):
        return super().forward(
            act_seq=act_seq,
            res_seq=res_seq,
            time_features=time_features,
            time_matrix=time_matrix,
            graph_matrix=graph_matrix,
            padding_mask=padding_mask,
            return_dist=return_dist,
        )
