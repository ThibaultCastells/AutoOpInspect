from torch import nn

leaf_types = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Linear,
    nn.Identity,

    nn.PReLU,
    nn.ReLU,
    nn.ReLU6,
    nn.RReLU,
    nn.SELU,
    nn.CELU,
    nn.GELU,
    nn.Sigmoid,
    nn.SiLU,
    nn.Mish,
    nn.Threshold,
    nn.GLU,
    nn.Hardsigmoid,
    nn.Hardtanh,
    nn.Hardswish,
    nn.LeakyReLU,
    nn.LogSigmoid,

    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AdaptiveMaxPool1d,
    nn.AdaptiveMaxPool2d,
    nn.AdaptiveMaxPool3d,

    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,

    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.SyncBatchNorm,
    nn.LayerNorm,
    
    nn.Dropout, 
    nn.Embedding,
    nn.Softmax,
    nn.Softmax2d,
    nn.LogSoftmax,
)

def get_other_info(module):
    # If a handler function exists for the module's type, call it
    extra_rep = ''
    if isinstance(module, leaf_types):
        extra_rep = module.extra_repr()
        if extra_rep:
            return '(' + extra_rep + ')'

    # If no specific handler, return an empty string
    return extra_rep
