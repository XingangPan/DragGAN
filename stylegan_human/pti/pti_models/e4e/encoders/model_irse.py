from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Dropout, Sequential, Module
from encoder4editing.models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE, l2_norm

"""
Modified Backbone implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
"""


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir', drop_ratio=0.4, affine=True):
        super(Backbone, self).__init__()
        assert input_size in [112, 224], "input_size should be 112 or 224"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        if input_size == 112:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(drop_ratio),
                                           Flatten(),
                                           Linear(512 * 7 * 7, 512),
                                           BatchNorm1d(512, affine=affine))
        else:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(drop_ratio),
                                           Flatten(),
                                           Linear(512 * 14 * 14, 512),
                                           BatchNorm1d(512, affine=affine))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)


def IR_50(input_size):
    """Constructs a ir-50 model."""
    model = Backbone(input_size, num_layers=50, mode='ir', drop_ratio=0.4, affine=False)
    return model


def IR_101(input_size):
    """Constructs a ir-101 model."""
    model = Backbone(input_size, num_layers=100, mode='ir', drop_ratio=0.4, affine=False)
    return model


def IR_152(input_size):
    """Constructs a ir-152 model."""
    model = Backbone(input_size, num_layers=152, mode='ir', drop_ratio=0.4, affine=False)
    return model


def IR_SE_50(input_size):
    """Constructs a ir_se-50 model."""
    model = Backbone(input_size, num_layers=50, mode='ir_se', drop_ratio=0.4, affine=False)
    return model


def IR_SE_101(input_size):
    """Constructs a ir_se-101 model."""
    model = Backbone(input_size, num_layers=100, mode='ir_se', drop_ratio=0.4, affine=False)
    return model


def IR_SE_152(input_size):
    """Constructs a ir_se-152 model."""
    model = Backbone(input_size, num_layers=152, mode='ir_se', drop_ratio=0.4, affine=False)
    return model
