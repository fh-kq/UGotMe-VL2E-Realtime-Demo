# -*- coding:utf-8 -*-
"""Local drop-in replacement for ``mmcv.cnn.xavier_init``.

The official repo depends on ``mmcv-full`` only for this single helper.
mmcv-full is painful to build on Windows / against recent PyTorch, so we
reimplement the exact semantics here and import from this module instead.

Reference behaviour (mmcv/cnn/utils/weight_init.py)::

    def xavier_init(module, gain=1, bias=0, distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if hasattr(module, 'weight') and module.weight is not None:
            if distribution == 'uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            else:
                nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
"""

import torch.nn as nn


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    """Initialize the weights of a module with Xavier initialization."""
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
