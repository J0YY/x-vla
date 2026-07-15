"""χ-VLA: a fully tensor-decomposable Vision-Language-Action model.

See ``spec.md`` for the full specification. This package builds the model
bottom-up following the recommended build sequence (spec §21):

    Milestone 1  tensor-transformer core  (this stage)
    Milestone 2  single χ-ViT
    Milestone 3  dual χ-vision system
    Milestone 4  small χ-VLA
    ...

Every learned component is a structured polynomial: linear/affine maps,
tensor contractions, element-wise products, residual sums, fixed masks,
learned constants, and foldable scalar rescalings. No softmax, no GELU/
ReLU/SiLU, no LayerNorm/RMSNorm survive in the exported inference graph.
"""

from xvla.nn.bilinear import BilinearFFN
from xvla.nn.attention import BilinearAttention
from xvla.nn.normalization import RmsBatchNorm
from xvla.nn.block import ChiTransformerBlock, ChiTransformer

__all__ = [
    "BilinearFFN",
    "BilinearAttention",
    "RmsBatchNorm",
    "ChiTransformerBlock",
    "ChiTransformer",
]
