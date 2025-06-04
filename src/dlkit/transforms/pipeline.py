from torch import Tensor
import torch.nn as nn
from dlkit.transforms.chain import TransformChain


class FeaturesTargetsPipeline(nn.Module):
    feature_chain: TransformChain
    target_chain: TransformChain
    model: nn.Module

    def __init__(
        self,
        x: Tensor,
        pre_chain: TransformChain | None = None,
        after_chain: TransformChain | None = None,
    ):
        super().__init__()
        self.x = x
        self.pre_chain = pre_chain
        self.after_chain = after_chain

    def __enter__(self):
        if self.pre_chain is not None:
            self.x = self.pre_chain(self.x)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.after_chain is not None:
            self.x = self.after_chain(self.x)
