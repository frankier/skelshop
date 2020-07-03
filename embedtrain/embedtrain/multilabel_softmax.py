"""
Exploring the Limits of Weakly Supervised Pretraining, D Mahajan, et al. 2018
https://arxiv.org/abs/1805.00932
"""
import pytorch_metric_learning.utils.common_functions as c_f
import torch
from pytorch_metric_learning.losses.base_metric_loss_function import (
    BaseMetricLossFunction,
)
from pytorch_metric_learning.losses.weight_regularizer_mixin import (
    WeightRegularizerMixin,
)
from torch.nn import functional as F


def cross_entropy(input, target):
    return torch.sum(-target * F.log_softmax(input), dim=1)


class NormalizedMultilabelSoftmaxLoss(WeightRegularizerMixin, BaseMetricLossFunction):
    """
    Taken from pytorch_metric_learning and patched to use cross_entropy, above.
    """

    def __init__(self, temperature, embedding_size, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.W = torch.nn.Parameter(torch.randn(embedding_size, num_classes))
        self.num_classes = num_classes

    def compute_loss(self, embeddings, multilabels, indices_tuple):
        # Put multilabels into matrix
        target = torch.zeros(len(multilabels), self.num_classes)
        for idx, ml in enumerate(multilabels):
            target[idx, ml] = 1
        # Compute normal loss
        normalized_W = torch.nn.functional.normalize(self.W, p=2, dim=0)
        exponent = torch.matmul(embeddings, normalized_W) / self.temperature
        unweighted_loss = cross_entropy(exponent, target)
        miner_weighted_loss = unweighted_loss
        loss_dict = {
            "loss": {
                "losses": miner_weighted_loss,
                "indices": c_f.torch_arange_from_size(embeddings),
                "reduction_type": "element",
            }
        }
        loss_dict["reg_loss"] = self.regularization_loss(self.W.t())
        return loss_dict

    def sub_loss_names(self):
        return ["loss", "reg_loss"]
