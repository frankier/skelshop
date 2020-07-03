import torch
import torch.nn as nn
import torch.nn.functional as F
from mmskeleton.models.backbones.st_gcn_aaai18 import st_gcn_block


class FlexStGcn(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_size,
        graph,
        edge_importance_weighting=True,
        data_bn=True,
        temporal_kernel_size=9,
        **kwargs
    ):
        super().__init__()

        # load graph
        self.graph = graph
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A", A)

        # build networks
        spatial_kernel_size = A.size(0)
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = (
            nn.BatchNorm1d(in_channels * A.size(1)) if data_bn else lambda x: x
        )
        kwargs0 = {k: v for k, v in kwargs.items() if k != "dropout"}
        self.st_gcn_networks = nn.ModuleList(
            (
                st_gcn_block(
                    in_channels, 64, kernel_size, 1, residual=False, **kwargs0
                ),
                st_gcn_block(64, 64, kernel_size, 1, **kwargs),
                st_gcn_block(64, 64, kernel_size, 1, **kwargs),
                st_gcn_block(64, 64, kernel_size, 1, **kwargs),
                st_gcn_block(64, 128, kernel_size, 2, **kwargs),
                st_gcn_block(128, 128, kernel_size, 1, **kwargs),
                st_gcn_block(128, 128, kernel_size, 1, **kwargs),
                st_gcn_block(128, 256, kernel_size, 2, **kwargs),
                st_gcn_block(256, 256, kernel_size, 1, **kwargs),
                st_gcn_block(256, 256, kernel_size, 1, **kwargs),
            )
        )

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(self.A.size())) for i in self.st_gcn_networks]
            )
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, embed_size, kernel_size=1)

    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature
