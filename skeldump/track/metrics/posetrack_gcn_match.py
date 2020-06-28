import argparse

import torch
from lighttrack.graph.gcn_utils.io import IO
from lighttrack.graph.gcn_utils.processor_siamese_gcn import SGCN_Processor


class PoseMatcher(SGCN_Processor):
    def __init__(self, config):
        self.config = config
        self.load_arg([])
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()
        return

    def get_parser(self, add_help=False):
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=False,
            parents=[parent_parser],
            description="Graph Convolution Network for Pose Matching",
        )

        parser.set_defaults(config=self.config)
        return parser

    def inference(self, data_1, data_2):
        self.model.eval()

        with torch.no_grad():
            data_1 = torch.from_numpy(data_1)
            data_1 = data_1.unsqueeze(0)
            data_1 = data_1.float().to(self.dev)

            data_2 = torch.from_numpy(data_2)
            data_2 = data_2.unsqueeze(0)
            data_2 = data_2.float().to(self.dev)

            feature_1, feature_2 = self.model.forward(data_1, data_2)

        # euclidian distance
        diff = feature_1 - feature_2
        dist_sq = torch.sum(pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        return dist.data.cpu().numpy()[0]
