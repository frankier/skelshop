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

    def preproc(self, data):
        self.model.eval()

        with torch.no_grad():
            data = torch.from_numpy(data)
            data = data.unsqueeze(0)
            data = data.float().to(self.dev)
            data = self.model.extract_feature(data)
            return data.cpu().numpy()[0]
