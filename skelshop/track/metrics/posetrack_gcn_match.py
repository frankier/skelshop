from tempfile import TemporaryDirectory
from typing import Any

PoseMatcher: Any


def _mk_pose_matcher():
    import argparse

    import torch
    from lighttrack.graph.gcn_utils.io import IO
    from lighttrack.graph.gcn_utils.processor_siamese_gcn import SGCN_Processor

    class PoseMatcher(SGCN_Processor):
        def __init__(self, config):
            self.config = config
            argv = []
            if not torch.cuda.is_available():
                argv.extend(["--use_gpu", "0"])
            self.temp_dir = TemporaryDirectory()
            argv.extend(["--work_dir", self.temp_dir.name])
            self.load_arg(argv)
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

        def __del__(self):
            self.temp_dir.cleanup()

    return PoseMatcher


def __getattr__(name: str) -> Any:
    if name == "PoseMatcher":
        global PoseMatcher
        PoseMatcher = _mk_pose_matcher()
        return PoseMatcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
