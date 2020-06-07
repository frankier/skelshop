import argparse
import sys

import torch
from lighttrack.graph.gcn_utils.io import IO
from lighttrack.graph.gcn_utils.processor_siamese_gcn import SGCN_Processor

from .bbox import bbox_invalid
from .utils import graph_pair_to_data, keypoints_to_graph


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


def mk_posetrack_gcn_pose_matcher(pose_matcher_config):
    pose_matcher_obj = PoseMatcher(pose_matcher_config)

    def pose_matcher(det_cur, det_prev):
        return get_pose_matching_score(
            pose_matcher_obj,
            det_cur.posetrack_kps,
            det_prev.posetrack_kps,
            det_cur.bbox,
            det_prev.bbox,
        )

    return pose_matcher


def get_pose_matching_score(pose_matcher, keypoints_A, keypoints_B, bbox_A, bbox_B):
    if bbox_invalid(bbox_A) or bbox_invalid(bbox_B):
        print("graph not correctly generated!")
        return sys.maxsize

    graph_A = keypoints_to_graph(keypoints_A, bbox_A)
    graph_B = keypoints_to_graph(keypoints_B, bbox_B)
    data_A, data_B = graph_pair_to_data((graph_A, graph_B))
    return pose_matcher.inference(data_A, data_B)
