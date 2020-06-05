import argparse
import sys
from pprint import pprint

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

        margin = 0.2
        distance = dist.data.cpu().numpy()[0]
        print("_____ Pose Matching: [dist: {:04.2f}]".format(distance))
        if dist >= margin:
            return False, distance  # Do not match
        else:
            return True, distance  # Match


def get_track_id_SGCN_plus(
    pose_matcher, dets_cur_frame, dets_list_prev_frame, pose_matching_threshold=0.5
):
    min_index = None
    min_matching_score = sys.maxsize
    bbox_cur_frame = dets_cur_frame.bbox
    keypoints_cur_frame = dets_cur_frame.keypoints
    track_id = -1
    for det_index, det_dict in enumerate(dets_list_prev_frame):
        bbox_prev_frame = det_dict.bbox
        keypoints_prev_frame = det_dict.keypoints
        pose_matching_score = get_pose_matching_score(
            pose_matcher,
            keypoints_cur_frame,
            keypoints_prev_frame,
            bbox_cur_frame,
            bbox_prev_frame,
        )
        pprint(
            [keypoints_cur_frame, keypoints_prev_frame, bbox_cur_frame, bbox_prev_frame]
        )
        print("pose_matching_score", pose_matching_score)

        if (
            pose_matching_score <= pose_matching_threshold
            and pose_matching_score <= min_matching_score
        ):
            # match the target based on the pose matching score
            min_matching_score = pose_matching_score
            min_index = det_index

    if min_index is None:
        return -1, None, 0
    else:
        track_id = dets_list_prev_frame[min_index].track_id
        return track_id, min_index, min_matching_score


def get_pose_matching_score(pose_matcher, keypoints_A, keypoints_B, bbox_A, bbox_B):
    if keypoints_A == [] or keypoints_B == []:
        print("graph not correctly generated!")
        return sys.maxsize

    if bbox_invalid(bbox_A) or bbox_invalid(bbox_B):
        print("graph not correctly generated!")
        return sys.maxsize

    graph_A, flag_pass_check = keypoints_to_graph(keypoints_A, bbox_A)
    if flag_pass_check is False:
        print("graph not correctly generated!")
        return sys.maxsize

    graph_B, flag_pass_check = keypoints_to_graph(keypoints_B, bbox_B)
    if flag_pass_check is False:
        print("graph not correctly generated!")
        return sys.maxsize

    sample_graph_pair = (graph_A, graph_B)
    data_A, data_B = graph_pair_to_data(sample_graph_pair)
    flag_match, dist = pose_matcher.inference(data_A, data_B)
    return dist
