import argparse
import collections
import numpy as np
import sys
from lighttrack.graph.gcn_utils.processor_siamese_gcn import SGCN_Processor
from lighttrack.graph.gcn_utils.io import IO
import torch
from pprint import pprint


class objectview(object):
    def __init__(self, **d):
        self.__dict__ = d


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
            description='Graph Convolution Network for Pose Matching')

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
        print("_____ Pose Matching: [dist: {:04.2f}]". format(distance))
        if dist >= margin:
            return False, distance  # Do not match
        else:
            return True, distance  # Match


class PoseTrack():
    def __init__(self, pose_matcher, enlarge_scale=0.2):
        self.pose_matcher = pose_matcher
        self.enlarge_scale = enlarge_scale
        self.next_id = 0
        self.img_id = 0
        self.dets_list_q = collections.deque(maxlen=5)

    def reset(self):
        self.next_id = 0
        self.img_id = 0
        self.dets_list_q.clear()

    # MAIN FUNCTION
    def pose_track(self, kps):
        for i in range(len(self.dets_list_q)):
            index = -(i+1)
            prev_candidates = list(self.dets_list_q)[index]
            next_ids = [prev_candidates[item]['track_id'] for item in range(len(prev_candidates)) if prev_candidates[item]['track_id'] != None]
            if next_ids != []:
                self.next_id = max(max(next_ids)+1, self.next_id)

        self.posetrack(kps)
        self.img_id += 1

    def posetrack(self, kps):
        human_candidates = self.get_human_bbox_and_keypoints(kps)
        num_dets = len(human_candidates)
        print("num_dets", num_dets)
        if num_dets <= 0:
            self.dets_list_q.append([])
            return

        print("self.img_id", self.img_id)
        if self.img_id == 0:
            self.first_frame(human_candidates)
            return

        ##### traverse all prev frame dicts #####
        tracked_dets_list = []
        tracked_dets_ids = []
        untracked_dets_ids = list(range(len(human_candidates)))
        for i in range(len(self.dets_list_q)):
            index = -(i+1)
            dets_list_prev_frame = self.dets_list_q[index]
            if len(untracked_dets_ids) > 0:
                self.traverse_each_prev_frame(human_candidates, dets_list_prev_frame, tracked_dets_list,tracked_dets_ids,  untracked_dets_ids)
            untracked_dets_ids = list(set(untracked_dets_ids)-set(tracked_dets_ids))

        ##handle all unmatched item

        for det_id in untracked_dets_ids:
            det_dict = {"img_id":self.img_id,
                                "det_id": 0,
                                "track_id": -1,
                                "bbox":[0,0,2,2],
                                "openpose_kps": [],
                                "keypoints": []}
            bbox_det = human_candidates[det_id][0]
            bbox_x1y1x2y2 = xywh_to_x1y1x2y2(bbox_det)
            bbox_in_xywh = enlarge_bbox(bbox_x1y1x2y2, self.enlarge_scale)
            bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)
            openpose_kps = human_candidates[det_id][1]
            keypoints = openpose_kps.as_posetrack()
            det_dict["track_id"] = self.next_id
            det_dict['det_id'] = det_id
            det_dict['openpose_kps'] = openpose_kps
            det_dict['bbox'] = bbox_det
            det_dict['keypoints'] = keypoints

            self.next_id += 1
            tracked_dets_list.append(det_dict)
            
        self.dets_list_q.append(tracked_dets_list)

    def traverse_each_prev_frame(self, human_candidates, dets_list_prev_frame, tracked_dets_list, tracked_dets_ids, untracked_dets_ids):
        # first travese all bbox candidates
        print("untracked_dets_ids", untracked_dets_ids)
        for det_id in untracked_dets_ids:
            bbox_det = human_candidates[det_id][0]
            bbox_x1y1x2y2 = xywh_to_x1y1x2y2(bbox_det)
            bbox_in_xywh = enlarge_bbox(bbox_x1y1x2y2, self.enlarge_scale)
            bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)
            openpose_kps = human_candidates[det_id][1]
            keypoints = openpose_kps.as_posetrack()
            det_dict = {"img_id":self.img_id,
                                "det_id":det_id,
                                "bbox":bbox_det,
                                "track_id": -1,
                                "openpose_kps": openpose_kps,
                                "keypoints":keypoints}

            track_id, match_index = get_track_id_SpatialConsistency(bbox_det, dets_list_prev_frame)
            print("det", det_id, track_id, match_index)
            if track_id != -1:  # if candidate from prev frame matched, prevent it from matching another
                del dets_list_prev_frame[match_index]
                det_dict['track_id'] = track_id
                tracked_dets_list.append(det_dict)
                tracked_dets_ids.append(det_id)
                continue 

        untracked_dets_ids = list(set(untracked_dets_ids)-set(tracked_dets_ids))
        # second travese all pose candidates
        for det_id in untracked_dets_ids:
            bbox_det = human_candidates[det_id][0]
            bbox_x1y1x2y2 = xywh_to_x1y1x2y2(bbox_det)
            bbox_in_xywh = enlarge_bbox(bbox_x1y1x2y2, self.enlarge_scale)
            bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)
            openpose_kps = human_candidates[det_id][1]
            keypoints = openpose_kps.as_posetrack()
            det_dict = {"img_id":self.img_id,
                                "det_id":det_id,
                                "bbox":bbox_det,
                                "track_id": -1,
                                "openpose_kps": openpose_kps,
                                "keypoints":keypoints}
            track_id, match_index, score = get_track_id_SGCN_plus(
                self.pose_matcher,
                det_dict,
                dets_list_prev_frame,
                pose_matching_threshold=0.4
            )
            if track_id != -1:  # if candidate from prev frame matched, prevent it from matching another
                print("match score is" , score)
                del dets_list_prev_frame[match_index]
                det_dict["track_id"] = track_id
                tracked_dets_list.append(det_dict)
                tracked_dets_ids.append(det_id)
                continue
            

    def first_frame(self, candidates):
        dets_list = []
        for i in range(len(candidates)):
            candidate = candidates[i]
            bbox = candidate[0]
            openpose_kps = candidate[1] 
            keypoints = openpose_kps.as_posetrack()
            det_dict = {"img_id":self.img_id,
                                "det_id":  i,
                                "track_id": self.next_id,
                                "bbox": bbox,
                                "openpose_kps": openpose_kps,
                                "keypoints": keypoints}
            dets_list.append(det_dict)
            self.next_id += 1
        self.dets_list_q.append(dets_list)        


    def get_human_bbox_and_keypoints(self, kps):
        human_candidates = []
        for kpt_item in kps:
            all = kpt_item.all()
            kpt_score = self.get_total_score_from_kpt(all)
            if kpt_score < 5:
                continue
            bbox = get_bbox_from_keypoints(all)
            human_candidate = [bbox, kpt_item]
            human_candidates.append(human_candidate)

        return human_candidates

    def get_total_score_from_kpt(self, kpt_item):
        scores = np.sum(kpt_item[...,[2]])
        return scores


def get_bbox_from_keypoints(keypoints_python_data, enlarge_scale=0.2):
    if keypoints_python_data == [] or keypoints_python_data == 45*[0]:
        return [0, 0, 2, 2]
    x_list = []
    y_list = []
    for keypoint in keypoints_python_data:
        x, y, vis = keypoint
        if vis != 0 and vis != 3:
            x_list.append(x)
            y_list.append(y)
    min_x = min(x_list)
    min_y = min(y_list)
    max_x = max(x_list)
    max_y = max(y_list)

    if not x_list or not y_list:
        return [0, 0, 2, 2]

    #  min_y = min_y - 0.05 * (max_y - min_y)
    scale = enlarge_scale # enlarge bbox by 20% with same center position
    bbox = enlarge_bbox([min_x, min_y, max_x, max_y], scale*1)
    bbox_in_xywh = x1y1x2y2_to_xywh(bbox)
    return bbox_in_xywh


def x1y1x2y2_to_xywh(det):
    x1, y1, x2, y2 = det
    w, h = int(x2) - int(x1), int(y2) - int(y1)
    return [x1, y1, w, h]


def xywh_to_x1y1x2y2(det):
    x1, y1, w, h = det
    x2, y2 = x1 + w, y1 + h
    return [x1, y1, x2, y2]


def enlarge_bbox(bbox, scale):
    assert(scale > 0)
    min_x, min_y, max_x, max_y = bbox
    margin_x = int(0.5 * scale * (max_x - min_x))
    margin_y = int(0.5 * scale * (max_y - min_y))
    if margin_x < 0: margin_x = 2
    if margin_y < 0: margin_y = 2

    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    width = max_x - min_x
    height = max_y - min_y
    if max_y < 0 or max_x < 0 or width <= 0 or height <= 0 or width > 2000 or height > 2000:
        min_x=0
        max_x=2
        min_y=0
        max_y=2

    bbox_enlarged = [min_x, min_y, max_x, max_y]
    return bbox_enlarged


def get_track_id_SGCN_plus(pose_matcher, dets_cur_frame, dets_list_prev_frame, pose_matching_threshold=0.5):
    min_index = None
    min_matching_score = sys.maxsize
    bbox_cur_frame = dets_cur_frame['bbox']
    keypoints_cur_frame = dets_cur_frame['keypoints']
    track_id = -1
    for det_index, det_dict in enumerate(dets_list_prev_frame):
        bbox_prev_frame = det_dict["bbox"]
        keypoints_prev_frame = det_dict["keypoints"]
        pose_matching_score = get_pose_matching_score(pose_matcher, keypoints_cur_frame, keypoints_prev_frame, bbox_cur_frame, bbox_prev_frame)
        pprint([
            keypoints_cur_frame, keypoints_prev_frame, bbox_cur_frame, bbox_prev_frame
        ])
        print("pose_matching_score", pose_matching_score)

        if pose_matching_score <= pose_matching_threshold and pose_matching_score <= min_matching_score:
            # match the target based on the pose matching score
            min_matching_score = pose_matching_score
            min_index = det_index

    if min_index is None:
        return -1, None, 0
    else:
        track_id = dets_list_prev_frame[min_index]["track_id"]
        print("match score is ", min_matching_score)
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


def bbox_invalid(bbox):
    if bbox == [0, 0, 2, 2]:
        return True
    if bbox[2] <= 0 or bbox[3] <= 0 or bbox[2] > 2000 or bbox[3] > 2000:
        return True
    return False


def get_track_id_SpatialConsistency(bbox_cur_frame, bbox_list_prev_frame):
    thresh = 0.3
    max_iou_score = 0
    max_index = -1

    for bbox_index, bbox_det_dict in enumerate(bbox_list_prev_frame):
        bbox_prev_frame = bbox_det_dict["bbox"]

        boxA = xywh_to_x1y1x2y2(bbox_cur_frame)
        boxB = xywh_to_x1y1x2y2(bbox_prev_frame)
        iou_score = iou(boxA, boxB)
        if iou_score > max_iou_score:
            max_iou_score = iou_score
            max_index = bbox_index

    if max_iou_score > thresh:
        track_id = bbox_list_prev_frame[max_index]["track_id"]
        return track_id, max_index
    else:
        return -1, None #未匹配成功


def iou(boxA, boxB):
    # box: (x1, y1, x2, y2)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def keypoints_to_graph(keypoints, bbox):
    print("keypoints_to_graph", keypoints, bbox)
    num_elements = len(keypoints)
    num_keypoints = num_elements/3
    assert(num_keypoints == 15)

    x0, y0, w, h = bbox
    flag_pass_check = True

    graph = 15*[(0, 0)]
    for id in range(15):
        x = keypoints[3*id] - x0
        y = keypoints[3*id+1] - y0
        score = keypoints[3*id+2]

        graph[id] = (int(x), int(y))
    return graph, flag_pass_check


def graph_pair_to_data(sample_graph_pair):
    data_numpy_pair = []
    for siamese_id in range(2):
        # fill data_numpy
        data_numpy = np.zeros((2, 1, 15, 1))

        pose = sample_graph_pair[:][siamese_id]
        data_numpy[0, 0, :, 0] = [x[0] for x in pose]
        data_numpy[1, 0, :, 0] = [x[1] for x in pose]
        data_numpy_pair.append(data_numpy)
    return data_numpy_pair[0], data_numpy_pair[1]
