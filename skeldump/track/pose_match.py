def pose_match_track_id(pose_matcher, det_cur_frame, dets_prev_frame, threshold=0.5):
    min_index = None
    min_matching_score = float("inf")
    for det_index, det_dict in enumerate(dets_prev_frame):
        pose_matching_score = pose_matcher(det_cur_frame, det_dict)

        if (
            pose_matching_score <= threshold
            and pose_matching_score <= min_matching_score
        ):
            # match the target based on the pose matching score
            min_matching_score = pose_matching_score
            min_index = det_index

    if min_index is None:
        return None
    track_id = dets_prev_frame[min_index].track_id
    return track_id, min_index, min_matching_score
