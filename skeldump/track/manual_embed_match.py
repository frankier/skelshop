from skeldump.embed.manual import man_dist


def man_embed_match(det_cur, det_prev):
    return man_dist(det_cur.openpose_kps, det_prev.openpose_kps)
