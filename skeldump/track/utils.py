import numpy as np


def keypoints_to_graph(keypoints, bbox):
    num_elements = len(keypoints)
    num_keypoints = num_elements / 3
    assert num_keypoints == 15

    x0, y0, w, h = bbox
    flag_pass_check = True

    graph = 15 * [(0, 0)]
    for id in range(15):
        x = keypoints[3 * id] - x0
        y = keypoints[3 * id + 1] - y0

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
