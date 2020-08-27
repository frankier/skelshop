import logging
from typing import Dict, List, Optional, Tuple

import click
import h5py
import numpy
import torch

from embedtrain.litmod import MetGcnLit
from skeldump.io import AsIfOrdered, UnsegmentedReader
from skeldump.utils.geom import x1y1x2y2_to_cxywh

logger = logging.getLogger(__name__)


def get_right_hand(pose_np):
    return numpy.vstack([[pose_np[4]], pose_np[45:65]])


def get_left_hand(pose_np):
    return numpy.vstack([[pose_np[7]], pose_np[25:45]])


def has_all(pose):
    return numpy.all(pose[:, 2] > 0)


class Batcher:
    def __init__(self, batch_size, cb):
        self.batch = []
        self.batch_info = []
        self.batch_size = batch_size
        self.cb = cb

    def feed(self, inp, info):
        self.batch.append(inp)
        self.batch_info.append(info)
        if len(self.batch) >= self.batch_size:
            self.cb(self.batch, self.batch_info)
            self.batch = []
            self.batch_info = []


FrameBuf = Tuple[int, List[numpy.ndarray]]


class ProcessorWriter:
    def __init__(self, model, left_out, right_out):
        self.model = model
        self.left_out = h5py.File(left_out, "w", track_order=True)
        self.right_out = h5py.File(right_out, "w", track_order=True)
        self.mat_buf: Dict[bool, Optional[FrameBuf]] = {False: None, True: None}

    def process_batch(self, batch, batch_info):
        # N, C, T, V, M
        # N = id_within_minibatch (hint: use a DataLoader to make minibatches in the 1st dimension)
        # C = channels (x, y, score) OR (x, y) -- has to match num_channels
        # T = frame_num_aka_time
        # V = keypoint/joint (probably stands for vertex)
        # M = person ID (for when there are multiple people within a frame I would suppose)
        result = self.model(
            torch.stack(
                [
                    torch.from_numpy(vec)
                    .float()
                    .to(self.model.device)
                    .permute(1, 0)[:, None, :, None]
                    for vec in batch
                ]
            )
        )
        for vec, info in zip(result, batch_info):
            self.process_one(vec.detach().cpu().numpy(), info)

    def process_one(self, vec, info):
        frame_idx, pose_idx, is_left = info
        mat_buf = self.mat_buf[is_left]
        if mat_buf is None or mat_buf[0] != frame_idx:
            self.flush_mat_buf(is_left)
            self.mat_buf[is_left] = (frame_idx, [vec])
        else:
            # self.mat_buf[is_left][0] == frame_idx:
            mat_buf[1].append(vec)

    def flush_mat_buf(self, is_left):
        mat_buf = self.mat_buf[is_left]
        if mat_buf is None:
            return
        frame_idx, mat = mat_buf
        if not mat:
            return
        if is_left:
            out = self.left_out
        else:
            out = self.right_out
        out[str(frame_idx)] = numpy.vstack(mat)

    def close(self):
        self.flush_mat_buf(False)
        self.flush_mat_buf(True)
        self.left_out.close()
        self.right_out.close()


def minmax_scale(pose):
    from ufunclab import minmax

    bbox = minmax(pose, axes=[(0,), (1,)])[:2]
    bbox = numpy.transpose(bbox).reshape(-1)
    xywh_bbox = x1y1x2y2_to_cxywh(bbox)
    return (pose - xywh_bbox[:2]) / xywh_bbox[2:]


@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("h5fn", type=click.Path(exists=True))
@click.argument("left_out", type=click.Path())
@click.argument("right_out", type=click.Path())
@click.option("--batch-size", type=int, default=8192)
@click.option("--device", type=str, default="cuda")
def mk_pre_embeds(model_path, h5fn, left_out, right_out, batch_size, device):
    model = MetGcnLit.load_from_checkpoint(model_path)
    model.eval()
    model.to(torch.device(device))
    writer = ProcessorWriter(model, left_out, right_out)
    batcher = Batcher(batch_size, writer.process_batch)

    with h5py.File(h5fn, "r") as h5f:
        reader = AsIfOrdered(UnsegmentedReader(h5f))
        for frame_idx, bundle in enumerate(reader):
            for pose_idx, pose in bundle:
                pose_np = pose.all()
                hand = get_right_hand(pose_np)
                if has_all(hand):
                    batcher.feed(
                        minmax_scale(hand[:, :2]), (frame_idx, pose_idx, False)
                    )
                hand = get_left_hand(pose_np)
                if has_all(hand):
                    batcher.feed(
                        minmax_scale(hand[:, :2]) * [1, -1], (frame_idx, pose_idx, True)
                    )
    writer.close()


if __name__ == "__main__":
    mk_pre_embeds()
