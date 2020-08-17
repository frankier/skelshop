import io

import cv2
import numpy as np
from wcmatch.glob import GLOBSTAR, globmatch

THRESH = 0.02


def sane_globmatch(path, matchers):
    if len(matchers) == 0:
        return False
    return globmatch(path, matchers, flags=GLOBSTAR)


def has_img_ext(filename):
    return filename.rsplit(".", 1)[-1].lower() in ("jpg", "pgm", "png", "ppm", "tiff")


def get_pads(pad):
    top = pad // 2
    bottom = top + pad % 2
    return top, bottom


def get_square_padding(img):
    top = bottom = left = right = 0
    height, width = img.shape[:2]
    if height < width:
        top, bottom = get_pads(width - height)
    else:
        left, right = get_pads(height - width)
    return top, bottom, left, right


def squarify(img, pad):
    """
    >Cropping the Image for Hand/Face Keypoint Detection
    >If you are using your own hand or face images, you should leave about 10-20% margin between the end of the hand/face and the sides (left, top, right, bottom) of the image. We trained with that configuration, so it should be the ideal one for maximizing detection.
    >We did not use any solid-color-based padding, we simply cropped from the whole image. Thus, if you can, use the image rather than adding a color-based padding. Otherwise black padding should work good.
    """
    top, bottom, left, right = pad
    return cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0)
    )


def put_sprite(sheet, idx_j, idx_i, im, target_size):
    orig_h, orig_w = im.shape[:2]
    if orig_w < orig_h:
        new_size = ((orig_w * target_size) // orig_h, target_size)
    else:
        new_size = (target_size, (orig_h * target_size) // orig_w)

    im = cv2.resize(im, new_size)

    pad = get_square_padding(im)

    square_j_top = idx_j * target_size + pad[0]
    square_i_left = idx_i * target_size + pad[2]
    square_j_bottom = square_j_top + new_size[1]
    square_i_right = square_i_left + new_size[0]

    sheet[square_j_top:square_j_bottom, square_i_left:square_i_right, :3] = im
    sheet[square_j_top:square_j_bottom, square_i_left:square_i_right, 3] = 255


def save_fig_np(fig, dpi=96):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="raw", dpi=dpi)
    dim = np.rint(fig.get_size_inches() * dpi)
    io_buf.seek(0)
    mat = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(dim[0]), int(dim[1]), -1),
    )
    mat = mat[:, :, :3].transpose((2, 1, 0))
    return mat


def is_included(skel, pose):
    for idx, (x, y, c) in enumerate(pose):
        if idx not in skel.kp_idxs:
            continue
        if c < THRESH:
            return False
    return True
