from wcmatch.glob import globmatch
import cv2


def sane_globmatch(path, matchers):
    if len(matchers) == 0:
        return False
    return globmatch(path, matchers)


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


def resize_sq_aspect(im, target_size):
    orig_h, orig_w = im.shape[:2]
    if orig_w < orig_h:
        new_size = (target_size, orig_w * target_size // orig_h)
    else:
        new_size = (orig_h * target_size // orig_w, target_size)

    im = cv2.resize(im, new_size)

    return squarify(im, get_square_padding(im))
