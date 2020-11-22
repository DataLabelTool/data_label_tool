from typing import List, Union, Tuple
import numpy as np

typing_bbox = Union[List[Union[int, float]], Tuple[Union[int, float]], np.ndarray]


def change_bbox_type(bbox: typing_bbox, bbox_type: str):
    """Change box type between
    (xmin, ymin, xmax, ymax),
    (ymin, xmin, ymax, xmax),
    (xmin, ymin, width, height),
    (xcenter, ycenter, width, height).

    :param bbox: bbox
    :param bbox_type: (str) either 'xyxy2xywh', 'xywh2xyxy' or 'cxcywh2xyxy', 'xyxy2cxcywh'.
    """
    supported_type = [
        'xyxy2xyxy', 'xyxy2xywh', 'xyxy2cxcywh', 'xyxy2yxyx',  # xyxy
        'xywh2xywh',  'xywh2xyxy', 'xywh2cxcywh', 'xywh2yxyx',  # xywh
        'cxcywh2cxcywh', 'cxcywh2xyxy',  'cxcywh2xywh', 'cxcywh2yxyx',  # cxcywh
        'yxyx2yxyx', 'yxyx2xyxy', 'yxyx2xywh', 'yxyx2cxcywh',  # yxyx
    ]
    assert bbox_type in supported_type, f"bbox_type '{bbox_type}' not supported"

    if isinstance(bbox, (list, tuple)):
        return_type = "list"
    elif isinstance(bbox, np.ndarray):
        return_type = "ndarray"
    else:
        raise AssertionError(f"change_bbox_order: can't work with bbox type: {type(bbox)}")

    bbox = np.array(bbox)
    use_squeeze = True if bbox.ndim == 1 else False
    assert bbox.shape[-1] == 4, \
        f"bbox last dimension must be equal to 4, got: {bbox.shape[-1]}"
    bbox = bbox.reshape(-1, 4)

    a = bbox[:, :2]
    b = bbox[:, 2:]

    if bbox_type == 'xyxy2xywh':
        bbox = np.concatenate([a, b - a], 1)
    elif bbox_type == 'xyxy2cxcywh':
        bbox = np.concatenate([(a + b) / 2, b - a], 1)
    elif bbox_type == 'xyxy2yxyx':
        bbox = np.stack([bbox[:, 1], bbox[:, 0], bbox[:, 3], bbox[:, 2]]).transpose()
    elif bbox_type == 'xywh2xyxy':
        bbox = np.concatenate([a, a + b], 1)
    elif bbox_type == 'xywh2cxcywh':
        bbox = np.concatenate([a + b / 2, b], 1)
    elif bbox_type == 'xywh2yxyx':
        bbox = np.concatenate([a, a + b], 1)
        bbox = np.stack([bbox[:, 1], bbox[:, 0], bbox[:, 3], bbox[:, 2]]).transpose()
    elif bbox_type == 'cxcywh2yxyx':
        bbox = np.concatenate([a - b / 2, a + b / 2], 1)
        bbox = np.stack([bbox[:, 1], bbox[:, 0], bbox[:, 3], bbox[:, 2]]).transpose()
    elif bbox_type == 'cxcywh2xyxy':
        bbox = np.concatenate([a - b / 2, a + b / 2], 1)
    elif bbox_type == 'cxcywh2xywh':
        bbox = np.concatenate([a - b / 2, b], 1)
    elif bbox_type == 'yxyx2xyxy':
        bbox = np.stack([bbox[:, 1], bbox[:, 0], bbox[:, 3], bbox[:, 2]]).transpose()
    elif bbox_type == 'yxyx2xywh':
        bbox = np.stack([bbox[:, 1], bbox[:, 0], bbox[:, 3], bbox[:, 2]]).transpose()
        a = bbox[:, :2]
        b = bbox[:, 2:]
        bbox = np.concatenate([a, b - a], 1)
    elif bbox_type == 'yxyx2cxcywh':
        bbox = np.stack([bbox[:, 1], bbox[:, 0], bbox[:, 3], bbox[:, 2]]).transpose()
        a = bbox[:, :2]
        b = bbox[:, 2:]
        bbox = np.concatenate([a + b / 2, b], 1)

    if use_squeeze:
        bbox = bbox.squeeze()

    if return_type == "list":
        return bbox.tolist()
    elif return_type == "ndarray":
        return bbox


def validate_point(point: typing_bbox, image_w: int = None, image_h: int = None):
    assert len(point) == 2, f"point should be list of 2 elements, got: point: {point}"
    assert point[0] >= 0 and point[1] >= 0, f"point coordinates must be >= 0, got: point: {point}"
    if image_w is not None:
        assert point[0] <= image_w, f"point coordinates must be <= image_w, got: point[0] {point[0]}, image_w: {image_w}"
    if image_h is not None:
        assert point[1] <= image_h, f"point coordinates must be <= image_h, got: point[1] {point[1]}, image_h: {image_h}"


def validate_bbox(bbox: typing_bbox, image_w=None, image_h=None, bbox_type='xyxy'):

    assert bbox_type in ['xyxy', 'xywh']
    assert len(bbox) == 4, \
        f"bbox should consist 4 elements: [x1, y1, x2, y2], got: {len(bbox)}"

    assert bbox[0] >= 0 and bbox[1] >= 0 and bbox[2] >= 0 and bbox[3] >= 0, \
        "bbox coordinates must be >= 0"

    if bbox_type == 'xyxy':
        assert bbox[2] - bbox[0] > 0 and bbox[3] - bbox[1] > 0, "width and height must be > 0"

        if image_w is not None:
            assert bbox[2] - bbox[0] <= image_w, \
                f"bbox out of image boundaries (w): bbox[2]({bbox[2]}) - bbox[0]({bbox[0]})={bbox[2] - bbox[0]} <= image_w({image_w})"
        if image_h is not None:
            assert bbox[3] - bbox[1] <= image_h, \
                f"bbox out of image boundaries (h) bbox[3]({bbox[3]}) - bbox[1]({bbox[1]})={bbox[3] - bbox[1]} <= image_h({image_h})"

    if bbox_type == 'xywh':
        assert bbox[2] > 0 and bbox[3] > 0, "width and height must be > 0"

        if image_w is not None:
            assert bbox[0] + bbox[2] <= image_w, \
                f"bbox out of image boundaries (w): bbox[2]({bbox[2]}) + bbox[0]({bbox[0]})={bbox[2] + bbox[0]} <= image_w({image_w})"
        if image_h is not None:
            assert bbox[1] + bbox[3] <= image_h, \
                f"bbox out of image boundaries (h): bbox[3]({bbox[3]}) + bbox[1]({bbox[1]})={bbox[3] + bbox[1]} <= image_h({image_h})"


def make_bbox(point1, point2):
    bbox = [0, 0, 0, 0]
    bbox[0] = min(point1[0], point2[0])
    bbox[1] = min(point1[1], point2[1])
    bbox[2] = max(point1[0], point2[0])
    bbox[3] = max(point1[1], point2[1])
    return bbox


def fix_bbox(bbox, w, h, bbox_type='xyxy'):
    """fix bbox"""
    if bbox_type == 'xyxy':
        bbox = [max(0, x) for x in bbox]
        bbox[0] = min(w, bbox[0])
        bbox[2] = min(w, bbox[2])
        bbox[1] = min(h, bbox[1])
        bbox[3] = min(h, bbox[3])
    if bbox_type == 'xywh':
        if bbox[0] < 0:
            bbox[2] -= abs(bbox[0])
            bbox[0] = 0
        if bbox[1] < 0:
            bbox[3] -= abs(bbox[1])
            bbox[1] = 0
        bbox[2] = min(w - bbox[0], bbox[2])
        bbox[3] = min(h - bbox[1], bbox[3])
    return bbox


def fix_bbox_and_mask(bbox, mask: np.ndarray, w, h, bbox_type='xyxy'):
    """

    :param bbox:
    :param mask:
    :param w:
    :param h:
    :param bbox_type:
    :return:
    """
    if bbox_type == 'xyxy':
        if bbox[0] < 0:
            dx = abs(bbox[0])
            bbox[0] = 0
            mask = mask[:, dx:]

        if bbox[1] < 0:
            dy = abs(bbox[1])
            bbox[1] = 0
            mask = mask[dy:, :]

        if bbox[2] > w:
            dx = bbox[2] - w
            bbox[2] = w
            mask = mask[:, :-dx]

        if bbox[3] > h:
            dy = bbox[3] - h
            bbox[3] = h
            mask = mask[:-dy, :]
    if bbox_type == 'xywh':
        if bbox[0] < 0:
            dx = abs(bbox[0])
            bbox[0] = 0
            bbox[2] -= dx
            mask = mask[:, dx:]

        if bbox[1] < 0:
            dy = abs(bbox[1])
            bbox[1] = 0
            bbox[3] -= dy
            mask = mask[dy:, :]

        if bbox[2] > w - bbox[0]:
            bbox[2] = w - bbox[0]
            mask = mask[:, :bbox[2]]

        if bbox[3] > h - bbox[1]:
            bbox[3] = h - bbox[1]
            mask = mask[:, :bbox[3]]
    return bbox, mask


def scale_list(bbox, scale):
    bbox = [x * scale for x in bbox]
    return bbox


def get_bbox_center(bbox):
    """xywh order"""
    return [
        int((bbox[0] + bbox[2] / 2)),
        int((bbox[1] + bbox[3] / 2))
    ]


def check_point_in_bbox(point, bbox, pad=0):
    """
    :param point: [x, y]
    :param bbox: [x1, y1, x2, y2]
    :param pad: pad for point in pixels
    :return:
    """
    if bbox[0] + pad < point[0] < bbox[2] - pad and \
            bbox[1] + pad < point[1] < bbox[3] - pad:
        return True
    else:
        return False
