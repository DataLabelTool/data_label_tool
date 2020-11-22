import numpy as np
import json

import time
import random
import logging
from datetime import datetime


def create_filename():
    return f"{time.time()}_{random.randint(100, 50000)}.png"


def imread(url, proxy_url=None, timeout=30, read_timeout=120):
    """
    load image with specified user_agent
    :param url:
    :param proxy_url: url of proxy to use, example: https://ip.port
    :param timeout: connection timeout
    :param read_timeout: read timeout
    :return: np.ndarray image
    """
    import numpy as np
    import requests
    from PIL import Image
    import io

    if proxy_url is not None:
        proxy_url = {
            proxy_url.split(':')[0]: proxy_url
        }

    r = requests.get(
        url,
        proxies=proxy_url,
        headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36'
        },
        timeout=(timeout, read_timeout)
    )
    image = Image.open(
        io.BytesIO(
            r.content
        )
    )
    mode_to_bpp = {"1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32, "CMYK": 32, "YCbCr": 24, "LAB": 24, "HSV": 24,
                   "I": 32, "F": 32}

    if image.mode == "P":
        image = image.convert("L")
    return np.array(image)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types
    dump:
    >>>data = np.array([10,15,16])
    >>>payload = {"a": data}
    >>>dumped = json.dumps(payload, cls=NumpyEncoder)
    >>>with open(path, 'w') as f:
    >>>    json.dump(dumped, f)

    restore:
    >>>with open(path, 'r') as f:
    >>>    data = json.load(f)
    >>>data["a"] = np.asarray("a")
    """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def timeit(func, min_ms=0):
    """

    :param func: wrapping function
    :param min_ms: min time in ms for logging
    :return:
    """
    def timeit_func(*func_args, **func_kwargs):
        if (len(func_args) > 0 and
                hasattr(func_args[0], '__class__') and
                hasattr(func_args[0].__class__, '__name__')
        ):
            class_name = func_args[0].__class__.__name__
        else:
            class_name = ''
        start_time = datetime.now()
        result = func(*func_args, **func_kwargs)
        end_time = (datetime.now() - start_time).microseconds / 1000
        if end_time > min_ms:
            logging.debug(f"{func.__module__}.{class_name}.{func.__name__}(): time: {end_time} ms")
        return result

    return timeit_func


def except_decorator(func):
    def execp_func(*func_args, **func_kwargs):
        result = None
        try:
            result = func(*func_args, **func_kwargs)
        except (AssertionError, ValueError, Exception) as e:
            if (len(func_args) > 0 and
                    hasattr(func_args[0], '__class__') and
                    hasattr(func_args[0].__class__,'__name__')
            ):
                class_name = func_args[0].__class__.__name__
            else:
                class_name = ''
            logging.error(f"{func.__module__}.{class_name}.{func.__name__}(): \n {e}")
        return result

    return execp_func

# не подходит никуда
from src.utils.bbox import change_bbox_type, scale_list, check_point_in_bbox
# from src.utils.mask import check_point_in_mask
from PIL import Image
from PIL import ImageDraw


def draw_point_on_item(item, point, pen_width, fill=255):

    # get coordinates of new point
    point_bbox = [
        int(point[0] - pen_width / 2),
        int(point[1] - pen_width / 2),
        int(point[0] + pen_width / 2),
        int(point[1] + pen_width / 2)
    ]
    # get new mask image
    if item.have_mask_data() and item.have_bbox():  # item have mask
        mask = item.get_mask()
        item_bbox = item.get_bbox(bbox_type='xyxy')

        if check_point_in_bbox(point, item_bbox, pen_width / 2):
            # make ref coordinates
            ref_point_bbox = [
                point_bbox[0] - item_bbox[0],
                point_bbox[1] - item_bbox[1],
                point_bbox[2] - item_bbox[0],
                point_bbox[3] - item_bbox[1],
            ]
        else:
            # calc new bbox
            new_bbox = [
                min(point_bbox[0], item_bbox[0]),
                min(point_bbox[1], item_bbox[1]),
                max(point_bbox[2], item_bbox[2]),
                max(point_bbox[3], item_bbox[3]),
            ]
            # make new mask
            new_mask = np.zeros(
                [
                    new_bbox[3] - new_bbox[1],
                    new_bbox[2] - new_bbox[0],
                ],
                dtype=np.uint8
            )
            # calc rel coordinates of old mask in new mask
            rel_bbox = [
                item_bbox[0] - new_bbox[0],
                item_bbox[1] - new_bbox[1],
                item_bbox[2] - new_bbox[0],
                item_bbox[3] - new_bbox[1]
            ]
            # place old mask to new mask
            new_mask[rel_bbox[1]:rel_bbox[3], rel_bbox[0]:rel_bbox[2]] = mask
            mask = new_mask

            item.set_bbox(new_bbox, bbox_type='xyxy')
            # make ref coordinates
            ref_point_bbox = [
                point_bbox[0] - new_bbox[0],
                point_bbox[1] - new_bbox[1],
                point_bbox[2] - new_bbox[0],
                point_bbox[3] - new_bbox[1],
            ]

    else:  # item does not have mask, create new
        item.set_bbox(point_bbox, bbox_type='xyxy')
        w = point_bbox[2] - point_bbox[0]
        h = point_bbox[3] - point_bbox[1]
        mask = np.zeros([w, h], dtype=np.uint8)

        # make ref coordinates
        ref_point_bbox = [0, 0, w, h]

    # draw point on mask
    pil_image = Image.fromarray(mask)
    draw = ImageDraw.Draw(pil_image)
    draw.ellipse(ref_point_bbox, fill=fill)
    mask = np.asarray(pil_image)

    item.set_mask(mask)


def get_bbox_from_mask(mask):
    if np.count_nonzero(mask) > 0:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return cmin, rmin, cmax, rmax
    else:
        return 0, 0, 0, 0


def clip_item_mask(item):
    """
    clip zero paddings on mask (can del mask if it is empty)
    :param item:
    :return:
    """
    # get new mask image
    if item.have_mask_data() and item.have_bbox():  # item have mask
        mask = item.get_mask()
        item_bbox = item.get_bbox(bbox_type='xyxy')
        mask_bbox = get_bbox_from_mask(mask)

        # print('mask shape', mask.shape)
        # print('item_bbox', item_bbox)
        # print('mask_bbox', mask_bbox)
        if sum(mask_bbox) > 0:
            cliped_item_bbox = [
                item_bbox[0] + mask_bbox[0],
                item_bbox[1] + mask_bbox[1],
                item_bbox[0] + (mask_bbox[2] + 1),
                item_bbox[1] + (mask_bbox[3] + 1),
            ]
            # print('cliped_item_bbox', cliped_item_bbox)
            item.set_bbox(cliped_item_bbox, bbox_type='xyxy')
            cliped_mask = mask[
                          mask_bbox[1]:(mask_bbox[3] + 1),
                          mask_bbox[0]:(mask_bbox[2] + 1)
                          ]
            item.set_mask(cliped_mask)
        else:
            item.del_bbox()
            item.del_mask()
