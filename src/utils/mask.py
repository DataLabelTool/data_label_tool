import os
import numpy as np
import skimage.io as skio
from PIL import Image
from PIL import ImageDraw
from skimage import measure
import io
import base64

from src.utils.utils import create_filename
from src.utils.bbox import validate_bbox


# mask to path
def mask_2_path(mask: np.ndarray, base_path: str, filename=None) -> str:
    mask = mask.copy()
    mask[mask > 0] = 255
    if filename is None:
        filename = create_filename()
    path = os.path.join(base_path, filename)
    skio.imsave(path, mask)
    return filename


def path_2_mask(path: str) -> np.ndarray:
    mask = skio.imread(path)
    mask[mask > 0] = 1
    return mask


# mask to base 64
def mask_to_base64(mask: np.ndarray) -> str:
    """
    Converts binary mask to base64 representation
    :param mask:
    :return:
    """
    mask = mask.copy()
    mask[mask > 0] = 255
    img = Image.fromarray(mask)
    img = img.convert("L")

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(
        buffered.getvalue()
    ).decode("utf-8")
    return img_str


def base64_to_mask(mask: str) -> np.ndarray:
    """

    :param mask:
    :return:
    """
    img = Image.open(
        io.BytesIO(
            base64.b64decode(mask)
        )
    )
    mask = np.array(img)
    mask[mask > 0] = 1
    return mask


# image to polygon
def mask_to_polygon(binary_mask: np.ndarray, tolerance=0) -> str:
    """
    Converts a binary mask to COCO polygon representation
    :param binary_mask: a 2D binary numpy array where '1's represent the object
    :param tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    :return: string with encoded "width; height; polygon; polygon; ..." (ploygon is a string of y, x, y, x coordinates)
    """
    def close_contour(cnt):
        if not np.array_equal(cnt[0], cnt[-1]):
            cnt = np.vstack((cnt, cnt[0]))
        return cnt

    def polygons2string(polygons, width, height):
        polystring = '; '.join([
            ' '.join([str(x) for x in polygon])
            for polygon in polygons
        ])
        return '; '.join([str(width), str(height), polystring])

    polygons = []

    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance=tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    return polygons2string(polygons, width=binary_mask.shape[1], height=binary_mask.shape[0])


def polygon_to_mask(polystring: str) -> np.ndarray:
    """

    :param polystring: encoded polygons "width; height; polygon; polygon; ..."
    :return: numpy mask of 1 and 0
    """
    def string2polygon(polystring):
        pp = polystring.split(';')
        w = int(pp[0])
        h = int(pp[1])
        polygon = [
            [float(x) for x in poly.split()]
            for poly in pp[2:]
        ]
        return polygon, w, h
    polygon, width, height = string2polygon(polystring)
    img = Image.new('L', (width, height), 0)
    for poly in polygon:
        ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
    return np.array(img)


# image to rle
def mask_to_rle(binary_mask: np.ndarray) -> str:
    """
    Run-Length Encode and Decode
    ref.: https://www.kaggle.com/stainsby/fast-tested-rle
    :param binary_mask: a 2D binary numpy array where '1's represent the object
    :return:
    """
    def rle2string(runs: np.ndarray, width: int, height: int):
        encoded = ' '.join(str(x) for x in runs)
        return '; '.join([str(width), str(height), encoded])

    pixels = binary_mask.flatten()

    pixels = np.where(pixels > 0.5, 1, 0).astype(np.uint8)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return rle2string(runs, width=binary_mask.shape[1], height=binary_mask.shape[0])


def rle_to_mask(mask_rle: str) -> np.ndarray:
    """

    :param mask_rle: run-length as string formated (start length)
    :param width: width of image
    :param height: height of image
    :return: numpy 2d array, 1 - mask, 0 - background
    """
    def string2rle(rlestring: str):
        pp = rlestring.split(';')
        w = int(pp[0])
        h = int(pp[1])
        rleencoded = pp[2]
        return rleencoded, w, h

    rleencoded, width, height = string2rle(mask_rle)
    shape = (height, width)
    s = rleencoded.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def change_mask_type(mask, mask_type: str, **kwargs):
    """
    Change mask type between:
    "numpy", "polygon", "rle", "base64", "path"

    if convert from/to "path" type you must specify "base_path" arg and "filename" if you want

    :param mask:
    :param mask_type:
    :return:
    """
    supported_type = [
        "numpy2numpy", "numpy2polygon", "numpy2rle", "numpy2base64", "numpy2path",
        "polygon2numpy", "polygon2polygon", "polygon2rle", "polygon2base64", "polygon2path",
        "poly2numpy", "poly2polygon", "poly2rle", "poly2base64", "poly2path",
        "numpy2poly", "polygon2poly", "rle2poly", "base642poly", "path2poly",
        "rle2numpy", "rle2polygon", "rle2rle", "rle2base64", "rle2path",
        "base642numpy", "base642polygon", "base642rle", "base642base64", "base642path",
        "path2numpy", "path2polygon", "path2rle", "path2base64", "path2path",
    ]
    assert mask_type in supported_type, f"mask_type '{mask_type}' not supported"

    if mask_type == "numpy2polygon" or mask_type == "numpy2poly":
        mask = mask_to_polygon(mask, tolerance=kwargs.get('tolerance', 0))
    elif mask_type == "numpy2rle":
        mask = mask_to_rle(mask)
    elif mask_type == "numpy2base64":
        mask = mask_to_base64(mask)
    elif mask_type == "numpy2path":
        assert "base_path" in kwargs, f"you must add 'base_path' arg to convert {mask_type}"
        mask = mask_2_path(mask=mask, base_path=kwargs["base_path"], filename=kwargs.get("filename", None))

    elif mask_type == "polygon2numpy" or mask_type == "poly2numpy":
        mask = polygon_to_mask(mask)
    elif mask_type == "polygon2rle" or mask_type == "poly2rle":
        mask = polygon_to_mask(mask)
        mask = mask_to_rle(mask)
    elif mask_type == "polygon2base64" or mask_type == "poly2base64":
        mask = polygon_to_mask(mask)
        mask = mask_to_base64(mask)
    elif mask_type == "polygon2path" or mask_type == "poly2path":
        assert "base_path" in kwargs, f"you must add 'base_path' arg to convert {mask_type}"
        mask = polygon_to_mask(mask)
        mask = mask_2_path(mask=mask, base_path=kwargs["base_path"], filename=kwargs.get("filename", None))

    elif mask_type == "rle2numpy":
        mask = rle_to_mask(mask)
    elif mask_type == "rle2polygon" or mask_type == "rle2poly":
        mask = rle_to_mask(mask)
        mask = mask_to_polygon(mask)
    elif mask_type == "rle2base64":
        mask = rle_to_mask(mask)
        mask = mask_to_base64(mask)
    elif mask_type == "rle2path":
        assert "base_path" in kwargs, f"you must add 'base_path' arg to convert {mask_type}"
        mask = rle_to_mask(mask)
        mask = mask_2_path(mask=mask, base_path=kwargs["base_path"], filename=kwargs.get("filename", None))

    elif mask_type == "base642numpy":
        mask = base64_to_mask(mask)
    elif mask_type == "base642polygon" or mask_type == "base642poly":
        mask = base64_to_mask(mask)
        mask = mask_to_polygon(mask)
    elif mask_type == "base642rle":
        mask = base64_to_mask(mask)
        mask = mask_to_rle(mask)
    elif mask_type == "base642path":
        assert "base_path" in kwargs, f"you must add 'base_path' arg to convert {mask_type}"
        mask = base64_to_mask(mask)
        mask = mask_2_path(mask=mask, base_path=kwargs["base_path"], filename=kwargs.get("filename", None))

    elif mask_type == "path2numpy":
        assert "base_path" in kwargs, f"you must add 'base_path' arg to convert {mask_type}"
        path = os.path.join(kwargs["base_path"], mask)
        mask = path_2_mask(path=path)
    elif mask_type == "path2polygon" or mask_type == "path2poly":
        assert "base_path" in kwargs, f"you must add 'base_path' arg to convert {mask_type}"
        path = os.path.join(kwargs["base_path"], mask)
        mask = path_2_mask(path=path)
        mask = mask_to_polygon(mask)
    elif mask_type == "path2rle":
        assert "base_path" in kwargs, f"you must add 'base_path' arg to convert {mask_type}"
        path = os.path.join(kwargs["base_path"], mask)
        mask = path_2_mask(path=path)
        mask = mask_to_rle(mask)
    elif mask_type == "path2base64":
        assert "base_path" in kwargs, f"you must add 'base_path' arg to convert {mask_type}"
        path = os.path.join(kwargs["base_path"], mask)
        mask = path_2_mask(path=path)
        mask = mask_to_base64(mask)

    return mask


def validate_mask(mask: np.ndarray):
    assert len(mask.shape) >= 2, "mask shape < 2"
    assert mask.shape[0] > 0 and mask.shape[1] > 0, "mask shape have zero dimensions"


def validate_mask_and_bbox(mask: np.ndarray, bbox, image_w=None, image_h=None, bbox_type='xyxy'):
    validate_bbox(bbox, image_w=image_w, image_h=image_h, bbox_type=bbox_type)
    validate_mask(mask)

    if bbox_type == 'xyxy':
        w, h = int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])
    if bbox_type == 'xywh':
        w, h = int(bbox[2]), int(bbox[3])

    assert len(mask.shape) == 2, \
        f"mask shape should be two dimensional, got: {len(mask.shape)}"

    assert mask.shape[0] == h and mask.shape[1] == w, \
        f"mask shape should match to bbox shape, got: " + \
        f"mask shape {mask.shape}, bbox shape [{h}, {w}]"


def check_point_in_mask(point, mask: np.ndarray) -> bool:
    """
    :param point: (x, y) - relative coords in mask
    :param mask:
    :return:
    """
    if mask[int(point[1]), int(point[0])] > 0:
        return True
    else:
        return False


def check_circle_in_mask(point, pen_width: int, mask: np.ndarray) -> bool:
    """
    :param point: (x, y) - relative coords in mask
    :param mask:
    :return:
    """
    point = [int(point[0]), int(point[1])]

    if point[0] < mask.shape[1] and point[1] < mask.shape[0] and mask[point[1], point[0]] > 0:
        point_bbox = [
            int(point[0] - pen_width / 2),
            int(point[1] - pen_width / 2),
            int(point[0] + pen_width / 2),
            int(point[1] + pen_width / 2)
        ]
        point_mask = np.zeros(
            [
                point_bbox[3] - point_bbox[1],
                point_bbox[2] - point_bbox[0]
            ],
            dtype=np.uint8
        )
        # draw point on mask
        pil_image = Image.fromarray(point_mask)
        draw = ImageDraw.Draw(pil_image)
        draw.ellipse([0, 0, pen_width, pen_width], fill=1)
        point_mask = np.asarray(pil_image)

        crop_bbox = [0, 0, pen_width, pen_width]
        if point_bbox[0] < 0:
            crop_bbox[0] = abs(point_bbox[0])
            point_bbox[0] = 0

        if point_bbox[1] < 0:
            crop_bbox[1] = abs(point_bbox[1])
            point_bbox[1] = 0

        if point_bbox[2] > mask.shape[1]:
            crop_bbox[2] = mask.shape[1] - point_bbox[2]
            point_bbox[2] = mask.shape[1]

        if point_bbox[3] > mask.shape[0]:
            crop_bbox[3] = mask.shape[0] - point_bbox[3]
            point_bbox[3] = mask.shape[0]

        intersection = np.logical_and(
            mask[point_bbox[1]:point_bbox[3], point_bbox[0]:point_bbox[2]],
            point_mask[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]]
        )
        if intersection.sum() > 0:
            return True
        else:
            return False
    else:
        return False
