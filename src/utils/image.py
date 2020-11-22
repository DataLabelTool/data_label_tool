import os
import skimage.io as skio

import numpy as np
import io
import base64
from PIL import Image
from PIL import ImageDraw
from skimage.filters import threshold_otsu
from skimage import morphology
from src.utils.utils import create_filename, imread


# image to base 64
def image_to_base64(image: np.ndarray, mode: str = None) -> str:
    if mode is None:
        mode = "RGB"
    img = Image.fromarray(image)
    img = img.convert(mode)

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(
        buffered.getvalue()
    ).decode("utf-8")
    return img_str


def base64_to_image(image: str) -> np.ndarray:
    img = Image.open(
        io.BytesIO(
            base64.b64decode(image)
        )
    )
    return np.array(img)


def image_to_url(image: np.ndarray) -> str:
    raise NotImplementedError("can't convert image to url")


def url_to_image(path: str) -> np.ndarray:
    try:
        image = imread(path)
    except Exception:
        return None
    # image[image > 0] = 255
    return image


def path_to_image(path: str) -> np.ndarray:
    image = skio.imread(path)
    return image


def image_to_path(image: np.ndarray, base_path: str, filename=None) -> str:
    if filename is None:
        filename = create_filename()
    path = os.path.join(base_path, filename)
    skio.imsave(path, image)
    return filename

# def open_cached_image(data_path, url, unique_name=False):
#     if unique_name:
#         name = str(time.time()) + '_'.join(url.split('/')[-2:])
#     else:
#         name = '_'.join(url.split('/')[-2:])
#
#     image_path = os.path.join(data_path, 'images', name)
#     if os.path.exists(image_path):
#         image = skio.imread(image_path)
#         if len(image.shape) == 3:
#             image = image[:, :, 0]
#         # print(image.shape)
#         # image = image[:, :, np.newaxis]
#         # image = np.concatenate([image, image, image, image], axis=2)
#         # print(image.shape)
#     else:
#         image = skio.imread(url)
#         image[image > 0] = 255
#         if len(image.shape) == 3:
#             image = image[:, :, 0]
#         skio.imsave(image_path, image)
#
#     if unique_name:
#         return image, name
#     else:
#         return image


def change_image_type(image, image_type: str, **kwargs):
    """
    Change image type between:
    "numpy", "base64", "url", "path"
    :param image:
    :param image_type:
    :return:
    """
    supported_type = [
        "numpy2numpy", "numpy2base64", "numpy2path", "numpy2url",
        "base642numpy", "base642base64", "base642path",  "base642url",
        "url2numpy", "url2base64", "url2path", "url2url",
        "path2numpy", "path2base64", "path2path",  "path2url",
    ]
    assert image_type in supported_type, f"image_type '{image_type}' not supported"

    if image_type == "numpy2base64":
        image = image_to_base64(image=image, mode=kwargs.get("mode", None))
    elif image_type == "numpy2path":
        assert "base_path" in kwargs, f"you must add 'base_path' arg to convert {image_type}"
        image = image_to_path(image=image, base_path=kwargs["base_path"], filename=kwargs.get("filename", None))
    elif image_type == "numpy2url":
        image = image_to_url(image=image)

    elif image_type == "base642numpy":
        image = base64_to_image(image=image)
    elif image_type == "base642path":
        assert "base_path" in kwargs, f"you must add 'base_path' arg to convert {image_type}"
        image = base64_to_image(image=image)
        image = image_to_path(image=image, base_path=kwargs["base_path"], filename=kwargs.get("filename", None))
    elif image_type == "base642url":
        image = base64_to_image(image=image)
        image = image_to_url(image=image)

    elif image_type == "url2numpy":
        image = url_to_image(image)
    elif image_type == "url2base64":
        image = url_to_image(image)
        image = image_to_base64(image, mode=kwargs.get("mode", None))
    elif image_type == "url2path":
        assert "base_path" in kwargs, f"you must add 'base_path' arg to convert {image_type}"
        image = url_to_image(image)
        image = image_to_path(image=image, base_path=kwargs["base_path"], filename=kwargs.get("filename", None))

    elif image_type == "path2numpy":
        assert "base_path" in kwargs, f"you must add 'base_path' arg to convert {image_type}"
        path = os.path.join(kwargs["base_path"], image)
        image = path_to_image(path=path)
    elif image_type == "path2base64":
        assert "base_path" in kwargs, f"you must add 'base_path' arg to convert {image_type}"
        path = os.path.join(kwargs["base_path"], image)
        image = path_to_image(path=path)
        image = image_to_base64(image, mode=kwargs.get("mode", None))
    elif image_type == "path2url":
        assert "base_path" in kwargs, f"you must add 'base_path' arg to convert {image_type}"
        path = os.path.join(kwargs["base_path"], image)
        image = path_to_image(path=path)
        image = image_to_url(image=image)

    return image


def draw_point_on_mask(mask, point, pen_width, fill=255):
    pil_image = Image.fromarray(mask)
    draw = ImageDraw.Draw(pil_image)
    x1 = int(point[0] - pen_width / 2)
    y1 = int(point[1] - pen_width / 2)
    x2 = int(point[0] + pen_width / 2)
    y2 = int(point[1] + pen_width / 2)

    draw.ellipse([x1, y1, x2, y2], fill=fill)
    mask = np.asarray(pil_image)
    return mask

def image2gray(image: np.ndarray) -> np.ndarray:

    if image.ndim == 3:
        if image.shape[2] == 4:
            gray = 0.2989 * image[:,:,1] + 0.5870 * image[:,:,2] + 0.1140 * image[:,:,3]
            return gray
        elif image.shape[2] == 3:
            gray = 0.2989 * image[:,:,0] + 0.5870 * image[:,:,1] + 0.1140 * image[:,:,2]
            return gray
        elif image.shape[2] == 1:
            return image
        else:
            raise ValueError(f"image have not supported ndim: {image.ndim}")
    elif image.ndim == 2:
        return image
    else:
        raise ValueError(f"image have not supported ndim: {image.ndim}")



def cut_image_by_bbox(image: np.ndarray, bbox: (tuple, list), min_side_size=10):
    """get sub_image with bbox [x1, y1, x2, y2]."""
    image_h, image_w = image.shape[:2]
    # validate_bbox(bbox, image_h, image_w)
    bbox = [int(x) for x in bbox]
    if (bbox[3] - bbox[1]) < min_side_size or (bbox[2] - bbox[0]) < min_side_size:
        raise AssertionError("bbox square is to low")
    sub_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    # print(sub_image.shape)
    return sub_image


def erode_mask(image: np.ndarray, mask: np.ndarray, iterations=3):
    image = image.copy()
    image = image2gray(image)
    try:
        thresh = threshold_otsu(image)
    except ValueError as e:
        thresh = 127
    bin_image = image > thresh

    bin_image = bin_image.astype(np.uint8)
    bin_image[bin_image == 1] = 255
    bin_image[bin_image == 0] = 0

    inv_image = bin_image.copy()
    inv_image[bin_image == 255] = 0
    inv_image[bin_image == 0] = 255

    # делаем делатацию над маской
    # оставляем только части с маской
    inv_image[mask == 0] = 0

    # делаем делатацию над изображением

    dilated_image = morphology.binary_dilation(inv_image, selem=morphology.disk(3)).astype(np.uint8)
    # разрушаем маску
    for i in range(iterations):
        mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
        mask = morphology.binary_erosion(mask, selem=None).astype(np.uint8)
        mask = mask[1:-1, 1:-1]
        mask[dilated_image > 0] = 255

    return mask




