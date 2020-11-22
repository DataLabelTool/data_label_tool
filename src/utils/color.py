from typing import Union, List, Tuple


def validate_color(color: Union[List[int], Tuple[int]]):
    """check if color is list or tuple with len between 3 and 4"""
    assert 3 <= len(color) <= 4, \
        f"color len should be between 3 and 4, got: {len(color)}"


def shift_color(color: Union[List[int], Tuple[int]], delta_color=(-50, -50, -50, 0)) -> List[int]:
    """colors have RGBA format"""
    color = list(color)
    delta_color = list(delta_color)
    validate_color(color)
    for i in range(len(color)):
        color[i] += delta_color[i]
        color[i] = max(min(color[i], 255), 0)
    return color
