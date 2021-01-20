import numpy as np
from matplotlib.colors import hsv_to_rgb
import matplotlib.cm

from math import pi


def angle_pi_to_rgb(angle: np.ndarray) -> np.ndarray:
    """
    Takes an np image with angles mod pi and creates an rgb visualisation.
    Args:
        param angle: 2D np array with angle values mod pi
    Returns: rgb image, channel last
    """
    angle = np.remainder(angle, pi)/pi  # normalise to [0, 1)
    rgb = hsv_to_rgb(np.stack([angle, np.ones_like(angle), np.ones_like(angle)], axis=2))
    return rgb


def visualize_direction_to_rgb(union: np.ndarray,
                               intersection: np.ndarray,
                               intersection_dilated: np.ndarray,
                               angle: np.ndarray,
                               c: float = 0.0)\
        -> np.ndarray:
    """
    Creates an rgb image visualising the output of direction type networks
    Args:
        union: 2D np array
        intersection: 2D np array
        intersection_dilated: 2D np array
        angle: 2D np array
        c: base saturation and value (value between 0 and 1)
    Returns: rgb image, channel last
    """
    union = union.astype(np.float)
    intersection = intersection.astype(np.float)
    intersection_dilated = intersection_dilated.astype(np.float)
    angle = angle.astype(np.float)

    intersection = (intersection + intersection_dilated) / 2
    angle = np.remainder(angle, pi) / pi  # normalise to [0, 1)
    saturation = c + (1 - intersection) * (1 - c)
    value = c + union * (1 - c)

    saturation = np.clip(saturation, 0, 1)
    value = np.clip(value, 0, 1)

    rgb = hsv_to_rgb(np.stack([angle, saturation, value], axis=-1))
    return rgb


def visualise_categories_to_rgb(categories_image: np.ndarray) -> np.ndarray:
    """
    Change a categories image to rgb
    Args:
        param categories_image: 2d np image of ints, where the value corresponds to distinct categories
    Returns: rgb image, channel last
    """
    cmap = matplotlib.cm.get_cmap('tab10')

    rgb = np.zeros([categories_image.shape[0], categories_image.shape[1], 3], dtype=np.float)
    for r in range(categories_image.shape[0]):
        for c in range(categories_image.shape[1]):
            rgb[r, c, :] = cmap(categories_image[r, c])[0:3]

    return rgb
