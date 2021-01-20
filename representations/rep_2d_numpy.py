import numpy as np

from representations.rep_nd_numpy import torch_2_numpy
from representations import rep_2d_pytorch

from typing import Callable


def get_angle_2_repr(representation: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns the function to transform angle to the correct representation
    Args:
        representation: one of the following: ('vector', 'da_vector', 'piecewise', 'piecewise_adjusted_linear_choice',
                                               'piecewise_adjusted_smooth_choice', 'piecewise_adjusted_sum')
    """
    return torch_2_numpy(rep_2d_pytorch.get_angle_2_repr(representation))


def get_repr_2_angle(representation: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns the function to transform the specified representation to angle
    Args:
        representation: one of the following: ('vector', 'da_vector', 'piecewise_choice', 'piecewise_sum'
    """
    return torch_2_numpy(rep_2d_pytorch.get_repr_2_angle(representation))


angle_2_vector = torch_2_numpy(rep_2d_pytorch.angle_2_vector)
vector_2_angle = torch_2_numpy(rep_2d_pytorch.vector_2_angle)
angle_2_da_vector = torch_2_numpy(rep_2d_pytorch.angle_2_da_vector)
da_vector_2_angle = torch_2_numpy(rep_2d_pytorch.da_vector_2_angle)
angle_2_piecewise = torch_2_numpy(rep_2d_pytorch.angle_2_piecewise)
angle_2_piecewise_adjusted_linear_choice = torch_2_numpy(rep_2d_pytorch.angle_2_piecewise_adjusted_linear_choice)
angle_2_piecewise_adjusted_smooth_choice = torch_2_numpy(rep_2d_pytorch.angle_2_piecewise_adjusted_smooth_choice)
angle_2_piecewise_adjusted_sum = torch_2_numpy(rep_2d_pytorch.angle_2_piecewise_adjusted_sum)
piecewise_choice_2_angle = torch_2_numpy(rep_2d_pytorch.piecewise_choice_2_angle)
piecewise_sum_2_angle = torch_2_numpy(rep_2d_pytorch.piecewise_sum_2_angle)
