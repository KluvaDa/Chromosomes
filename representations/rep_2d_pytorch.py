import torch
from math import pi
from representations import rep_nd_pytorch

from typing import Callable


def identity(x):
    return x


def get_angle_2_repr(representation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns the function to transform angle to the correct representation
    Args:
        representation: one of the following: ('angle', 'vector', 'da_vector', 'piecewise',
            'piecewise_adjusted_linear_choice', 'piecewise_adjusted_smooth_choice', 'piecewise_adjusted_sum')
    """
    if representation == 'angle':
        return identity
    elif representation == 'vector':
        return angle_2_vector
    elif representation == 'da_vector':
        return angle_2_da_vector
    elif representation == 'piecewise':
        return angle_2_piecewise
    elif representation == 'piecewise_adjusted_linear_choice':
        return angle_2_piecewise_adjusted_linear_choice
    elif representation == 'piecewise_adjusted_smooth_choice':
        return angle_2_piecewise_adjusted_smooth_choice
    elif representation == 'piecewise_adjusted_sum':
        return angle_2_piecewise_adjusted_sum
    else:
        raise ValueError(f"Invalid representation: {representation}")


def get_repr_2_angle(representation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns the function to transform the specified representation to angle
    Args:
        representation: one of the following: ('angle', 'vector', 'da_vector', 'piecewise_choice', 'piecewise_sum'
    """
    if representation == 'angle':
        return identity
    elif representation == 'vector':
        return vector_2_angle
    elif representation == 'da_vector':
        return da_vector_2_angle
    elif representation in ['piecewise_choice',
                            'piecewise',
                            'piecewise_adjusted_linear_choice',
                            'piecewise_adjusted_smooth_choice']:
        return piecewise_choice_2_angle
    elif representation in ['piecewise_sum',
                            'piecewise_adjusted_sum']:
        return piecewise_sum_2_angle
    else:
        raise ValueError(f"Invalid representation: {representation}")


def angle_2_vector(angles: torch.Tensor) -> torch.Tensor:
    """
    Angles in radians to vector space; 0 radians -> (1, 0), pi/2 radians -> (0, 1)
    Args:
        angles: torch.Tenor of shape (batch, 1, x, y)
    Returns: torch tensor of shape (batch, 2, x, y)
    """
    vectors_x = torch.cos(angles)
    vectors_y = torch.sin(angles)
    vectors = torch.cat([vectors_x, vectors_y], dim=1)
    return vectors


def vector_2_angle(vectors: torch.Tensor) -> torch.Tensor:
    """
    Vector space to angles in radians in range [0, 2pi); (1, 0) -> 0 radians, (0, 1) -> pi/2 radians
    Args:
        vectors: torch.Tensor of shape (batch, 2, x, y)
    Returns: torch.Tensor of shape (batch, 1, x, y)
    """
    angle = torch.atan2(vectors[:, 1:2, ...], vectors[:, 0:1, ...])
    angle = torch.remainder(angle, 2*pi)
    return angle


def angle_2_da_vector(angles: torch.Tensor) -> torch.Tensor:
    """
    Angles in radians to double-angle vector space; 0 radians -> (1, 0), pi/4 radians -> (0, 1)
        Args:
            angles: torch.Tenor of shape (batch, 1, x, y)
        Returns: torch tensor of shape (batch, 2, x, y)
        """
    doubleangles = angles*2
    da_vectors_x = torch.cos(doubleangles)
    da_vectors_y = torch.sin(doubleangles)
    da_vectors = torch.cat([da_vectors_x, da_vectors_y], dim=1)
    return da_vectors


def da_vector_2_angle(vectors: torch.Tensor) -> torch.Tensor:
    """
    Double-angle vector space to angles in radians in range [0, pi); (1, 0) -> 0 radians, (0, 1) -> pi/4 radians
    Args:
        vectors: torch.Tensor of shape (batch, 2, x, y)
    Returns: torch.Tensor of shape (batch, 1, x, y)
    """
    double_angle = torch.atan2(vectors[:, 1:2, ...], vectors[:, 0:1, ...])
    double_angle = torch.remainder(double_angle, 2*pi)
    angle = double_angle / 2
    return angle


def angle_2_piecewise(angles: torch.Tensor) \
        -> torch.Tensor:
    """
    Angles in radians to piecewise representation. No adjustment function is used.

    The vector space is repeated n times and multiplied by either -1 or 1 such that the nth dimension is positive.
    The dimensionality n is deduced from the length of dim 1.
    Args:
        angles: Tensor of shape (batch, 1, ...)
    Returns: Tensor of shape (batch, n**2, ...)
    """
    vectors = angle_2_vector(angles)
    piecewise = rep_nd_pytorch.vector_2_piecewise(vectors)
    return piecewise


def angle_2_piecewise_adjusted_linear_choice(angles: torch.Tensor) \
        -> torch.Tensor:
    """
    Angles in radians to piecewise representation. Adjusted using the adjust_vector_space_linear_choice function.

    The vector space is repeated n times and multiplied by either -1 or 1 such that the nth dimension is positive.
    The dimensionality n is deduced from the length of dim 1.
    Args:
        angles: Tensor of shape (batch, 1, ...)
    Returns: Tensor of shape (batch, n**2, ...)
    """
    vectors = angle_2_vector(angles)
    piecewise = rep_nd_pytorch.vector_2_piecewise_adjusted_linear_choice(vectors)
    return piecewise


def angle_2_piecewise_adjusted_smooth_choice(angles: torch.Tensor) \
        -> torch.Tensor:
    """
    Angles in radians to piecewise representation. Adjusted using the adjust_vector_space_smooth_choice function.

    The vector space is repeated n times and multiplied by either -1 or 1 such that the nth dimension is positive.
    The dimensionality n is deduced from the length of dim 1.
    Args:
        angles: Tensor of shape (batch, 1, ...)
    Returns: Tensor of shape (batch, n**2, ...)
    """
    vectors = angle_2_vector(angles)
    piecewise = rep_nd_pytorch.vector_2_piecewise_adjusted_smooth_choice(vectors)
    return piecewise


def angle_2_piecewise_adjusted_sum(angles: torch.Tensor) \
        -> torch.Tensor:
    """
    Angles in radians to piecewise representation. Adjusted using the adjust_vector_space_sum function.

    The vector space is repeated n times and multiplied by either -1 or 1 such that the nth dimension is positive.
    The dimensionality n is deduced from the length of dim 1.
    Args:
        angles: Tensor of shape (batch, 1, ...)
    Returns: Tensor of shape (batch, n**2, ...)
    """
    vectors = angle_2_vector(angles)
    piecewise = rep_nd_pytorch.vector_2_piecewise_adjusted_sum(vectors)
    return piecewise


def piecewise_choice_2_angle(piecewise: torch.Tensor) \
        -> torch.Tensor:
    """
    Selects the vector that is closest to it's respective axis from the piecewise representation.
    Can be applied to a piecewise_adjusted_linear_choice or piecewise_adjusted_smooth_choice piecewise representation.
    Args:
        piecewise: The piecewise representation tensor of shape (batch, n**2, ...)
    Returns: A tensor of angles in radians of shape (batch, 1, ...)
    """
    vector_space = rep_nd_pytorch.piecewise_choice_2_vector(piecewise, False)
    angles = vector_2_angle(vector_space)
    return angles


def piecewise_sum_2_angle(piecewise: torch.Tensor) \
        -> torch.Tensor:
    """
    Adds the vectors from the piecewise representation.
    The vectors are first aligned with one of the axes before being summed. This is performed for each axis and the
    one with the largest resultant vector magnitude is selected.
    Args:
        piecewise: The piecewise representation tensor of shape (batch, n**2, ...)
    Returns: A tensor of angles of shape (batch, 1, ...)
    """
    vector_space = rep_nd_pytorch.piecewise_sum_2_vector(piecewise, False)
    angles = vector_2_angle(vector_space)
    return angles
