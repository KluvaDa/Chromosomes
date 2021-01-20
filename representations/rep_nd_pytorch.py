import torch
import torch.nn.functional
import math
from math import pi

from typing import Callable


def cosine_similarity(vector_0: torch.Tensor,
                      vector_1: torch.Tensor,
                      vector_dim: int = 1,
                      keepdim: bool = True) \
        -> torch.Tensor:
    """
    Calculates the cosine of the angle between two vectors (or vector spaces)
    Args:
        vector_0: torch.Tensor - the first vector/vector space
        vector_1: torch.Tensor - the second vector/vector space
        vector_dim: which dimension in the vector space is the vector space
        keepdim: whether to keep the vector dimension
    """
    dot_product = torch.sum(vector_0 * vector_1, dim=vector_dim, keepdim=keepdim)
    magnitude_0 = torch.norm(vector_0, dim=vector_dim, keepdim=keepdim)
    magnitude_1 = torch.norm(vector_1, dim=vector_dim, keepdim=keepdim)
    cos = dot_product/(magnitude_0 * magnitude_1)
    return cos


def cosine_similarity_angle(vector_0: torch.Tensor,
                            vector_1: torch.Tensor,
                            vector_dim: int = 1,
                            keepdim: bool = True) \
        -> torch.Tensor:
    """
    Calculates the angle between two vectors (or vector spaces) using arccos of cosine similarity
    Args:
        vector_0: torch.Tensor - the first vector/vector space
        vector_1: torch.Tensor - the second vector/vector space
        vector_dim: which dimension in the vector space is the vector space
        keepdim: whether to keep the vector dimension
    """
    cos = cosine_similarity(vector_0, vector_1, vector_dim, keepdim)
    angle = torch.acos(cos)
    return angle


def cosine_similarity_to_axis(vector: torch.Tensor,
                              axis: int,
                              vector_dim: int = 1,
                              keepdim: bool = True) \
        -> torch.Tensor:
    """
    Calculates the cosine similarity between the vectors and one of the axes
    Args:
        vector: torch.Tensor - the vector/vector space
        axis: int - which axis to calculate distance to
        vector_dim: which dimension in the vector space is the vector space
        keepdim: whether to keep the vector dimension
    """
    assert vector_dim < len(vector.shape)
    broadcast_shape = [1] * len(vector.shape)
    broadcast_shape[vector_dim] = vector.shape[vector_dim]
    axis_vector = torch.zeros(broadcast_shape, dtype=vector.dtype, device=vector.device)
    slices = [slice(None, None)] * len(vector.shape)
    slices[vector_dim] = slice(axis, axis+1)
    axis_vector[tuple(slices)] = 1

    cos = cosine_similarity(vector, axis_vector, vector_dim=vector_dim, keepdim=keepdim)
    return cos


def angle_to_axis(vector: torch.Tensor,
                  axis: int,
                  vector_dim: int = 1,
                  keepdim: bool = True) \
        -> torch.Tensor:
    """
    Calculates the angle between the vectors and one of the axes
    Args:
        vector: torch.Tensor - the vector/vector space
        axis: int - which axis to calculate distance to
        vector_dim: which dimension in the vector space is the vector space
        keepdim: whether to keep the vector dimension
    """
    cos = cosine_similarity_to_axis(vector, axis, vector_dim=vector_dim, keepdim=keepdim)
    angle = torch.acos(cos)
    return angle


def align_vector_space(vector_space: torch.Tensor,
                       align_channel: int,
                       align_negative: bool = False) \
        -> torch.Tensor:
    """
    Takes a n dimensional vector space with the shape (batch, n, ...), where the n channels in dim 1 are the
    n components of the vectors. The vectors will be multiplied by 1 or -1 such that the specified align_channel element
    in dim 1 is positive (or negative if align_negative) for every point in the space.

    Args:
        vector_space: Pytorch Tensor of shape (batch, n, ...).
        align_channel: The index (dimension) of the vectors that is meant to be positive.
        align_negative: Whether to align to negative values instead.
    Returns: The aligned vector space, a tensor of shape (batch, n, ...)
    """
    assert align_channel < vector_space.shape[1]  # ensure align_channel < n

    sign = -1 if align_negative else 1

    vector_space_is_positive = vector_space[:, align_channel:align_channel + 1, ...] >= 0
    vector_space_multiplier = vector_space_is_positive.type(torch.int) * 2 - 1
    vector_space_aligned = vector_space * vector_space_multiplier * sign

    return vector_space_aligned


def adjust_vector_space_linear_choice(vector_space: torch.Tensor,
                                      align_channel: int)\
        -> torch.Tensor:
    """
    Adjusts the radius of the vector space depending on the angular distance between the vector and the align_channel
    axis. The linear choice will taper from an angle of 45 degrees to 90 degrees linearly from 1 to 0.
    Args:
        vector_space: Tensor of shape (batch, n, ...)
        align_channel: The channel/dimension from which to calculate the angle
    Returns: the aligned vector space, a tensor of shape (batch, n, ...)
    """
    theta = angle_to_axis(vector_space, axis=align_channel)
    multiplier = torch.clamp(2 - 4 * theta / pi, 0, 1)
    adjusted_vector_space = multiplier * vector_space
    return adjusted_vector_space


def adjust_vector_space_smooth_choice(vector_space: torch.Tensor,
                                      align_channel: int)\
        -> torch.Tensor:
    """
    Adjusts the radius of the vector space depending on the angular distance between the vector and the align_channel
    axis. The linear choice will taper from an angle of 45 degrees to 90 degrees sinusoidally from 1 to 0.
    Args:
        vector_space: Tensor of shape (batch, n, ...)
        align_channel: The channel/dimension from which to calculate the angle
    Returns: the aligned vector space, a tensor of shape (batch, n, ...)
    """
    cos_theta = cosine_similarity_to_axis(vector_space, axis=align_channel)
    # sin(2x)^2 is the sinusoidal between 45 and 90 degrees. It is equal to 4*cos(x)^2*(1-cos(x)^2)
    cos_theta_squared = torch.pow(cos_theta, 2)
    smooth_multiplier = 4*cos_theta_squared*(1-cos_theta_squared)
    smooth_mask = torch.lt(cos_theta, math.cos(math.pi/4)) * torch.ge(cos_theta, 0)
    ones_value = torch.ge(cos_theta, math.cos(math.pi/4)).type(vector_space.dtype).type(vector_space.dtype)

    multiplier = ones_value + smooth_multiplier * smooth_mask

    adjusted_vector_space = multiplier * vector_space
    return adjusted_vector_space


def adjust_vector_space_sum(vector_space: torch.Tensor,
                            align_channel: int)\
        -> torch.Tensor:
    """
    Adjusts the radius of the vector space depending on the angular distance between the vector and the align_channel
    axis. The linear choice will taper from an angle of 0 degrees to 90 degrees sinusoidally from 1 to 0.
    Args:
        vector_space: Tensor of shape (batch, n, ...)
        align_channel: The channel/dimension from which to calculate the angle
    Returns: the aligned vector space, a tensor of shape (batch, n, ...)
    """
    cos_theta = cosine_similarity_to_axis(vector_space, axis=align_channel)
    # cos(x)^2 is the sinusoidal between 0 and 90 degrees
    cos_theta_squared = torch.pow(cos_theta, 2)
    smooth_mask = torch.gt(cos_theta, 0)

    multiplier = cos_theta_squared * smooth_mask

    adjusted_vector_space = multiplier * vector_space
    return adjusted_vector_space


def vector_2_piecewise_adjusted(vector_space: torch.Tensor,
                                adjustment_function: Callable[[torch.Tensor, int], torch.Tensor]) \
        -> torch.Tensor:
    """
    Vector space to piecewise representation, the magnitude of which is adjusted using the specified
    adjustment_function.

    The vector space is repeated n times and multiplied by either -1 or 1 such that the nth dimension is positive.
    The dimensionality n is deduced from the length of dim 1.
    Args:
        vector_space: Tensor of shape (batch, n, ...)
        adjustment_function: Adjustment function applied to each of the piecewise vector spaces
    Returns: Tensor of shape (batch, n**2, ...)
    """
    n = vector_space.shape[1]
    piecewise_vector_spaces = []
    for i_n in range(n):
        vector_space_aligned = align_vector_space(vector_space, i_n)
        vector_space_aligned_adjusted = adjustment_function(vector_space_aligned, i_n)
        piecewise_vector_spaces.append(vector_space_aligned_adjusted)
    piecewise = torch.cat(piecewise_vector_spaces, dim=1)
    return piecewise


def vector_2_piecewise(vector_space: torch.Tensor) \
        -> torch.Tensor:
    """
    Vector space to piecewise representation. No adjustment function is used.

    The vector space is repeated n times and multiplied by either -1 or 1 such that the nth dimension is positive.
    The dimensionality n is deduced from the length of dim 1.
    Args:
        vector_space: Tensor of shape (batch, n, ...)
    Returns: Tensor of shape (batch, n**2, ...)
    """
    return vector_2_piecewise_adjusted(vector_space, lambda x, y: x)


def vector_2_piecewise_adjusted_linear_choice(vector_space: torch.Tensor) \
        -> torch.Tensor:
    """
    Vector space to piecewise representation. Adjusted using the adjust_vector_space_linear_choice function.

    The vector space is repeated n times and multiplied by either -1 or 1 such that the nth dimension is positive.
    The dimensionality n is deduced from the length of dim 1.
    Args:
        vector_space: Tensor of shape (batch, n, ...)
    Returns: Tensor of shape (batch, n**2, ...)
    """
    return vector_2_piecewise_adjusted(vector_space, adjust_vector_space_linear_choice)


def vector_2_piecewise_adjusted_smooth_choice(vector_space: torch.Tensor) \
        -> torch.Tensor:
    """
    Vector space to piecewise representation. Adjusted using the adjust_vector_space_smooth_choice function.

    The vector space is repeated n times and multiplied by either -1 or 1 such that the nth dimension is positive.
    The dimensionality n is deduced from the length of dim 1.
    Args:
        vector_space: Tensor of shape (batch, n, ...)
    Returns: Tensor of shape (batch, n**2, ...)
    """
    return vector_2_piecewise_adjusted(vector_space, adjust_vector_space_smooth_choice)


def vector_2_piecewise_adjusted_sum(vector_space: torch.Tensor) \
        -> torch.Tensor:
    """
    Vector space to piecewise representation. Adjusted using the adjust_vector_space_sum function.

    The vector space is repeated n times and multiplied by either -1 or 1 such that the nth dimension is positive.
    The dimensionality n is deduced from the length of dim 1.
    Args:
        vector_space: Tensor of shape (batch, n, ...)
    Returns: Tensor of shape (batch, n**2, ...)
    """
    return vector_2_piecewise_adjusted(vector_space, adjust_vector_space_sum)


def piecewise_choice_2_vector(piecewise: torch.Tensor,
                              normalise: bool = True)\
        -> torch.Tensor:
    """
    Selects the vector that is closest to it's respective axis from the piecewise representation.
    Can be applied to a piecewise_adjusted_linear_choice or piecewise_adjusted_smooth_choice piecewise representation.
    Args:
        piecewise: The piecewise representation tensor of shape (batch, n**2, ...)
        normalise: Whether to normalise the vector length to 1 at the end
    Returns: A vector space tensor of shape (batch, n, ...)
    """
    n = math.sqrt(piecewise.shape[1])
    assert float.is_integer(n)
    n = int(n)

    cosines_to_axis = torch.empty((piecewise.shape[0], n) + piecewise.shape[2:],
                                  dtype=piecewise.dtype,
                                  device=piecewise.device)
    for i_n in range(n):
        vector_space = piecewise[:, i_n*n: (i_n+1)*n, ...]
        cosines_to_axis[:, i_n:i_n+1, ...] = cosine_similarity_to_axis(vector_space, axis=i_n)

    piecewise_select = torch.argmax(cosines_to_axis, dim=1, keepdim=True)

    output = torch.zeros_like(cosines_to_axis)
    for i_n in range(n):
        vector_space = piecewise[:, i_n*n: (i_n+1)*n, ...]
        output += vector_space * (piecewise_select == i_n)

    if normalise:
        output /= torch.norm(output, dim=1)

    return output


def piecewise_sum_2_vector(piecewise: torch.Tensor,
                           normalise: bool = True)\
        -> torch.Tensor:
    """
    Adds the vectors from the piecewise representation.
    The vectors are first aligned with one of the axes before being summed. This is performed for each axis and the
    one with the largest resultant vector magnitude is selected.
    Args:
        piecewise: The piecewise representation tensor of shape (batch, n**2, ...)
        normalise: Whether to normalise the vector length to 1 at the end
    Returns: A vector space tensor of shape (batch, n, ...)
    """
    n = math.sqrt(piecewise.shape[1])  # how many dimensions we are working with
    assert float.is_integer(n)
    n = int(n)

    # Calculate the sum of the piecewise vectors, when aligned with each possible dim.
    sums_aligned = torch.zeros((n, piecewise.shape[0], n) + piecewise.shape[2:],
                               dtype=piecewise.dtype,
                               device=piecewise.device)
    for align_with_dim in range(n):
        for i_n in range(n):
            vector_space_i_n = piecewise[:, i_n*n: (i_n+1)*n, ...]
            vector_space_i_n_aligned = align_vector_space(vector_space_i_n, align_with_dim)
            sums_aligned[align_with_dim, :, :, ...] += vector_space_i_n_aligned

    # Magnitudes for each alignment dim
    magnitudes_aligned = torch.zeros((piecewise.shape[0], n) + piecewise.shape[2:],
                                     dtype=piecewise.dtype,
                                     device=piecewise.device)
    for align_with_dim in range(n):
        magnitudes_aligned[:, align_with_dim, ...] = torch.norm(sums_aligned[align_with_dim, :, :, ...], dim=1)

    # Select the alignment dim with the largest magnitude
    piecewise_select = torch.argmax(magnitudes_aligned, dim=1, keepdim=True)
    output = torch.zeros_like(magnitudes_aligned)
    for aligned_with_dim in range(n):
        vector_space = sums_aligned[aligned_with_dim, :, :, ...]
        output += vector_space * (piecewise_select == aligned_with_dim)

    if normalise:
        output /= torch.norm(output, dim=1)

    return output
