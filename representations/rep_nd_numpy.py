import numpy as np
import torch
import functools
from typing import Any, Callable

from representations import rep_nd_pytorch


def torch_2_numpy(torch_function: Callable[[torch.Tensor, Any], torch.Tensor]) \
        -> Callable[[np.ndarray, Any], np.ndarray]:
    """
    Decorator that makes a torch function work as a numpy function.
    Args:
        torch_function: Any function that takes a torch.Tensor as its first parameter and returns a torch.Tensor.
    Returns: The decorator function that takes a numpy array as its first parameter and returns a numpy array.
    """
    @functools.wraps(torch_function)
    def decorated_function(np_input: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        torch_input = torch.from_numpy(np_input)
        torch_output = torch_function(torch_input, *args, **kwargs)
        np_output = torch_output.numpy()
        return np_output

    return decorated_function


def torch_2_numpy_two_inputs(torch_function: Callable[[torch.Tensor, torch.Tensor, Any], torch.Tensor]) \
        -> Callable[[np.ndarray, Any], np.ndarray]:
    """
    Decorator that makes a torch function work as a numpy function.
    Args:
        torch_function: Any function that takes two torch.Tensor as its first parameters and returns a torch.Tensor.
    Returns: The decorator function that takes two numpy array as its first parameters and returns a numpy array.
    """
    @functools.wraps(torch_function)
    def decorated_function(np_input_1: np.ndarray, np_input_2: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        torch_input_1 = torch.from_numpy(np_input_1)
        torch_input_2 = torch.from_numpy(np_input_2)
        torch_output = torch_function(torch_input_1, torch_input_2, *args, **kwargs)
        np_output = torch_output.numpy()
        return np_output

    return decorated_function


cosine_similarity = torch_2_numpy_two_inputs(rep_nd_pytorch.cosine_similarity)
cosine_similarity_angle = torch_2_numpy_two_inputs(rep_nd_pytorch.cosine_similarity_angle)
cosine_similarity_to_axis = torch_2_numpy(rep_nd_pytorch.cosine_similarity_to_axis)
angle_to_axis = torch_2_numpy(rep_nd_pytorch.angle_to_axis)
align_vector_space = torch_2_numpy(rep_nd_pytorch.align_vector_space)
adjust_vector_space_linear_choice = torch_2_numpy(rep_nd_pytorch.adjust_vector_space_linear_choice)
adjust_vector_space_smooth_choice = torch_2_numpy(rep_nd_pytorch.adjust_vector_space_smooth_choice)
adjust_vector_space_sum = torch_2_numpy(rep_nd_pytorch.adjust_vector_space_sum)
vector_2_piecewise_adjusted = torch_2_numpy(rep_nd_pytorch.vector_2_piecewise_adjusted)
vector_2_piecewise = torch_2_numpy(rep_nd_pytorch.vector_2_piecewise)
vector_2_piecewise_adjusted_linear_choice = torch_2_numpy(rep_nd_pytorch.vector_2_piecewise_adjusted_linear_choice)
vector_2_piecewise_adjusted_smooth_choice = torch_2_numpy(rep_nd_pytorch.vector_2_piecewise_adjusted_smooth_choice)
vector_2_piecewise_adjusted_sum = torch_2_numpy(rep_nd_pytorch.vector_2_piecewise_adjusted_sum)
piecewise_choice_2_vector = torch_2_numpy(rep_nd_pytorch.piecewise_choice_2_vector)
piecewise_sum_2_vector = torch_2_numpy(rep_nd_pytorch.piecewise_sum_2_vector)
