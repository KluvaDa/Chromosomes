import torch
import torch.nn.functional

from functools import wraps

from typing import Callable, Sequence, Tuple, Optional

from representations import rep_nd_pytorch


"""
This file defines nored losses that apply no reduction. They maintain dimensionality except for the channel axis which
they reduce to 1 (but keep the dimension)

To create a complete loss function, apply the relevant reduction decorator to one of the nored losses.

To create a loss that applies various losses on different channels, use the combine_losses function.

The shape of the data is (batch, channels, (spacial_dimensions,)), e.g (batch, channels, x, y, z)
"""


# DECORATORS
def combine_losses(loss_functions_and_channels:
        Sequence[Tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Tuple[int, int], Tuple[int, int]]]) \
        -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    This function creates a single loss function that consists of separate loss functions applied to different channels.
    The losses are averaged.
    Args:
        loss_functions_and_channels: A list of losses to be applied with their respective start and end channels for 
            the prediction and the labels.
            The list elements are tuples of (
                loss_function,
                (prediction_channel_start, prediction_channel_end),
                (label_channel_start, label_channel_end)
            )
    Returns: A loss function
    """
    def loss_function(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        loss_values = []
        for loss_f, prediction_channels, label_channels in loss_functions_and_channels:
            loss_value = loss_f(
                prediction[:, prediction_channels[0]:prediction_channels[1], ...],
                label[:, label_channels[0]:label_channels[1], ...]
            )
            loss_values.append(loss_value)
        loss = torch.mean(torch.stack(loss_values))
        return loss
    return loss_function


def modify_label(modification_function: Callable[[torch.Tensor], torch.Tensor]) \
        -> Callable[[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
                    Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """
    Returns a decorator function, decorating a loss function by apply modification_function to the labels.
    Args:
        modification_function: A function that modifies the labels
    Returns: The decorator
    """
    def decorator(loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) \
            -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:

        @wraps(loss_function)
        def decorated_loss_function(prediction, label):
            modified_label = modification_function(label)
            loss = loss_function(prediction, modified_label)
            return loss
        return decorated_loss_function
    return decorator


def modify_prediction(modification_function: Callable[[torch.Tensor], torch.Tensor]) \
        -> Callable[[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
                    Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """
    Returns a decorator function, decorating a loss function by apply modification_function to the prediction.
    Args:
        modification_function: A function that modifies the labels
    Returns: The decorator
    """
    def decorator(loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) \
            -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:

        @wraps(loss_function)
        def decorated_loss_function(prediction, label):
            modified_prediction = modification_function(prediction)
            loss = loss_function(modified_prediction, label)
            return loss
        return decorated_loss_function
    return decorator


def metric_piecewise_decorator(piecewise_2_vector_function: Callable[[torch.Tensor], torch.Tensor])\
        -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Decorator that creates a metric from a piecewise_2_vector_function.
    """
    @wraps(piecewise_2_vector_function)
    def metric(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Calculates a nores cosine similarity angle from a collapsed piecewise prediction to a label. The label is
        aligned with each of the three axes and the smallest angular distance is selected
        """
        label_aligned_x = rep_nd_pytorch.align_vector_space(label, 0)
        label_aligned_y = rep_nd_pytorch.align_vector_space(label, 1)
        label_aligned_z = rep_nd_pytorch.align_vector_space(label, 2)

        predicted_vector = piecewise_2_vector_function(prediction)

        angle_x = cosine_similarity_angle(predicted_vector, label_aligned_x)
        angle_y = cosine_similarity_angle(predicted_vector, label_aligned_y)
        angle_z = cosine_similarity_angle(predicted_vector, label_aligned_z)

        similarity = torch.min(torch.min(angle_x, angle_y), angle_z)

        return similarity
    return metric


def square_loss_function(loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) \
        -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """ Creates a loss function with its loss values squared. """
    def loss_f(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        loss = loss_function(prediction, label)
        loss = torch.square(loss)
        return loss
    return loss_f


# REDUCTIONS - DECORATORS
def mean_reduction(loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) \
        -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Decorator function that adds a mean reduction to a loss function.
    Args:
        loss_function: A nored loss function that returns the loss in the shape (batch, 1, ...)
    Returns: The decorated loss function
    """
    @wraps(loss_function)
    def decorated_loss_function(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        loss_nored = loss_function(prediction, label)
        loss = torch.mean(loss_nored)
        return loss

    return decorated_loss_function


def max_reduction(loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) \
        -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Decorator function that adds a max reduction all axes apart from batch (dim 0) and a mean over the batch (dim 0).
    Args:
        loss_function: A nored loss function that returns the loss in the shape (batch, 1, ...)
    Returns: The decorated loss function
    """
    @wraps(loss_function)
    def decorated_loss_function(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        loss_nored = loss_function(prediction, label)
        loss = loss_nored.view(loss_nored.shape[0], -1).max(dim=-1)  # max over dimensions 1:end
        loss = torch.mean(loss)  # mean over batch dim 0
        return loss

    return decorated_loss_function


#  CORE LOSSES
def mae_nored(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """ Mean Absolute Error that maintains dimension. """
    return torch.abs(label - prediction)


def mse_nored(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """ Mean Square Error that maintains dimension. """
    return torch.pow(label - prediction, 2)


def rmse_nored(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """ Root Mean Squared Error that maintains dimension. The channel dim 1 is collapsed to a single element. """
    return torch.sqrt(torch.mean(mse_nored(prediction, label), dim=1, keepdim=True))


def cosine_similarity(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    DO NOT USE AS LOSS FUNCTION, ONLY AS METRIC
    Calculates the angular distance using cosine similarity over the channel dimension 1.
    """
    return torch.sum(prediction * label, dim=1, keepdim=True)


def cosine_similarity_angle(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    DO NOT USE AS LOSS FUNCTION, ONLY AS METRIC
    Calculates the angular distance using cosine similarity over the channel dimension 1.
    """
    return torch.acos(torch.sum(prediction * label, dim=1, keepdim=True))


# LOSSES
def binary_iou(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Calculates the iou between the binary tensors prediction and label, averaged over the batch dimension 0
    Args:
        prediction: binary tensor of shape (batch, channel, ...)
        label: binary tensor of shape (batch, channel, ...)
    Returns: A scalar torch.Tensor
    """
    assert prediction.dtype is torch.bool
    assert label.dtype is torch.bool

    intersection = torch.logical_and(prediction, label)
    union = torch.logical_or(prediction, label)

    intersection_sum = torch.sum(intersection.float(), dim=list(range(1, len(prediction.shape))))
    union_sum = torch.sum(union.float(), dim=list(range(1, len(prediction.shape))))

    iou = intersection_sum / union_sum
    iou = torch.mean(iou[~torch.isnan(iou)])  # nanmean

    return iou


def iou_loss(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Calculates the intersection over union i.e. Jaccard index, averaged over the batch dimension
    Args:
        prediction: torch.Tensor of shape (batch, channels, ...)
        label: torch.Tensor with integer indices of the correct classes of shape (batch, 1, ...)
    Returns: The loss as a torch.Tensor scalar
    """
    label = label.int()
    prediction_category = torch.argmax(prediction, dim=1, keepdim=True)
    n_channels = prediction.shape[1]
    ious = []
    for i_channel in range(n_channels):
        prediction_is_category = torch.eq(prediction_category, i_channel)
        label_is_category = torch.eq(label, i_channel)
        channel_iou_value = binary_iou(prediction_is_category, label_is_category)
        ious.append(channel_iou_value)
    iou_average = torch.mean(torch.stack(ious))
    return iou_average


def iou_loss_onehot(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Calculates the intersection over union i.e. Jaccard index, averaged over the batch dimension
    Args:
        prediction: torch.Tensor of shape (batch, channels, ...)
        label: torch.Tensor shape(batch, channels, ...) as a onehot tensor
    Returns: the loss as a torch.Tensor scalar
    """
    label_channel_index = torch.argmax(label, 1, keepdim=True)
    return iou_loss(prediction, label_channel_index)


def define_iou_loss_on_channels(channels: Sequence[int]) \
        -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Defines a iou loss function that will treat the specified channels/categories as one category.
    The labels should be category indices.
    Args:
        channels: A list of channels to unite and calculate iou over
    Returns: the iou loss function
    """
    def loss_f(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        prediction_category = torch.argmax(prediction, dim=1, keepdim=True)
        prediction_is_category = torch.any(torch.stack(
            [torch.eq(prediction_category, channel) for channel in channels],
            dim=0), dim=0)

        label = label.int()
        label_is_category = torch.any(torch.stack(
            [torch.eq(label, channel) for channel in channels], dim=0), dim=0)

        return binary_iou(prediction_is_category, label_is_category)
    return loss_f


def define_iou_loss_on_channels_onehot(channels: Sequence[int]) \
        -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Defines a iou loss function that will treat the specified channels/categories as one category.
    The labels should be one-hot vectors
    Args:
        channels: A list of channels to unite and calculate iou over
    Returns: the iou loss function
    """
    iou_loss_function_on_channels = define_iou_loss_on_channels(channels)

    def loss_f(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        label_category = torch.argmax(label, dim=1, keepdim=True)
        return iou_loss_function_on_channels(prediction, label_category)
    return loss_f


def define_weighted_cross_entropy_loss(weight: Optional[torch.Tensor]) \
        -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Defines a weighted cross entropy loss, where the labels are class indices
    Args:
        weight: A torch.Tensor with the same size as the number of channels/classes containing their respective weights
    Returns: The loss function
    """
    def loss_f(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(prediction, label[:, 0, ...].long(), weight=weight)
    return loss_f


def define_weighted_cross_entropy_loss_onehot(weight: Optional[torch.Tensor])  \
        -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Defines a weighted cross entropy loss, where the labels are onehot vectors
    Args:
        weight: A torch.Tensor with the same size as the number of channels/classes containing their respective weights
    Returns: The loss function
    """
    def loss_f(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(prediction, torch.argmax(label, 1), weight=weight)
    return loss_f


cross_entropy_loss = define_weighted_cross_entropy_loss(None)
cross_entropy_loss_onehot = define_weighted_cross_entropy_loss_onehot(None)


