import torch
import torch.nn.functional
from math import pi
from typing import Callable, Any, Optional, Sequence

import losses
from representations import rep_2d_pytorch


def _identity(x: Any) -> Any:
    return x


def define_loss(representation_type: str,
                loss_type: str,
                repr_reduction_type: str,
                include_classification: bool = False,
                cross_entropy_weight: Optional[Sequence[float]] = None)\
        -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    A selector function that outputs a loss function based on the two keywords representation and loss.
    Not all losses are applicable to all representations.
    The first 3 channels are binary categories, with channel 1 and 2 as the union and intersection. Their xor is used
    as the mask for the valid entries for the loss function.
    Args:
        :param representation_type: keyword from the following: 'angle', 'vector', 'da_vector', 'piecewise',
            'piecewise_adjusted_linear_choice', 'piecewise_adjusted_smooth_choice', 'piecewise_adjusted_sum'
        :param loss_type: keyword from the following: 'mae', 'mse', 'rmse', 'mod_angular', 'squared_mod_angular',
        :param repr_reduction_type: keyword from the following: 'mean', 'valid_mean', 'sum', 'max'
        :param include_classification: Whether the loss function should apply a binary_cross_entropy_loss_with_logits to
            the first 3 channels or not
        :param cross_entropy_weight: The weighting to apply to the 3 classification channels
    :return: loss function
    """
    # define 4 parts of the loss
    # 1. the core nored loss to be used
    # 2. a function to modify the labels before feeding into the core nored loss
    # 3. a function to modify the prediction before feeding into the core nored loss
    # 4. a combine function applied to the representation that will mask the relevant direction values

    # expects the label to be an angle in the range [0, pi)

    # check values
    if cross_entropy_weight is not None:
        assert len(cross_entropy_weight) == 3, "The cross_entropy_weight must have 3 elements"
    else:
        cross_entropy_weight = [1.0, 1.0, 1.0]

    # 1. define the core nored loss function
    loss_nored_functions = {
        'mae': losses.mae_nored,
        'mse': losses.mse_nored,
        'rmse': losses.rmse_nored,
        'mod_angular': define_angular_loss_nored(pi),
        'squared_mod_angular': losses.square_loss_function(define_angular_loss_nored(pi))
    }
    loss_nored = loss_nored_functions.get(loss_type, None)
    if loss_nored is None:
        raise ValueError(f"invalid loss_type: '{loss_type}'")

    # 2. define a function to modify the labels before feeding into the core nored loss
    if any([
            loss_type == 'mod_angular',
            loss_type == 'squared_mod_angular',
            representation_type == 'angle'
    ]):
        modify_label_function = _identity
    elif representation_type == 'vector':
        modify_label_function = rep_2d_pytorch.angle_2_vector
    elif representation_type == 'da_vector':
        modify_label_function = rep_2d_pytorch.angle_2_da_vector
    elif representation_type == 'piecewise':
        modify_label_function = rep_2d_pytorch.angle_2_piecewise
    elif representation_type == 'piecewise_adjusted_linear_choice':
        modify_label_function = rep_2d_pytorch.angle_2_piecewise_adjusted_linear_choice
    elif representation_type == 'piecewise_adjusted_smooth_choice':
        modify_label_function = rep_2d_pytorch.angle_2_piecewise_adjusted_smooth_choice
    elif representation_type == 'piecewise_adjusted_sum':
        modify_label_function = rep_2d_pytorch.angle_2_piecewise_adjusted_sum
    else:
        raise ValueError(f"invalid representation_type: '{representation_type}'")

    # 3. define a function to modify the prediction before feeding into the core nored loss
    if any([
            loss_type == 'mod_angular',
            loss_type == 'squared_mod_angular'
    ]):
        if representation_type == 'angle':
            modify_prediction_function = _identity
        elif representation_type == 'vector':
            modify_prediction_function = rep_2d_pytorch.vector_2_angle
        elif representation_type == 'da_vector':
            modify_prediction_function = rep_2d_pytorch.da_vector_2_angle
        elif representation_type in ['piecewise',
                                     'piecewise_adjusted_linear_choice',
                                     'piecewise_adjusted_smooth_choice']:
            modify_prediction_function = rep_2d_pytorch.piecewise_choice_2_angle
        elif representation_type == 'piecewise_adjusted_sum':
            modify_prediction_function = rep_2d_pytorch.piecewise_sum_2_angle
        else:
            raise ValueError(f"invalid representation_type: '{representation_type}'")
    else:
        modify_prediction_function = _identity

    # 4. define a combine function that will mask the relevant direction values
    reduction_function = _define_masked_reduction(repr_reduction_type)

    def loss_function(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # channel 0 = union, channel 1 = intersection, channel 2 = intersection_dilated
        mask = torch.logical_xor(label[:, 0:1, ...], label[:, 1:2, ...])

        # 2. modify labels
        label_representation = modify_label_function(label[:, 3:, ...])
        # 3. modify predictions
        prediction_representation = modify_prediction_function(prediction[:, 3:, ...])
        # 1. apply nored loss function
        loss_representation = loss_nored(prediction_representation, label_representation)
        # 4. apply masked reduction to loss_representation
        loss_representation = reduction_function(loss_representation, mask)
        # average the two losses
        if include_classification:
            segmentation_losses = []
            for channel_i, weight in enumerate(cross_entropy_weight):
                segmentation_losses.append(
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        prediction[:, channel_i: channel_i+1, ...],
                        label[:, channel_i: channel_i+1, ...]
                    ) * weight
                )
            segmentation_loss = (segmentation_losses[0] + segmentation_losses[1] + segmentation_losses[2]) / 3
            loss = (segmentation_loss + loss_representation) / 2
        else:
            loss = loss_representation
        return loss
    return loss_function


def _define_masked_reduction(masked_reduction: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """ Reduction definitions from a string. Always applies mean over batch dimension; 'mean', 'sum' or 'max' over the
    other dimensions.
    Args:
        masked_reduction: 'mean', 'valid_mean', 'sum', or 'max'
    """
    if isinstance(masked_reduction, str):
        if masked_reduction == 'mean':
            def masked_reduction(tensor, mask):
                tensor = tensor * mask
                tensor = torch.mean(tensor)
                return tensor
        elif masked_reduction == 'valid_mean':
            def masked_reduction(tensor, mask):
                tensor = tensor * mask
                tensor = torch.sum(tensor) / torch.sum(mask)  # mean over valid mask entries
                return tensor
        elif masked_reduction == 'sum':
            def masked_reduction(tensor, mask):
                tensor = tensor * mask
                # sum along all dimensions except batch
                tensor = torch.sum(tensor, dim=list(range(1, len(tensor.shape))))
                tensor = torch.mean(tensor)  # mean over batch
                return tensor
        elif masked_reduction == 'max':
            def masked_reduction(tensor, mask):
                tensor = tensor * mask
                for axis in reversed(range(1, len(tensor.shape))):
                    tensor, _ = torch.max(tensor, dim=axis)  # max over all dimensions except batch dim 0
                tensor = torch.mean(tensor)  # mean over batch
                return tensor
        else:
            raise ValueError(f"Unknown reduction {masked_reduction}")
    return masked_reduction


def define_angular_loss_nored(angle_range: float) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Defines a loss function that calculates the angular difference between two angles in the closer direction when
    working in modulo arithmetic with modulo angle_range.
    As a nored function it does not apply any reduction to the data, merely calculates the loss for every point.
    Args:
        angle_range: the value for modulo arithmetic, also the largest possible angle
    Returns: the loss function
    """
    def angular_loss_nored(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        prediction = torch.remainder(prediction, angle_range)
        label = torch.remainder(label, angle_range)
        return - torch.abs(torch.abs(prediction-label) - (angle_range/2)) + (angle_range/2)
    return angular_loss_nored


def unit_vector_loss(prediction: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
    """ Applies a normalisation to encourage the output vectors to have magnitude 1"""
    return torch.mean(torch.abs(torch.norm(prediction, dim=1, keepdim=True)-1))


angular_loss_vector = losses.mean_reduction(
                                losses.modify_prediction(rep_2d_pytorch.vector_2_angle)(
                                    define_angular_loss_nored(2*pi)))
angular_loss_da_vector = losses.mean_reduction(
                                   losses.modify_prediction(rep_2d_pytorch.da_vector_2_angle)(
                                       define_angular_loss_nored(pi)))

angular_loss_vector_unit = losses.combine_losses([
    (angular_loss_vector, (0, 2), (0, 1)),
    (unit_vector_loss, (0, 2), (0, 1))
])

angular_loss_da_vector_unit = losses.combine_losses([
    (angular_loss_da_vector, (0, 2), (0, 1)),
    (unit_vector_loss, (0, 2), (0, 1))
])
