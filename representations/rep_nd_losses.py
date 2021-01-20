from representations import rep_nd_pytorch
import losses
# EXAMPLE COMPOUND LOSSES
loss_mse_aligned_x = losses.mean_reduction(
                         losses.modify_label(lambda label: rep_nd_pytorch.align_vector_space(label, 0))(
                             losses.mse_nored))
loss_mse_aligned_y = losses.mean_reduction(
                         losses.modify_label(lambda label: rep_nd_pytorch.align_vector_space(label, 1))(
                             losses.mse_nored))
loss_mse_aligned_z = losses.mean_reduction(
                         losses.modify_label(lambda label: rep_nd_pytorch.align_vector_space(label, 2))(
                             losses.mse_nored))

loss_piecewise_mse_adjusted_linear_choice_from_vector = \
    losses.mean_reduction(
        losses.modify_label(rep_nd_pytorch.vector_2_piecewise_adjusted_linear_choice)(
            losses.mse_nored))
loss_piecewise_mse_adjusted_smooth_choice_from_vector = \
    losses.mean_reduction(
        losses.modify_label(rep_nd_pytorch.vector_2_piecewise_adjusted_smooth_choice)(
            losses.mse_nored))
loss_piecewise_mse_adjusted_sum_from_vector = \
    losses.mean_reduction(
        losses.modify_label(rep_nd_pytorch.vector_2_piecewise_adjusted_linear_choice)(
            losses.mse_nored))


# EXAMPLE COMPOUND METRICS
metric_mean_piecewise_choice = losses.mean_reduction(
                                   losses.metric_piecewise_decorator(
                                       rep_nd_pytorch.piecewise_choice_2_vector))
metric_max_piecewise_choice = losses.max_reduction(
                                  losses.metric_piecewise_decorator(rep_nd_pytorch.piecewise_choice_2_vector))
metric_mean_piecewise_sum = losses.mean_reduction(
                                losses.metric_piecewise_decorator(
                                    rep_nd_pytorch.piecewise_sum_2_vector))
metric_max_piecewise_sum = losses.max_reduction(
                               losses.metric_piecewise_decorator(
                                   rep_nd_pytorch.piecewise_sum_2_vector))

metric_mean_aligned_x = losses.mean_reduction(
                            losses.modify_label(lambda label: rep_nd_pytorch.align_vector_space(label, 0))(
                                    losses.cosine_similarity_angle))
metric_mean_aligned_y = losses.mean_reduction(
                            losses.modify_label(lambda label: rep_nd_pytorch.align_vector_space(label, 1))(
                                losses.cosine_similarity_angle))
metric_mean_aligned_z = losses.mean_reduction(
                            losses.modify_label(lambda label: rep_nd_pytorch.align_vector_space(label, 2))(
                                losses.cosine_similarity_angle))
metric_max_aligned_x = losses.max_reduction(
                           losses.modify_label(lambda label: rep_nd_pytorch.align_vector_space(label, 0))(
                               losses.cosine_similarity_angle))
metric_max_aligned_y = losses.max_reduction(
                           losses.modify_label(lambda label: rep_nd_pytorch.align_vector_space(label, 1))(
                               losses.cosine_similarity_angle))
metric_max_aligned_z = losses.max_reduction(
                           losses.modify_label(lambda label: rep_nd_pytorch.align_vector_space(label, 2))(
                               losses.cosine_similarity_angle))
