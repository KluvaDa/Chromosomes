from clustering import Clustering
import ax
from instance_segmentation_evaluation import evaluate


def optimise_clustering():
    """
    Performs an optimisation of the clustering parameters and saves the results in 'results/clustering_optimisation.txt
    """
    clustering_parameters = [
        {
            'name': 'minimum_dilated_intersection_area',
            'type': 'range',
            'bounds': [24, 36],
            'value_type': 'int',
            'log_scale': True
        },
        {
            'name': 'max_distance',
            'type': 'range',
            'bounds': [3, 5],
            'value_type': 'int',
            'log_scale': False
        },
        {
            'name': 'merge_peaks_distance',
            'type': 'range',
            'bounds': [1, 3],
            'value_type': 'int',
            'log_scale': False
        },
        {
            'name': 'minimum_clusters_area',
            'type': 'range',
            'bounds': [8, 16],
            'value_type': 'int',
            'log_scale': True
        },
        {
            'name': 'minimum_adjacent_area',
            'type': 'range',
            'bounds': [6, 14],
            'value_type': 'int',
            'log_scale': True
        },
        {
            'name': 'direction_sensitivity',
            'type': 'range',
            'bounds': [0.7, 0.95],
            'value_type': 'float',
            'log_scale': False
        },
        {
            'name': 'cluster_grow_radius',
            'type': 'fixed',
            'value': 1.2
        },
        {
            'name': 'max_chromosome_width',
            'type': 'range',
            'bounds': [6, 16],
            'value_type': 'int',
            'log_scale': True
        },
        {
            'name': 'intersection_grow_radius',
            'type': 'fixed',
            'value': 1.2
        },
        {
            'name': 'direction_local_weight',
            'type': 'range',
            'bounds': [0.8, 1],
            'value_type': 'float',
            'log_scale': False
        }
    ]
    best_parameters, best_values, experiment, model = ax.optimize(clustering_parameters,
                                                                  optimisation_function,
                                                                  minimize=False,
                                                                  total_trials=100)
    with open('results/clustering_optimisation.txt', 'w') as f:
        f.write(str(best_parameters))
        f.write('\n')
        f.write(str(best_values))


def optimisation_function(clustering_parameters):
    """ Takes the parameters for clustering and returns an overall metric by validating on da_vector_lnet_separate"""
    root_path = 'results/instance_segmentation'
    run_name = 'da_vector_lnet_separate'
    clustering = Clustering(**clustering_parameters)
    main_metric_average = 0
    for i_cv in range(4):
        metrics = evaluate(root_path, run_name, i_cv, clustering)
        main_metric = \
            metrics['val_synthetic_iou_separate_chromosomes/dataloader_idx_0'] * 0.5 \
            + metrics['val_real_iou_separate_chromosomes/dataloader_idx_2'] * 0.5 \
            - abs(metrics['val_synthetic_n_chromosomes_difference/dataloader_idx_0']) * 0.05 \
            - abs(metrics['val_real_n_chromosomes_difference/dataloader_idx_2']) * 0.05
        main_metric_average += main_metric
    main_metric_average /= 4
    return main_metric_average


if __name__ == '__main__':
    optimise_clustering()
