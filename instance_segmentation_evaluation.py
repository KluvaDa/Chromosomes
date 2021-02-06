import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from clustering import Clustering
import ax

from instance_segmentation import InstanceSegmentationModule, InstanceSegmentationDataModule


def load_module(dirpath: str, cross_validation: int):
    """
    Loads a pytorch lightning module. Always loads version_0
    :param dirpath: String path
    :param cross_validation: Which cross validation run to ue.
    :return:
    """
    # find the first checkpoint for best
    checkpoints_dir = os.path.join(dirpath, f'cv{cross_validation}', 'version_0', 'checkpoints')
    checkpoints = os.listdir(checkpoints_dir)
    checkpoint_name = ''
    for checkpoint_name in checkpoints:
        if checkpoint_name[0:4] == 'best' and checkpoint_name[-5:] == '.ckpt':
            break
    checkpoint = os.path.join(checkpoints_dir, checkpoint_name)

    # load module
    instance_segmentation_module = InstanceSegmentationModule.load_from_checkpoint(checkpoint)
    return instance_segmentation_module


def optimise_clustering():
    clustering_parameters = [
        {
            'name': 'minimum_dilated_intersection_area',
            'type': 'range',
            'bounds': [2, 64],
            'value_type': 'int',
            'log_scale': True
        },
        {
            'name': 'max_distance',
            'type': 'range',
            'bounds': [4, 16],
            'value_type': 'int',
            'log_scale': True
        },
        {
            'name': 'merge_peaks_distance',
            'type': 'range',
            'bounds': [0, 4],
            'value_type': 'int',
            'log_scale': False
        },
        {
            'name': 'minimum_clusters_area',
            'type': 'range',
            'bounds': [2, 32],
            'value_type': 'int',
            'log_scale': True
        },
        {
            'name': 'minimum_adjacent_area',
            'type': 'range',
            'bounds': [2, 32],
            'value_type': 'int',
            'log_scale': True
        },
        {
            'name': 'direction_sensitivity',
            'type': 'range',
            'bounds': [0, 1],
            'value_type': 'float',
            'log_scale': False
        },
        {
            'name': 'cluster_grow_radius',
            'type': 'range',
            'bounds': [1, 8],
            'value_type': 'float',
            'log_scale': False
        },
        {
            'name': 'max_chromosome_width',
            'type': 'range',
            'bounds': [10, 40],
            'value_type': 'int',
            'log_scale': True
        },
        {
            'name': 'intersection_grow_radius',
            'type': 'range',
            'bounds': [1, 8],
            'value_type': 'float',
            'log_scale': False
        },
        {
            'name': 'direction_local_weight',
            'type': 'range',
            'bounds': [0, 1],
            'value_type': 'float',
            'log_scale': False
        }
    ]
    best_parameters, best_values, experiment, model = ax.optimize(clustering_parameters,
                                                                  optimisation_function,
                                                                  minimize=False,
                                                                  total_trials=1000)
    with open('results/clustering_optimisation.txt', 'w') as f:
        f.write(str(best_parameters))
        f.write('\n')
        f.write(str(best_values))


def optimisation_function(clustering_parameters):
    root_path = 'results/instance_segmentation'
    run_name = 'da_vector_lnet_separate'
    i_cv = 0
    clustering = Clustering(**clustering_parameters)
    metrics = evaluate(root_path, run_name, i_cv, clustering)
    main_metric = metrics['val_synthetic_iou_separate_chromosomes/dataloader_idx_0'] * 0.5 + \
                  metrics['val_real_iou_separate_chromosomes/dataloader_idx_2'] * 0.5
    return main_metric


def evaluate(root_path:str, run_name: str, i_cv: int, clustering=None):
    run_name_parts = run_name.split('_')
    separate_input_channels = run_name_parts[-1] == 'separate'

    dirpath = os.path.join(root_path, run_name)
    data_module = InstanceSegmentationDataModule(i_cv, separate_input_channels)
    module = load_module(dirpath, i_cv)
    if clustering is not None:
        module.clustering = clustering
    trainer = pl.Trainer(gpus=1)
    test_metrics = trainer.test(module, datamodule=data_module)
    test_metrics = pd.DataFrame(test_metrics)
    test_metrics = test_metrics.mean()  # no repetition, should merely flatten
    return test_metrics


def evaluate_average_cv(root_path: str, run_name: str):
    all_cv_metrics = []
    for i_cv in (0, 1, 2, 3):
        test_metrics = evaluate(root_path, run_name, i_cv)
        all_cv_metrics.append(test_metrics)
    average_metrics = pd.concat(all_cv_metrics)
    average_metrics = average_metrics.groupby(average_metrics.index).mean()
    return average_metrics


def evaluate_all():
    root_path = 'results/instance_segmentation'
    run_names = os.listdir(root_path)
    all_metrics = dict()
    for run_name in run_names:
        if os.path.isdir(os.path.join(root_path, run_name)):
            metrics = evaluate_average_cv(root_path, run_name)
            all_metrics[run_name] = metrics
    all_metrics = pd.DataFrame(all_metrics)
    all_metrics.to_csv('results/instance_segmentation_test_metrics.csv')


if __name__ == '__main__':
    # evaluate_all()
    optimise_clustering()
