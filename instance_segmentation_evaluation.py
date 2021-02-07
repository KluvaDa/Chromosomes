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
    root_path = 'results/instance_segmentation'
    run_name = 'da_vector_lnet_separate'
    clustering = Clustering(**clustering_parameters)
    main_metric_average = 0
    for i_cv in range(4):
        metrics = evaluate(root_path, run_name, i_cv, clustering)
        main_metric = metrics['val_synthetic_iou_separate_chromosomes/dataloader_idx_0'] * 0.5 + \
                      metrics['val_real_iou_separate_chromosomes/dataloader_idx_2'] * 0.5
        main_metric_average += main_metric
    main_metric_average /= 4
    return main_metric_average


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


# def save_images(root_path, run_name, n_images, i_cv=0):
#     run_name_parts = run_name.split('_')
#     separate_input_channels = run_name_parts[-1] == 'separate'
#     smaller_network = run_name_parts[-2] == 'snet'
#     representation = '_'.join(run_name_parts[:-2])
#
#     dirpath = os.path.join(root_path, run_name)
#     data_module = InstanceSegmentationDataModule(i_cv, separate_input_channels)
#     module = load_module(dirpath, i_cv)
#
#     data_module.setup()
#
#     dataloaders = data_module.test_dataloader()
#     dataloader_synthetic_val = dataloaders[0]
#     dataloader_synthetic_test = dataloaders[1]
#     dataloader_real_val = dataloaders[2]
#     dataloader_real_test = dataloaders[3]
#
#     iterator_synthetic_val = iter(dataloader_synthetic_val)
#     iterator_synthetic_test = iter(dataloader_synthetic_test)
#     iterator_real_val = iter(dataloader_real_val)
#     iterator_real_test = iter(dataloader_real_test)
#
#     data_iterators = {
#         'synthetic_val': iterator_synthetic_val,
#         'synthetic_test': iterator_synthetic_test,
#         'real_val': iterator_real_val,
#         'real_test': iterator_real_test}
#
#     if not separate_input_channels:
#         dataloader_original_val = dataloaders[4]
#         dataloader_original_test = dataloaders[5]
#         iterator_original_val = iter(dataloader_original_val)
#         iterator_original_test = iter(dataloader_original_test)
#         data_iterators['original_val'] = iterator_original_val
#         data_iterators['original_test'] = iterator_original_test
#
#     for dataset_name, dataset_iterator in data_iterators.items():
#         image_i = 0
#         while image_i < n_images:
#             batch = next(dataset_iterator)
#             if image_i >= n_images:
#                 break
#             if separate_input_channels:
#                 batch_in = batch[:, 0:2, ...]
#                 batch_label = batch[:, 2:, ...]
#             else:
#                 batch_in = batch[:, 0:1, ...]
#                 batch_label = batch[:, 1:, ...]
#             batch_prediction, all_separate_chromosomes = module(batch_in)
#
#             batch_in = batch_in.detach().cpu().numpy()
#             batch_label = batch_label.detach().cpu().numpy()
#             batch_prediction = batch_prediction.detach().cpu().numpy()
#
#             for image_in, image_label, image_prediction, prediction_separate_chromosomes in \
#                     zip(batch_in, batch_label, batch_prediction, all_separate_chromosomes):
#                 prediction_separate_chromosomes = prediction_separate_chromosomes.detach().cpu().numpy()
#
#                 prediction_category = np.argmax(image_prediction[0:3], axis=0)
#                 prediction_angle = module.repr_2_angle()
#                 if image_i >= n_images:
#                     break
#
#                 # plot raw
#                 plt.clf()
#
#                 plt.subplot(231)
#                 plt.imshow(image_in[0], cmap='gray', vmin=0, vmax=1)
#                 plt.axis('off')
#
#                 if separate_input_channels:
#                     plt.subplot(234)
#                     plt.imshow(image_in[1], cmap='gray', vmin=0, vmax=1)
#                     plt.axis('off')
#
#                 if not separate_input_channels:
#                     plt.subplot(232)
#                     plt.imshow(image_label[0].astype(int), cmap='tab10', vmin=0, vmax=9)
#                     plt.axis('off')
#
#                 plt.subplot(233)
#                 plt.imshow(prediction_category, cmap='tab10', vmin=0, vmax=9)
#                 plt.axis('off')
#
#                 if not separate_input_channels:
#                     plt.subplot(235)
#                     plt.
#
#                 plt.savefig(os.path.join(root_path, run_name,f"{dataset_name}_cv{cross_validation}_{image_i:03}.png"),
#                             bbox_inches='tight',
#                             dpi=200)
#
#                 image_i += 1


# def save_all_images(n_images):
#     root_path = 'results/instance_segmentation'
#     run_names = os.listdir(root_path)
#     for run_name in run_names:
#         if os.path.isdir(os.path.join(root_path, run_name)):
#             save_images(root_path, run_name, n_images, 0)


if __name__ == '__main__':
    # evaluate_all()
    optimise_clustering()

    # clustering_parameters = {
    #     'minimum_dilated_intersection_area': 28,
    #     'max_distance': 7,
    #     'merge_peaks_distance': 2,
    #     'minimum_clusters_area': 10,
    #     'minimum_adjacent_area': 20,
    #     'direction_sensitivity': 0.43871344895396364,
    #     'cluster_grow_radius': 1.0,
    #     'max_chromosome_width': 27,
    #     'intersection_grow_radius': 1.2193580337740715,
    #     'direction_local_weight': 0.9175349822623694}
    #
    # clustering = Clustering(max_chromosome_width=10, direction_sensitivity=0.87, minimum_clusters_area=10,
    #                         merge_peaks_distance=1, max_distance=4, minimum_adjacent_area=8)
    # root_path = 'results/instance_segmentation'
    # run_name = 'da_vector_lnet_separate'
    # main_metric_average = 0
    # for i_cv in range(4):
    #     metrics = evaluate(root_path, run_name, i_cv, clustering)
    #     main_metric = metrics['val_synthetic_iou_separate_chromosomes/dataloader_idx_0'] * 0.5 + \
    #                   metrics['val_real_iou_separate_chromosomes/dataloader_idx_2'] * 0.5
    #     main_metric_average += main_metric
    # main_metric_average /= 4
    #
    # print(main_metric_average)
