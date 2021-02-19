import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from matplotlib.colors import hsv_to_rgb
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from clustering import Clustering
import ax
from math import pi

from typing import Optional

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


def evaluate(root_path: str, run_name: str, i_cv: int, clustering=None):
    """
    Runs validation and testing and returns the metrics dictionary
    :param root_path: Path to where the runs are saved
    :param run_name: Name of the run
    :param i_cv: The cross validation run in (0, 1, 2, 3) to evaluate
    :param clustering: a clustering object that overrides the default clustering parameters
    """
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
    """
    Finds the metrics averaged over the cross-validation runs
    :param root_path: Path to where the runs are saved
    :param run_name: Name of the run
    """
    all_cv_metrics = []
    for i_cv in (0, 1, 2, 3):
        test_metrics = evaluate(root_path, run_name, i_cv)
        all_cv_metrics.append(test_metrics)
    average_metrics = pd.concat(all_cv_metrics)
    average_metrics = average_metrics.groupby(average_metrics.index).mean()
    return average_metrics


def evaluate_all():
    """
    Evaluates all runs and saves the results as as csv file in results/instance_segmentation_test_metrics.csv
    """
    root_path = 'results/instance_segmentation'
    run_names = os.listdir(root_path)
    all_metrics = dict()
    for run_name in run_names:
        if os.path.isdir(os.path.join(root_path, run_name)):
            metrics = evaluate_average_cv(root_path, run_name)
            all_metrics[run_name] = metrics
    all_metrics = pd.DataFrame(all_metrics)
    all_metrics.to_csv('results/instance_segmentation_test_metrics.csv')


def visualise(root_path, run_name, n_images, i_cv=0, use_boundary=False):
    """
    Saves images of the test and validation in the root_path/run_name directory
    :param root_path: Path to where the runs are saved
    :param run_name: Name of the run
    :param n_images: How many images to save from each dataset
    :param i_cv: Which cross_validation run to use
    :param use_boundary: Whether to use boundary version of clustering
    """
    with torch.no_grad():
        run_name_parts = run_name.split('_')
        separate_input_channels = run_name_parts[-1] == 'separate'

        dirpath = os.path.join(root_path, run_name)
        data_module = InstanceSegmentationDataModule(i_cv, separate_input_channels, use_boundary=use_boundary)
        data_module.num_workers = 2
        module = load_module(dirpath, i_cv)

        data_module.setup()

        dataloaders = data_module.test_dataloader()
        dataloader_synthetic_val = dataloaders[0]
        dataloader_synthetic_test = dataloaders[1]
        dataloader_real_val = dataloaders[2]
        dataloader_real_test = dataloaders[3]

        iterator_synthetic_val = iter(dataloader_synthetic_val)
        iterator_synthetic_test = iter(dataloader_synthetic_test)
        iterator_real_val = iter(dataloader_real_val)
        iterator_real_test = iter(dataloader_real_test)

        data_iterators = {
            'synthetic_val': iterator_synthetic_val,
            'synthetic_test': iterator_synthetic_test,
            'real_val': iterator_real_val,
            'real_test': iterator_real_test}

        if not separate_input_channels:
            dataloader_original_val = dataloaders[4]
            dataloader_original_test = dataloaders[5]
            iterator_original_val = iter(dataloader_original_val)
            iterator_original_test = iter(dataloader_original_test)
            data_iterators['original_val'] = iterator_original_val
            data_iterators['original_test'] = iterator_original_test

        for dataset_name, dataset_iterator in data_iterators.items():
            image_i = 0
            while image_i < n_images:
                try:
                    batch = next(dataset_iterator)
                except StopIteration:
                    break
                if separate_input_channels:
                    batch_in = batch[:, 0:2, ...]
                    batch_label = batch[:, 2:, ...]
                else:
                    batch_in = batch[:, 0:1, ...]
                    batch_label = batch[:, 1:, ...]
                batch_prediction, all_separate_chromosomes = module(batch_in)

                batch_prediction_category = torch.argmax(batch_prediction[:, 0:3, ...], dim=1, keepdim=True)
                batch_prediction_dilated_intersection = (batch_prediction[:, 3:4, ...] > 0).type(module.dtype)
                batch_prediction_angle = module.repr_2_angle(batch_prediction[:, 4:, ...])

                for batch_elem_i in range(batch.shape[0]):
                    if image_i >= n_images:
                        break
                    if 'synthetic' in dataset_name:
                        label_category = batch_label[batch_elem_i, 0:1, ...].long()
                        label_dilated_intersection = batch_label[batch_elem_i, 1:2, ...]
                        label_angle = batch_label[batch_elem_i, 2:3, ...]
                        label_chromosomes = batch_label[batch_elem_i, 3:5, ...]
                        whole_image = return_visualisation_raw(
                            batch_in[batch_elem_i, :, ...].cpu().numpy(),
                            batch_prediction_category[batch_elem_i, :, ...].cpu().numpy(),
                            batch_prediction_dilated_intersection[batch_elem_i, :, ...].cpu().numpy(),
                            batch_prediction_angle[batch_elem_i, :, ...].cpu().numpy(),
                            label_category.cpu().numpy(),
                            label_dilated_intersection.cpu().numpy(),
                            label_angle.cpu().numpy()
                        )
                        plt.clf()
                        plt.imshow(whole_image)
                        plt.axis('off')
                        plt.savefig(os.path.join(root_path, run_name, f"{dataset_name}_cv{i_cv}_{image_i:03}_raw.png"),
                                    bbox_inches='tight',
                                    dpi=200)
                        
                        whole_image = return_visualisation_chromosomes(
                            batch_in[batch_elem_i, :, ...].cpu().numpy(),
                            all_separate_chromosomes[batch_elem_i],
                            label_chromosomes.cpu().numpy()
                        )
                        plt.clf()
                        plt.imshow(whole_image)
                        plt.axis('off')
                        plt.savefig(os.path.join(root_path, run_name, f"{dataset_name}_cv{i_cv}_{image_i:03}.png"),
                                    bbox_inches='tight',
                                    dpi=200)

                    if 'real' in dataset_name:
                        label_chromosomes = batch_label[batch_elem_i, :, ...]
                        whole_image = return_visualisation_raw(
                            batch_in[batch_elem_i, :, ...].cpu().numpy(),
                            batch_prediction_category[batch_elem_i, :, ...].cpu().numpy(),
                            batch_prediction_dilated_intersection[batch_elem_i, :, ...].cpu().numpy(),
                            batch_prediction_angle[batch_elem_i, :, ...].cpu().numpy(),
                            None,
                            None,
                            None
                        )
                        plt.clf()
                        plt.imshow(whole_image)
                        plt.axis('off')
                        plt.savefig(os.path.join(root_path, run_name, f"{dataset_name}_cv{i_cv}_{image_i:03}_raw.png"),
                                    bbox_inches='tight',
                                    dpi=200)
                        whole_image = return_visualisation_chromosomes(
                            batch_in[batch_elem_i, :, ...].cpu().numpy(),
                            all_separate_chromosomes[batch_elem_i],
                            label_chromosomes.cpu().numpy()
                        )
                        plt.clf()
                        plt.imshow(whole_image)
                        plt.axis('off')
                        plt.savefig(os.path.join(root_path, run_name, f"{dataset_name}_cv{i_cv}_{image_i:03}.png"),
                                    bbox_inches='tight',
                                    dpi=200)

                    if 'original' in dataset_name:
                        label_category = batch_label[batch_elem_i, 0:1, ...].long()
                        label_chromosomes = torch.cat([
                            torch.logical_or(torch.eq(label_category, 1), torch.eq(label_category, 3)),
                            torch.logical_or(torch.eq(label_category, 2), torch.eq(label_category, 3))
                        ])
                        whole_image = return_visualisation_raw(
                            batch_in[batch_elem_i, :, ...].cpu().numpy(),
                            batch_prediction_category[batch_elem_i, :, ...].cpu().numpy(),
                            batch_prediction_dilated_intersection[batch_elem_i, :, ...].cpu().numpy(),
                            batch_prediction_angle[batch_elem_i, :, ...].cpu().numpy(),
                            None,
                            None,
                            None
                        )
                        plt.clf()
                        plt.imshow(whole_image)
                        plt.axis('off')
                        plt.savefig(os.path.join(root_path, run_name, f"{dataset_name}_cv{i_cv}_{image_i:03}_raw.png"),
                                    bbox_inches='tight',
                                    dpi=200)
                        whole_image = return_visualisation_chromosomes(
                            batch_in[batch_elem_i, :, ...].cpu().numpy(),
                            all_separate_chromosomes[batch_elem_i],
                            label_chromosomes.cpu().numpy()
                        )
                        plt.clf()
                        plt.imshow(whole_image)
                        plt.axis('off')
                        plt.savefig(os.path.join(root_path, run_name, f"{dataset_name}_cv{i_cv}_{image_i:03}.png"),
                                    bbox_inches='tight',
                                    dpi=200)

                    image_i += 1


def return_visualisation_raw(in_image: np.ndarray,
                             prediction_category: np.ndarray,
                             prediction_dilated_intersection: np.ndarray,
                             prediction_angle: np.ndarray,
                             label_category: Optional[np.ndarray],
                             label_dilated_intersection: Optional[np.ndarray],
                             label_angle: Optional[np.ndarray]):
    padding = 2

    if any((label_category is None, label_dilated_intersection is None, label_angle is None)):
        n_rows = 1
    else:
        n_rows = 2
    n_columns = 4

    row_size = in_image.shape[1]
    col_size = in_image.shape[2]
    whole_row_size = row_size * n_rows + padding * (n_rows - 1)
    whole_col_size = col_size * n_columns + padding * (n_columns - 1)

    whole_image = np.ones((whole_row_size, whole_col_size, 3))

    def grid_slice(ir, ic):
        return slice((row_size+padding)*ir, row_size*(ir+1) + padding*ir),\
               slice((col_size+padding)*ic, col_size*(ic+1) + padding*ic),\
               slice(None)

    whole_image[grid_slice(0, 0)] = np.clip(np.mean(in_image, axis=0), 0, 1)[:, :, None]

    whole_image[grid_slice(0, 1)] = plt.get_cmap('tab10')(prediction_category[0])[:, :, 0:3]

    whole_image[grid_slice(0, 2)] = prediction_dilated_intersection[0, :, :, None] > 0

    prediction_rgb = angle_pi_to_rgb(prediction_angle[0])
    prediction_rgb *= prediction_category[0, :, :, None] == 1
    whole_image[grid_slice(0, 3)] = prediction_rgb

    if n_rows == 2:
        whole_image[grid_slice(1, 1)] = plt.get_cmap('tab10')(label_category[0])[:, :, 0:3]

        whole_image[grid_slice(1, 2)] = label_dilated_intersection[0, :, :, None] > 0

        label_rgb = angle_pi_to_rgb(label_angle[0])
        label_rgb *= label_category[0, :, :, None] == 1
        whole_image[grid_slice(1, 3)] = label_rgb
        
    return whole_image


def return_visualisation_chromosomes(in_image,
                                     prediction_chromosomes,
                                     label_chromosomes):
    padding = 2

    n_rows = 2
    n_columns = max(prediction_chromosomes.shape[0], label_chromosomes.shape[0]) + 1

    row_size = in_image.shape[1]
    col_size = in_image.shape[2]
    whole_row_size = row_size * n_rows + padding * (n_rows - 1)
    whole_col_size = col_size * n_columns + padding * (n_columns - 1)

    whole_image = np.ones((whole_row_size, whole_col_size, 3))

    def grid_slice(ir, ic):
        return slice((row_size + padding) * ir, row_size * (ir + 1) + padding * ir), \
               slice((col_size + padding) * ic, col_size * (ic + 1) + padding * ic), \
               slice(None)

    whole_image[grid_slice(0, 0)] = np.clip(np.mean(in_image, axis=0), 0, 1)[:, :, None]

    for i_prediction_chromosome in range(prediction_chromosomes.shape[0]):
        whole_image[grid_slice(0, i_prediction_chromosome + 1)] = \
            prediction_chromosomes[i_prediction_chromosome, :, :, None] > 0

    for i_label_chromosome in range(label_chromosomes.shape[0]):
        whole_image[grid_slice(1, i_label_chromosome + 1)] = \
            label_chromosomes[i_label_chromosome, :, :, None] > 0
    return whole_image


def angle_pi_to_rgb(angle: np.ndarray) -> np.ndarray:
    """
    Takes an np image with angles mod pi and creates an rgb visualisation.
    Args:
        param angle: 2D np array with angle values mod pi
    Returns: rgb image, channel last
    """
    angle = np.remainder(angle, pi)/pi  # normalise to [0, 1)
    rgb = hsv_to_rgb(np.stack([angle, np.ones_like(angle), np.ones_like(angle)], axis=2))
    return rgb


def visualise_all(n_images):
    root_path = 'results/instance_segmentation'
    run_names = os.listdir(root_path)
    for run_name in run_names:
        print(run_name)
        if os.path.isdir(os.path.join(root_path, run_name)):
            visualise(root_path, run_name, n_images, 0)


def visualise_all_boundary(n_images):
    root_path = 'results/instance_segmentation_boundary'
    run_names = os.listdir(root_path)
    for run_name in run_names:
        print(run_name)
        if os.path.isdir(os.path.join(root_path, run_name)):
            visualise(root_path, run_name, n_images, 0, use_boundary=True)


if __name__ == '__main__':
    # evaluate_all()
    # optimise_clustering()
    # visualise_all(10)

    root_path = 'results/instance_segmentation_boundary'
    run_name = 'da_vector_lnet_averaged'
    visualise(root_path, run_name, 10, 0, True)
    # metrics = evaluate(root_path, run_name, 0)
