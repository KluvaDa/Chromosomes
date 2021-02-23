import torch
import pytorch_lightning as pl
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from typing import Dict

from semantic_segmentation import ClassificationModule, ClassificationDataModule


def interpret_dirname(dirname: str):
    dirname_parts = dirname.split('_')
    network_identifier = dirname_parts[-1]
    dataset_identifier = '_'.join(dirname_parts[:-1])
    return dataset_identifier, network_identifier


def load_module(dirpath: str, cross_validation: int):
    """
    Loads the pytorch lightning module. Always loads version_0.
    :param dirpath: String path
    :param cross_validation: Which cross validation run to use.
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
    classification_module = ClassificationModule.load_from_checkpoint(checkpoint)
    return classification_module


def evaluate(root_path: str, dirname: str, i_cv: int):
    """
    Runs validation and testing and returns the metrics dictionary
    :param root_path: Path to where the runs are saved
    :param dirname: Name of the run
    :param i_cv: The cross validation run in (0, 1, 2, 3) to evaluate
    """
    dataset_identifier, net_identifier = interpret_dirname(dirname)
    dirpath = os.path.join(root_path, dirname)

    data_module = ClassificationDataModule(dataset_identifier=dataset_identifier, cross_validation_i=i_cv)
    module = load_module(dirpath, i_cv)
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
    root_path = 'results/semantic_segmentation'
    run_names = os.listdir(root_path)
    all_metrics = dict()
    for run_name in run_names:
        if os.path.isdir(os.path.join(root_path, run_name)):
            metrics = evaluate_average_cv(root_path, run_name)
            all_metrics[run_name] = metrics
    all_metrics = pd.DataFrame(all_metrics)
    all_metrics.to_csv('results/semantic_segmentation_test_metrics.csv')


def save_images(root_path, dirname, n_images, i_cv=0):
    dataset_identifier, net_identifier = interpret_dirname(dirname)
    dirpath = os.path.join(root_path, dirname)

    data_module = ClassificationDataModule(dataset_identifier=dataset_identifier, cross_validation_i=i_cv)
    module = load_module(dirpath, i_cv)

    data_module.setup()
    dataloader_synthetic, dataloader_real, dataloader_original = data_module.val_dataloader()
    iterator_synthetic = iter(dataloader_synthetic)
    iterator_real = iter(dataloader_real)
    iterator_original = iter(dataloader_original)
    data_iterators = {'synthetic': iterator_synthetic, 'real': iterator_real, 'original': iterator_original}
    for dataset_name, dataset_iterator in data_iterators.items():
        image_i = 0
        while image_i < n_images:
            try:
                batch = next(dataset_iterator)
            except StopIteration:
                break
            if image_i >= n_images and dataset_name != 'real':
                break
            batch_in = batch[:, 0:1, ...]
            batch_label = batch[:, 1:, ...]
            batch_prediction = torch.argmax(module(batch_in), dim=1, keepdim=True)

            batch_in = batch_in.detach().cpu().numpy()
            batch_label = batch_label.detach().cpu().numpy()
            batch_prediction = batch_prediction.detach().cpu().numpy()

            for image_in, image_label, image_prediction in zip(batch_in, batch_label, batch_prediction):
                if image_i >= n_images:
                    break
                plt.clf()

                plt.subplot(131)
                plt.imshow(image_in[0], cmap='gray', vmin=0, vmax=1)
                plt.axis('off')

                plt.subplot(132)
                plt.imshow(image_label[0].astype(int), cmap='tab10', vmin=0, vmax=9)
                plt.axis('off')

                plt.subplot(133)
                plt.imshow(image_prediction[0], cmap='tab10', vmin=0, vmax=9)
                plt.axis('off')

                plt.savefig(os.path.join(root_path, dirname, f"{dataset_name}_cv{i_cv}_{image_i:03}.png"),
                            bbox_inches='tight',
                            dpi=200)

                image_i += 1


def save_all_images(n_images):
    root_path = 'results/semantic_segmentation'
    run_names = os.listdir(root_path)
    for run_name in run_names:
        print(run_name)
        if os.path.isdir(os.path.join(root_path, run_name)):
            save_images(root_path, run_name, n_images, 0)


if __name__ == '__main__':
    evaluate_all()
    save_all_images(10)
