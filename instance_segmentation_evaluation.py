import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

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


def evaluate(root_path: str, run_name: str):
    run_name_parts = run_name.split('_')
    separate_input_channels = run_name_parts[-1] == 'separate'

    dirpath = os.path.join(root_path, run_name)
    all_cv_metrics = []
    for i_cv in (0, 1, 2, 3):
        data_module = InstanceSegmentationDataModule(i_cv, separate_input_channels)
        module = load_module(dirpath, i_cv)
        trainer = pl.Trainer(gpus=1)
        test_metrics = trainer.test(module, datamodule=data_module)
        test_metrics = pd.DataFrame(test_metrics)
        test_metrics = test_metrics.mean()  # no repetition, should merely flatten
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
            metrics = evaluate(root_path, run_name)
            all_metrics[run_name] = metrics
    all_metrics = pd.DataFrame(all_metrics)
    all_metrics.to_csv('results/instance_segmentation_test_metrics.csv')


if __name__ == '__main__':
    evaluate_all()
