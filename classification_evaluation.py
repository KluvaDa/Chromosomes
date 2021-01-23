import torch
import pytorch_lightning as pl
import os
import pathlib
import numpy as np
import pathlib

from tensorboard.backend.event_processing import event_accumulator
from typing import Dict

import classification
import tensorboard_reader


def interpret_dirname(dirname: str):
    # deduce parameters
    (train_on_original, segment_4_categories, smaller_network, shuffle_first, category_order) = \
        dirname.split('_')
    train_on_original = True if train_on_original == 'original' else False
    segment_4_categories = True if segment_4_categories == '4' else False
    smaller_network = True if smaller_network == 'snet' else False
    shuffle_first = True if shuffle_first == 'shuffled' else False
    # category order unchanged

    return train_on_original, segment_4_categories, smaller_network, shuffle_first, category_order


def load_module(dirpath: str):
    """
    Loads the pytorch lightning module. Always loads version_0.
    :param dirpath: String path
    """
    # find the first checkpoint for best
    checkpoints_dir = os.path.join(dirpath, 'version_0', 'checkpoints')
    checkpoints = os.listdir(checkpoints_dir)
    checkpoint_name = ''
    for checkpoint_name in checkpoints:
        if checkpoint_name[0:4] == 'best' and checkpoint_name[-5:] == '.ckpt':
            break
    checkpoint = os.path.join(checkpoints_dir, checkpoint_name)

    # load module
    classification_module = classification.ClassificationModule.load_from_checkpoint(checkpoint)
    return classification_module


def load_tensorboard(root_path: str):
    run_names = os.listdir(root_path)
    all_metrics = dict()
    for run_name in run_names:
        if len(run_name.split('_')) != 5:
            continue
        hyperparameters = interpret_dirname(run_name)
        train_on_original = hyperparameters[0]
        segment_4_categories = hyperparameters[1]
        if train_on_original:
            if segment_4_categories:
                main_metric = 'val_original_iou_best_match/dataloader_idx_2'
            else:
                main_metric = 'val_original_iou/dataloader_idx_2'
        else:
            if segment_4_categories:
                main_metric = 'val_synthetic_iou_best_match/dataloader_idx_0'
            else:
                main_metric = 'val_synthetic_iou/dataloader_idx_0'

        scalars_list = tensorboard_reader.read_scalars_in_directory(os.path.join(root_path, run_name))
        metrics = tensorboard_reader.find_best_metrics_averaged(scalars_list, main_metric, maximise=True)
        all_metrics[run_name] = metrics
    return all_metrics



if __name__ == '__main__':
    root_path = 'results/classification'
    all_metrics = load_tensorboard(root_path)

    runs = os.listdir(root_path)
    run = runs[0]

    # classification_module = load_module(os.path.join(parent_dir, runs[0]))
    # print(classification_module.segment_4_categories)
