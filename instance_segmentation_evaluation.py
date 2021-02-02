import torch
import pytorch_lightning as pl
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas

import instance_segmentation


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
    instance_segmentation_module = instance_segmentation.InstanceSegmentationModule.load_from_checkpoint(checkpoint)
    return instance_segmentation_module

def evaluate(root_path: str, run_name: str):
    pass

def evaluate_all():
    root_path = 'results/instance_segmentation'
    run_names = os.listdir(root_path)
    for run_name in run_names:
        if os.path.isdir(os.path.join(root_path, run_name)):
            evaluate(root_path, run_name)



