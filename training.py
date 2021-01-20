import os
from abc import ABC

import numpy as np
import torch
import torch.nn
import torch.nn.functional
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning.metrics.functional.classification

from typing import Sequence, Tuple, Union, Optional
from torch.utils.data import DataLoader

import networks
import datasets


class ClassificationDataModule(pl.LightningDataModule):
    """
    A DataModule that loads all the datasets with defaults preset for classification purposes.
    It outputs the relevant training dataloader, and for validation and testing th e
    """
    def __init__(self,
                 train_on_original: bool,
                 cross_validation_i: int,
                 segment_4_categories: bool,
                 shuffle_first: bool,
                 category_order: str = 'random',
                 ):
        """
        :param train_on_original: Whether to use the original dataset to train, or the new synthetic
        :param cross_validation_i: An integer in [0, 4) specifying which cross-validation split to select.
        :param segment_4_categories: Whether to segment ch0 and ch1 separately (True), or together as one class (False)
        :param shuffle_first: Whether to shuffle the images in the dataset before selecting the data_subset_ranges
        :param category_order: How to determine ch1 and ch2. 'random', 'position', 'orientation', 'length'
        """
        super().__init__()
        self.train_on_original = train_on_original
        self.cross_validation_i = cross_validation_i
        self.segment_4_categories = segment_4_categories
        self.shuffle_first = shuffle_first
        self.category_order = category_order

        # Default params
        self.batchsize = 64
        self.dtype = np.float32
        self.num_workers = 4

        # new synthetic dataset params
        self.imsize_synthetic = (128, 128)  # half resolution will half this
        self.train_batches_per_epoch = 128
        self.val_batches_per_epoch = 8
        self.test_batches_per_epoch = 8

        self.filepath_real = os.path.join('data', 'real_overlapping.pickle')
        self.filepath_new_synthetic = os.path.join('data', 'separate.pickle')
        self.filepath_original = os.path.join('data', 'Cleaned_LowRes_13434_overlapping_pairs.h5')

        # placeholders
        self.dataset_original_train = None
        self.dataset_original_val = None
        self.dataset_original_test = None

        self.dataset_synthetic_train = None
        self.dataset_synthetic_val = None
        self.dataset_synthetic_test = None

        self.dataset_real_val = None
        self.dataset_real_test = None

    def setup(self, stage=None):
        # original dataset
        if self.train_on_original:
            train_subsets_cv = {0: [(0.0, 0.6)],
                                1: [(0.0, 0.4), (0.6, 0.8)],
                                2: [(0.0, 0.2), (0.4, 0.8)],
                                3: [(0.2, 0.8)]}
            val_subsets_cv = {0: [(0.6, 0.8)],
                              1: [(0.4, 0.6)],
                              2: [(0.2, 0.4)],
                              3: [(0.0, 0.2)]}
            train_subset = train_subsets_cv[self.cross_validation_i]
            val_subset = val_subsets_cv[self.cross_validation_i]
            test_subset = [(0.8, 1.0)]
            self.dataset_original_train = datasets.OriginalChromosomeDataset(self.filepath_original,
                                                                             train_subset,
                                                                             self.segment_4_categories,
                                                                             self.shuffle_first,
                                                                             self.batchsize,
                                                                             self.dtype)
            self.dataset_original_val = datasets.OriginalChromosomeDataset(self.filepath_original,
                                                                           val_subset,
                                                                           self.segment_4_categories,
                                                                           self.shuffle_first,
                                                                           self.batchsize,
                                                                           self.dtype)
            self.dataset_original_test = datasets.OriginalChromosomeDataset(self.filepath_original,
                                                                            test_subset,
                                                                            self.segment_4_categories,
                                                                            self.shuffle_first,
                                                                            self.batchsize,
                                                                            self.dtype)
        # new synthetic dataset
        train_slides_cv = {0: (0, 1, 2,  3, 4, 5,  6, 7, 8),
                           1: (0, 1, 2,  3, 4, 5,  9, 10, 11),
                           2: (0, 1, 2,  6, 7, 8,  9, 10, 11),
                           3: (3, 4, 5,  6, 7, 8,  9, 10, 11)}
        val_slides_cv = {0: (9, 10, 11),
                         1: (6, 7, 8),
                         2: (3, 4, 5),
                         3: (0, 1, 2)}
        train_slides = train_slides_cv[self.cross_validation_i]
        val_slides = val_slides_cv[self.cross_validation_i]
        test_slides = [(12, 13, 14)]

        if self.segment_4_categories:
            output_channels_list = ['dapi_cy3', '4_channel']
        else:
            output_channels_list = ['dapi_cy3', '3_channel']

        if not self.train_on_original:
            self.dataset_synthetic_train = datasets.SyntheticChromosomeDataset(self.filepath_new_synthetic,
                                                                               self.imsize_synthetic,
                                                                               train_slides,
                                                                               True,
                                                                               self.batchsize,
                                                                               self.train_batches_per_epoch,
                                                                               output_channels_list,
                                                                               self.category_order,
                                                                               fix_random_seed=False,
                                                                               dtype=self.dtype)
        self.dataset_synthetic_val = datasets.SyntheticChromosomeDataset(self.filepath_new_synthetic,
                                                                         self.imsize_synthetic,
                                                                         val_slides,
                                                                         True,
                                                                         self.batchsize,
                                                                         self.val_batches_per_epoch,
                                                                         output_channels_list,
                                                                         self.category_order,
                                                                         fix_random_seed=True,
                                                                         dtype=self.dtype)
        self.dataset_synthetic_test = datasets.SyntheticChromosomeDataset(self.filepath_new_synthetic,
                                                                          self.imsize_synthetic,
                                                                          val_slides,
                                                                          True,
                                                                          self.batchsize,
                                                                          self.test_batches_per_epoch,
                                                                          output_channels_list,
                                                                          self.category_order,
                                                                          fix_random_seed=True,
                                                                          dtype=self.dtype)

        # real dataset
        self.dataset_real_val = datasets.RealOverlappingChromosomes(self.filepath_real,
                                                                    [0, 1, 2, 3, 4, 5, 6],
                                                                    separate_channels=False,
                                                                    half_resolution=True,
                                                                    dtype=self.dtype)
        self.dataset_real_test = datasets.RealOverlappingChromosomes(self.filepath_real,
                                                                     [7, 8, 9, 10, 11, 12, 13, 14],
                                                                     separate_channels=False,
                                                                     half_resolution=True,
                                                                     dtype=self.dtype)

    def train_dataloader(self):
        if self.train_on_original:
            return DataLoader(self.dataset_original_train,
                              batch_size=None,
                              num_workers=self.num_workers,
                              pin_memory=self.num_workers > 0)
        else:
            return DataLoader(self.dataset_synthetic_train,
                              batch_size=None,
                              num_workers=self.num_workers,
                              pin_memory=self.num_workers > 0)

    def val_dataloader(self):
        dataloader_synthetic = DataLoader(self.dataset_synthetic_val,
                                          batch_size=None,
                                          num_workers=self.num_workers,
                                          pin_memory=self.num_workers > 0)
        dataloader_real = DataLoader(self.dataset_real_val,
                                     batch_size=None,
                                     num_workers=self.num_workers,
                                     pin_memory=self.num_workers > 0)
        if self.train_on_original:
            dataloader_original = DataLoader(self.dataset_original_val,
                                             batch_size=None,
                                             num_workers=self.num_workers,
                                             pin_memory=self.num_workers > 0)
            return dataloader_synthetic, dataloader_real, dataloader_original
        else:
            return dataloader_synthetic, dataloader_real

    def test_dataloader(self):
        dataloader_synthetic = DataLoader(self.dataset_synthetic_test,
                                          batch_size=None,
                                          num_workers=self.num_workers,
                                          pin_memory=self.num_workers > 0)
        dataloader_real = DataLoader(self.dataset_real_test,
                                     batch_size=None,
                                     num_workers=self.num_workers,
                                     pin_memory=self.num_workers > 0)
        if self.train_on_original:
            dataloader_original = DataLoader(self.dataset_original_test,
                                             batch_size=None,
                                             num_workers=self.num_workers,
                                             pin_memory=self.num_workers > 0)
            return dataloader_synthetic, dataloader_real, dataloader_original
        else:
            return dataloader_synthetic, dataloader_real


class ClassificationModule(pl.LightningModule):
    def __init__(self, segment_4_categories: bool):
        """
        Module with hard coded parameters for everything.
        :param segment_4_categories: Whether to segment ch0 and ch1 separately (True), or together as one class (False)
        """
        super().__init__()
        self.segment_4_categories = segment_4_categories
        self.n_categories = 4 if segment_4_categories else 3

        # hardcoded parameters
        backbone_net = networks.FullyConv(32, [64, 64, 32], torch.nn.functional.relu, 3, 1, torch.nn.BatchNorm2d, False)
        n_channels_out = 4 if segment_4_categories else 3
        self.net = networks.Unet(1, n_channels_out, 16, 2, 2, 3, backbone_net, None, None, False)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_step, ):
        batch_prediction, batch_label, iou, iou_channels = self._step_core(batch)
        loss = torch.nn.functional.cross_entropy(batch_prediction, batch_label)
        metrics = self._name_metrics(iou, iou_channels, 'training')
        metrics['loss'] = loss
        return {'loss': loss, 'metrics': metrics}

    def training_epoch_end(self, step_outputs):
        self._log_metrics(step_outputs)

    def validation_step(self, batch, batch_step, dataloader_idx):
        batch_prediction, batch_label, iou, iou_channels = self._step_core(batch)

        if dataloader_idx == 0:
            dataset_name = 'val_synthetic'
        elif dataloader_idx == 1:
            dataset_name = 'val_real'
        else:
            dataset_name = 'val_original'
        metrics = self._name_metrics(iou, iou_channels, dataset_name)
        return {'metrics': metrics}

    def validation_epoch_end(self, step_outputs_multi_dataset):
        for step_outputs in step_outputs_multi_dataset:
            self._log_metrics(step_outputs)

    def test_step(self, batch, batch_step, dataloader_idx):
        batch_prediction, batch_label, iou, iou_channels = self._step_core(batch)

        if dataloader_idx == 0:
            dataset_name = 'val_synthetic'
        elif dataloader_idx == 1:
            dataset_name = 'val_real'
        else:
            dataset_name = 'val_original'
        metrics = self._name_metrics(iou, iou_channels, dataset_name)
        return {'metrics': metrics}

    def test_epoch_end(self, step_outputs):
        self._log_metrics(step_outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def _step_core(self, batch):
        batch_in = batch[:, 0:1, ...]
        batch_label = batch[:, 1, ...].long()
        batch_prediction = self.net(batch_in)

        batch_prediction_max = torch.argmax(batch_prediction, dim=1)

        iou = pytorch_lightning.metrics.functional.classification.iou(
            batch_prediction_max,
            batch_label,
            ignore_index=None,
            num_classes=self.n_categories
        )
        iou_channels = []
        for i in range(self.n_categories):
            iou_channels.append(
                pytorch_lightning.metrics.functional.classification.iou(
                    torch.eq(batch_prediction_max, i).long(),
                    torch.eq(batch_label, i).long(),
                    ignore_index=0,
                    num_classes=2
                )
            )

        return batch_prediction, batch_label, iou, iou_channels

    def _name_metrics(self, iou, iou_channels, dataset_name):
        metrics = {dataset_name+'_iou': iou,
                   dataset_name+'_iou_background': iou_channels[0]}
        if self.segment_4_categories:
            metrics[dataset_name+'_iou_ch0'] = iou_channels[1]
            metrics[dataset_name+'_iou_ch1'] = iou_channels[2]
            metrics[dataset_name+'_iou_overlap'] = iou_channels[3]
        else:
            metrics[dataset_name+'_iou_ch'] = iou_channels[1]
            metrics[dataset_name+'_iou_overlap'] = iou_channels[2]
        return metrics

    def _log_metrics(self, step_outputs):
        logs_mean = {}
        for metric_name in step_outputs[0]['metrics']:
            metric_list = [output['metrics'][metric_name] for output in step_outputs]
            logs_mean[metric_name] = torch.mean(torch.stack(metric_list))

        self.log_dict(logs_mean)

if __name__ == '__main__':
    train_on_original = True
    cross_validation_i = 0
    segment_4_categories = True
    shuffle_first = False
    category_order = 'random'

    classification_module = ClassificationModule(segment_4_categories)
    classification_data_module = ClassificationDataModule(train_on_original, cross_validation_i, segment_4_categories,
                                                          shuffle_first, category_order)
    logger = pl_loggers.TensorBoardLogger('results/test', name='test0', default_hp_metric=False)

    trainer = pl.Trainer(logger=logger, gpus=1, max_epochs=3)

    trainer.fit(classification_module, datamodule=classification_data_module)
