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
from representations import rep_2d_losses


class InstanceSegmentationDataModule(pl.LightningDataModule):
    """
    A DataModule that loads all the datasets with defaults preset for instance segmentation purposes.
    It outputs the relevant training dataloader, and for validation and testing a tuple of dataloaders in the following
    order: (synthetic, real, original)
    """

    def __init__(self,
                 cross_validation_i: int,
                 ):
        """
        The data has the following channels:
        Synthetic:
        [input, 3category, dilated_intersection, direction(angle), separate_chromosomes(2 or more)]
        Original:
        [input, 4category]
        Real:
        [input, separate_chromosomes(2 or more)]

        :param cross_validation_i: An integer in [0, 4) specifying which cross-validation split to select.
        """
        super().__init__()
        self.cross_validation_i = cross_validation_i

        # Default params
        self.batchsize = 64
        self.dtype = np.float32
        self.num_workers = 4

        # new synthetic dataset params
        self.imsize_synthetic = (128, 128)  # half resolution will half this
        self.train_batches_per_epoch = 128
        self.val_batches_per_epoch = 8
        self.test_batches_per_epoch = 8

        self.filepath_real = os.path.join('data')
        self.filepath_new_synthetic = os.path.join('data', 'separate.pickle')
        self.filepath_original = os.path.join('data', 'Cleaned_LowRes_13434_overlapping_pairs.h5')

        # placeholders
        self.dataset_original_val = None
        self.dataset_original_test = None

        self.dataset_synthetic_train = None
        self.dataset_synthetic_val = None
        self.dataset_synthetic_test = None

        self.dataset_real_val = None
        self.dataset_real_test = None

    def setup(self, stage=None):
        # original dataset
        val_subsets_cv = {0: [(0.6, 0.8)],
                          1: [(0.4, 0.6)],
                          2: [(0.2, 0.4)],
                          3: [(0.0, 0.2)]}
        val_subset = val_subsets_cv[self.cross_validation_i]
        test_subset = [(0.8, 1.0)]
        self.dataset_original_val = datasets.OriginalChromosomeDataset(self.filepath_original,
                                                                       val_subset,
                                                                       True,
                                                                       True,
                                                                       self.batchsize,
                                                                       fix_random_seed=True,
                                                                       dtype=self.dtype)
        self.dataset_original_test = datasets.OriginalChromosomeDataset(self.filepath_original,
                                                                        test_subset,
                                                                        True,
                                                                        True,
                                                                        self.batchsize,
                                                                        fix_random_seed=True,
                                                                        dtype=self.dtype)
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
        test_slides = (12, 13, 14)

        output_channels_list = ['dapi_cy3', '3_channel', 'intersection_dilated', 'direction', 'ch_0', 'ch_1']

        self.dataset_synthetic_train = datasets.SyntheticChromosomeDataset(self.filepath_new_synthetic,
                                                                           self.imsize_synthetic,
                                                                           train_slides,
                                                                           True,
                                                                           self.batchsize,
                                                                           self.train_batches_per_epoch,
                                                                           output_channels_list,
                                                                           'random',
                                                                           fix_random_seed=False,
                                                                           dtype=self.dtype)
        self.dataset_synthetic_val = datasets.SyntheticChromosomeDataset(self.filepath_new_synthetic,
                                                                         self.imsize_synthetic,
                                                                         val_slides,
                                                                         True,
                                                                         self.batchsize,
                                                                         self.val_batches_per_epoch,
                                                                         output_channels_list,
                                                                         'random',
                                                                         fix_random_seed=True,
                                                                         dtype=self.dtype)
        self.dataset_synthetic_test = datasets.SyntheticChromosomeDataset(self.filepath_new_synthetic,
                                                                          self.imsize_synthetic,
                                                                          test_slides,
                                                                          True,
                                                                          self.batchsize,
                                                                          self.test_batches_per_epoch,
                                                                          output_channels_list,
                                                                          'random',
                                                                          fix_random_seed=True,
                                                                          dtype=self.dtype)
        # real dataset
        self.dataset_real_val = datasets.RealOverlappingChromosomes(self.filepath_real,
                                                                    False,
                                                                    (0, 0.5),
                                                                    separate_channels=False,
                                                                    half_resolution=True,
                                                                    output_categories=None,
                                                                    dtype=self.dtype)
        self.dataset_real_test = datasets.RealOverlappingChromosomes(self.filepath_real,
                                                                     False,
                                                                     (0.5, 1),
                                                                     separate_channels=False,
                                                                     half_resolution=True,
                                                                     output_categories=None,
                                                                     dtype=self.dtype)

    def train_dataloader(self):
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
        dataloader_original = DataLoader(self.dataset_original_val,
                                         batch_size=None,
                                         num_workers=self.num_workers,
                                         pin_memory=self.num_workers > 0)
        return dataloader_synthetic, dataloader_real, dataloader_original

    def test_dataloader(self):
        dataloader_synthetic = DataLoader(self.dataset_synthetic_test,
                                          batch_size=None,
                                          num_workers=self.num_workers,
                                          pin_memory=self.num_workers > 0)
        dataloader_real = DataLoader(self.dataset_real_test,
                                     batch_size=None,
                                     num_workers=self.num_workers,
                                     pin_memory=self.num_workers > 0)
        dataloader_original = DataLoader(self.dataset_original_test,
                                         batch_size=None,
                                         num_workers=self.num_workers,
                                         pin_memory=self.num_workers > 0)
        return dataloader_synthetic, dataloader_real, dataloader_original


class InstanceSegmentationModule(pl.LightningModule):
    def __init__(self, representation: str, smaller_network: bool):
        """
        Module with hard coded parameters for everything.
        :param representation: Which direction representation to use
        :param smaller_network: Whether to use the smaller network (Hu et al) or larger (Saleh et al)
        """
        super().__init__()
        self.save_hyperparameters()

        self.representation = representation

        if representation == 'angle':
            n_output_channels = 5
        elif representation == 'vector':
            n_output_channels = 6
        elif representation == 'da_vector':
            n_output_channels = 6
        elif representation.__contains__('piecewise'):
            n_output_channels = 8
        else:
            raise ValueError(f"representations == '{representation}' is invalid")

        # hardcoded parameters
        if smaller_network:
            backbone_net = networks.FullyConv(n_channels_in=128,
                                              ns_channels_layers=[256, 128],
                                              activation=torch.nn.functional.relu,
                                              kernel_size=3,
                                              groups=1,
                                              norm_layer=torch.nn.BatchNorm2d,
                                              raw_output=False)
            self.net = networks.Unet(n_channels_in=1,
                                     n_channels_out=n_output_channels,
                                     n_channels_start=64,
                                     depth_encoder=2,
                                     depth_decoder=2,
                                     n_resolutions=3,
                                     backbone_net=backbone_net,
                                     input_net=None,
                                     output_net=None,
                                     mode_add=False)
        else:
            backbone_net = networks.FullyConv(n_channels_in=256,
                                              ns_channels_layers=[512, 256],
                                              activation=torch.nn.functional.relu,
                                              kernel_size=3,
                                              groups=1,
                                              norm_layer=torch.nn.BatchNorm2d,
                                              raw_output=False)
            self.net = networks.Unet(n_channels_in=1,
                                     n_channels_out=n_output_channels,
                                     n_channels_start=64,
                                     depth_encoder=2,
                                     depth_decoder=2,
                                     n_resolutions=4,
                                     backbone_net=backbone_net,
                                     input_net=None,
                                     output_net=None,
                                     mode_add=False)

        self.representation_loss = representations.


    def forward(self, x):
        # TODO
        raise NotImplementedError()

    def training_step(self, batch, batch_step):
        batch_in = batch[:, 0:1, ...]
        batch_label = batch[:, 1:, ...].long()
        batch_prediction = self.net(batch_in)

        loss = torch.nn.functional.cross_entropy(batch_prediction, batch_label)
        metrics = self._calculate_metrics(batch_prediction, batch_label, 'training')
        metrics['loss'] = loss
        self.log_dict(metrics, on_step=True)
        return loss

    def validation_step(self, batch, batch_step, dataloader_idx):
        batch_prediction, batch_label = self._step_core(batch)

        if dataloader_idx == 0:
            dataset_name = 'val_synthetic'
        elif dataloader_idx == 1:
            dataset_name = 'val_real'
        else:
            dataset_name = 'val_original'
        metrics = self._calculate_metrics(batch_prediction, batch_label, dataset_name)

        self.log_dict(metrics, on_epoch=True)

    def test_step(self, batch, batch_step, dataloader_idx):
        batch_prediction, batch_label = self._step_core(batch)

        if dataloader_idx == 0:
            dataset_name = 'test_synthetic'
        elif dataloader_idx == 1:
            dataset_name = 'test_real'
        else:
            dataset_name = 'test_original'
        metrics = self._calculate_metrics(batch_prediction, batch_label, dataset_name)

        self.log_dict(metrics, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer


    def _calculate_metrics(self, batch_prediction, batch_label, dataset_name):
        batch_prediction = batch_prediction.detach()
        batch_label = batch_label.detach()

        batch_prediction_class = torch.argmax(batch_prediction, dim=1)

        if self.segment_4_categories:
            iou_channel_pairs = [(0, 0), (1, 1), (2, 2), (3, 3), (1, 2), (2, 1)]
        else:
            iou_channel_pairs = [(0, 0), (1, 1), (2, 2)]

        iou_channels = [calculate_binary_iou_batch(batch_prediction_class == i_pred, batch_label == i_label)
                        for i_pred, i_label in iou_channel_pairs]

        iou = torch.mean(torch.stack(iou_channels[0:self.n_categories], dim=0), dim=0)

        metrics = {dataset_name+'_iou': torch.mean(iou),
                   dataset_name+'_iou_background': torch.mean(iou_channels[0])}
        if self.segment_4_categories:
            metrics[dataset_name+'_iou_ch0'] = torch.mean(iou_channels[1])
            metrics[dataset_name+'_iou_ch1'] = torch.mean(iou_channels[2])
            metrics[dataset_name+'_iou_overlap'] = torch.mean(iou_channels[3])
        else:
            metrics[dataset_name+'_iou_ch'] = torch.mean(iou_channels[1])
            metrics[dataset_name+'_iou_overlap'] = torch.mean(iou_channels[2])

        # calculate switched ch0/ch1 metrics
        if self.segment_4_categories:
            iou_switched = torch.mean(torch.stack([iou_channels[0],
                                                   iou_channels[4],
                                                   iou_channels[5],
                                                   iou_channels[3]], dim=0), dim=0)

            iou_best_match = torch.maximum(iou_switched, iou)
            metrics[dataset_name+'_iou_best_match'] = torch.mean(iou_best_match)

            switched_better = iou_switched > iou
            standard_better = iou_switched < iou

            iou_ch0_best_match = switched_better * iou_channels[4] + standard_better * iou_channels[1]
            iou_ch1_best_match = switched_better * iou_channels[5] + standard_better * iou_channels[2]
            metrics[dataset_name+'_iou_ch0_best_match'] = torch.mean(iou_ch0_best_match)
            metrics[dataset_name+'_iou_ch1_best_match'] = torch.mean(iou_ch1_best_match)

        return metrics


