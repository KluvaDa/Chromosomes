import os
from abc import ABC
import urllib.request
import tarfile
import numpy as np
import torch
import torch.nn
import torch.nn.functional
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning.metrics.functional.classification
from math import pi

from typing import Sequence, Tuple, Union, Optional
from torch.utils.data import DataLoader

import networks
import datasets
from representations import rep_2d_losses, rep_2d_pytorch
from classification import calculate_binary_iou_batch


def calculate_iou_separate_chromosomes(prediction_chromosomes,
                                       label_chromosomes):
    """
    tensors of shape [channels, x, y]
    :param prediction_chromosomes: tensor of individual chromosomes as channels
    :param label_chromosomes: tensor of individual chromosomes as channels
    :return: iou value over batch
    """
    n_chromosomes_label = prediction_chromosomes.shape[0]
    n_chromosomes_prediction = label_chromosomes.shape[0]

    best_iou = torch.zeros((n_chromosomes_label,))

    for label_chromosomes_i in range(n_chromosomes_label):
        for prediction_chromosome_i in range(n_chromosomes_prediction):
            iou_batch = calculate_binary_iou_batch(
                prediction_chromosomes[None, prediction_chromosome_i: prediction_chromosome_i + 1, ...],
                label_chromosomes[None, label_chromosomes_i: label_chromosomes_i + 1, ...]
            )[0, ...]
            is_best_iou = iou_batch >= best_iou[label_chromosomes_i]
            if is_best_iou:
                best_iou[label_chromosomes_i] = best_iou

    average_iou = torch.mean(best_iou)
    return average_iou


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

    def prepare_data(self):
        if not os.path.isfile(self.filepath_original):
            tar_path = os.path.join('data', 'Cleaned_LowRes_13434_overlapping_pairs.tar.xz')
            if not os.path.isfile(tar_path):
                url = "https://github.com/jeanpat/DeepFISH/blob/master/dataset/" \
                      "Cleaned_LowRes_13434_overlapping_pairs.tar.xz?raw=true"
                filename, headers = urllib.request.urlretrieve(url, tar_path)
            with tarfile.open(tar_path, 'r') as f:
                f.extractall('data')

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


class Clustering:
    def __init__(self):
        raise NotImplementedError()

    def direction_2_separate_chromosomes(self,
                                         batch_3category: torch.Tensor,
                                         batch_dilated_intersection: torch.Tensor,
                                         batch_direction_angle: torch.Tensor):
        raise NotImplementedError()


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

        self.angle_2_repr = rep_2d_pytorch.get_angle_2_repr(representation)
        self.repr_2_angle = rep_2d_pytorch.get_repr_2_angle(representation)
        self.angle_difference_function = rep_2d_losses.define_angular_loss_nored(pi)

        self.clustering = Clustering()

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

    def forward(self, x):
        batch_prediction = self.net(x)
        batch_prediction_3category = torch.argmax(batch_prediction[:, 0:3, ...], dim=1, keepdim=True).detach()
        batch_prediction_dilated_intersection = (batch_prediction[:, 3:4, ...] > 0).astype(float).detach()
        batch_prediction_direction_angle = self.angle_2_repr(batch_prediction[:, 4:, ...]).detach()

        all_separate_chromosomes = []
        for prediction_3_category, prediction_dilated_intersection, prediction_direction_angle in \
            zip(batch_prediction_3category, batch_prediction_dilated_intersection, batch_prediction_direction_angle):

            separate_chromosomes = self.clustering.direction_2_separate_chromosomes(prediction_3_category,
                                                                                    prediction_dilated_intersection,
                                                                                    prediction_direction_angle)
            all_separate_chromosomes.append(separate_chromosomes)

        return batch_prediction, all_separate_chromosomes

    def training_step(self, batch, batch_step):
        batch_in = batch[:, 0:1, ...]

        batch_label = batch_in[:, 1:, ...]
        batch_label_3_category_index = batch_label[:, 0:1, ...].long()
        batch_label_dilated_intersection = batch_label[:, 1:2, ...]
        batch_label_direction_angle = batch_label[:, 2:3, ...]
        batch_label_chromosomes = batch_label[:, 3:5, ...]

        batch_prediction = self.net(batch_in)
        batch_prediction_3_category_channels = batch_prediction[:, 0:3, ...]
        batch_prediction_dilated_intersection = batch_prediction[:, 3:4, ...]
        batch_prediction_direction_representation = batch_prediction[:, 4:, ...]

        loss_3category = torch.nn.functional.cross_entropy(
            batch_prediction_3_category_channels,
            batch_label_3_category_index
        )
        loss_dilated_intersection = torch.nn.functional.binary_cross_entropy_with_logits(
            batch_prediction_dilated_intersection,
            batch_label_dilated_intersection
        )

        # only calculate direction loss on pixels that have unique chromosomes.
        batch_label_direction_representation = self.angle_2_repr(batch_label_direction_angle)
        loss_direction_nored = torch.nn.functional.smooth_l1_loss(
            batch_prediction_direction_representation,
            batch_label_direction_representation,
            reduction='none'
        )
        mask = torch.eq(batch_label_3_category_index, 1).type(self.dtype)
        loss_direction = (loss_direction_nored * mask) / torch.sum(mask)

        loss = loss_3category + loss_dilated_intersection + loss_direction

        # metrics
        batch_prediction_3_category_index = torch.argmax(batch_prediction_3_category_channels, dim=1, keepdim=True)
        batch_prediction_direction_angle = self.repr_2_angle(batch_prediction_direction_representation)

        metrics = self._calculate_metrics_raw(
            batch_prediction_3_category_index,
            batch_prediction_dilated_intersection,
            batch_prediction_direction_angle,
            batch_label_3_category_index,
            batch_label_dilated_intersection,
            batch_label_direction_angle,
            'train'
        )

        all_iou_separate_chromosomes = []
        for i_batch in range(batch.shape[0]):
            prediction_separate_chromosomes = self.clustering.direction_2_separate_chromosomes(
                batch_prediction_3_category_index[i_batch].detach(),
                batch_prediction_dilated_intersection[i_batch].detach(),
                batch_prediction_direction_angle.detach())
            iou_separate_chromosomes = calculate_iou_separate_chromosomes(prediction_separate_chromosomes.detach(),
                                                                          batch_label_chromosomes[i_batch, ...].detach())
            all_iou_separate_chromosomes.append(iou_separate_chromosomes)
        iou_separate_chromosomes = torch.mean(torch.stack(all_iou_separate_chromosomes)).detach()

        metrics['train_iou_separate_chromosomes'] = iou_separate_chromosomes

        metrics['loss'] = loss
        metrics['loss_3category'] = loss_3category
        metrics['loss_dilated_intersection'] = loss_dilated_intersection
        metrics['loss_direction'] = loss_direction

        self.log_dict(metrics, on_step=True)

        return loss

    def validation_step(self, batch, batch_step, dataloader_idx):
        batch_in = batch[:, 0:1, ...].detach()
        batch_label = batch_in[:, 1:, ...].detach()

        batch_prediction = self.net(batch_in)
        batch_prediction_3_category_channels = batch_prediction[:, 0:3, ...]
        batch_prediction_dilated_intersection = batch_prediction[:, 3:4, ...]
        batch_prediction_direction_representation = batch_prediction[:, 4:, ...]

        # raw metrics
        batch_prediction_3_category_index = torch.argmax(batch_prediction_3_category_channels, dim=1, keepdim=True)
        batch_prediction_direction_angle = self.repr_2_angle(batch_prediction_direction_representation)

        if dataloader_idx == 0:
            dataset_name = 'val_synthetic'
            batch_label_3_category_index = batch_label[:, 0:1, ...].long()
            batch_label_dilated_intersection = batch_label[:, 1:2, ...]
            batch_label_direction_angle = batch_label[:, 2:3, ...]
            batch_label_chromosomes = batch_label[:, 3:5, ...]

            metrics = self._calculate_metrics_raw(
                batch_prediction_3_category_index,
                batch_prediction_dilated_intersection,
                batch_prediction_direction_angle,
                batch_label_3_category_index,
                batch_label_dilated_intersection,
                batch_label_direction_angle,
                dataset_name
            )
        elif dataloader_idx == 1:
            dataset_name = 'val_real'
            metrics = dict()
            batch_label_chromosomes = batch_label
        else:
            dataset_name = 'val_original'
            metrics = dict()
            batch_label_ch0 = torch.logical_or(torch.eq(batch_label, 1), torch.eq(batch_label, 3)).type(self.dtype)
            batch_label_ch1 = torch.logical_or(torch.eq(batch_label, 2), torch.eq(batch_label, 3)).type(self.dtype)
            batch_label_chromosomes = torch.cat([batch_label_ch0, batch_label_ch1], dim=1)

        all_iou_separate_chromosomes = []
        for i_batch in range(batch.shape[0]):
            prediction_separate_chromosomes = self.clustering.direction_2_separate_chromosomes(
                batch_prediction_3_category_index[i_batch],
                batch_prediction_dilated_intersection[i_batch],
                batch_prediction_direction_angle)
            iou_separate_chromosomes = calculate_iou_separate_chromosomes(prediction_separate_chromosomes,
                                                                          batch_label_chromosomes[
                                                                              i_batch, ...])
            all_iou_separate_chromosomes.append(iou_separate_chromosomes)
        iou_separate_chromosomes = torch.mean(torch.stack(all_iou_separate_chromosomes))

        metrics[f"{dataset_name}_iou_separate_chromosomes"] = iou_separate_chromosomes

        self.log_dict(metrics, on_epoch=True)

    def test_step(self, batch, batch_step, dataloader_idx):
        """ same as val only different dataset_name"""
        batch_in = batch[:, 0:1, ...].detach()
        batch_label = batch_in[:, 1:, ...].detach()

        batch_prediction = self.net(batch_in)
        batch_prediction_3_category_channels = batch_prediction[:, 0:3, ...]
        batch_prediction_dilated_intersection = batch_prediction[:, 3:4, ...]
        batch_prediction_direction_representation = batch_prediction[:, 4:, ...]

        # raw metrics
        batch_prediction_3_category_index = torch.argmax(batch_prediction_3_category_channels, dim=1, keepdim=True)
        batch_prediction_direction_angle = self.repr_2_angle(batch_prediction_direction_representation)

        if dataloader_idx == 0:
            dataset_name = 'test_synthetic'
            batch_label_3_category_index = batch_label[:, 0:1, ...].long()
            batch_label_dilated_intersection = batch_label[:, 1:2, ...]
            batch_label_direction_angle = batch_label[:, 2:3, ...]
            batch_label_chromosomes = batch_label[:, 3:5, ...]

            metrics = self._calculate_metrics_raw(
                batch_prediction_3_category_index,
                batch_prediction_dilated_intersection,
                batch_prediction_direction_angle,
                batch_label_3_category_index,
                batch_label_dilated_intersection,
                batch_label_direction_angle,
                dataset_name
            )
        elif dataloader_idx == 1:
            dataset_name = 'test_real'
            metrics = dict()
            batch_label_chromosomes = batch_label
        else:
            dataset_name = 'test_original'
            metrics = dict()
            batch_label_ch0 = torch.logical_or(torch.eq(batch_label, 1), torch.eq(batch_label, 3)).type(self.dtype)
            batch_label_ch1 = torch.logical_or(torch.eq(batch_label, 2), torch.eq(batch_label, 3)).type(self.dtype)
            batch_label_chromosomes = torch.cat([batch_label_ch0, batch_label_ch1], dim=1)

        all_iou_separate_chromosomes = []
        for i_batch in range(batch.shape[0]):
            prediction_separate_chromosomes = self.clustering.direction_2_separate_chromosomes(
                batch_prediction_3_category_index[i_batch],
                batch_prediction_dilated_intersection[i_batch],
                batch_prediction_direction_angle)
            iou_separate_chromosomes = calculate_iou_separate_chromosomes(prediction_separate_chromosomes,
                                                                          batch_label_chromosomes[
                                                                              i_batch, ...])
            all_iou_separate_chromosomes.append(iou_separate_chromosomes)
        iou_separate_chromosomes = torch.mean(torch.stack(all_iou_separate_chromosomes))

        metrics[f"{dataset_name}_iou_separate_chromosomes"] = iou_separate_chromosomes

        self.log_dict(metrics, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def _calculate_metrics_raw(self,
                               batch_prediction_3_category_index,
                               batch_prediction_dilated_intersection,
                               batch_prediction_direction_angle,
                               batch_label_3_category_index,
                               batch_label_dilated_intersection,
                               batch_label_direction_angle,
                               dataset_name):
        """
        Only works on synthetic dataset. Tensors of shape [batch, channels, x, y]
        """
        batch_prediction_3_category_index = batch_prediction_3_category_index.detach()
        batch_prediction_dilated_intersection = batch_prediction_dilated_intersection.detach()
        batch_prediction_direction_angle = batch_prediction_direction_angle.detach()
        batch_label_3_category_index = batch_label_3_category_index.detach()
        batch_label_dilated_intersection = batch_label_dilated_intersection.detach()
        batch_label_direction_angle = batch_label_direction_angle.detach()

        iou_channels = [
            torch.mean(
                calculate_binary_iou_batch(
                    torch.eq(batch_prediction_3_category_index, i),
                    torch.eq(batch_label_3_category_index, i)
                )
            )
            for i in range(3)
        ]
        iou_3category = torch.mean(torch.stack(iou_channels))

        iou_dilated_intersection = torch.mean(
            calculate_binary_iou_batch(
                batch_prediction_dilated_intersection > 0,
                batch_label_dilated_intersection > 0
            )
        )

        angle_difference = self.angle_difference_function(batch_prediction_direction_angle,
                                                          batch_label_direction_angle)
        mask = torch.eq(batch_prediction_3_category_index, 1).type(self.dtype)
        max_angle_difference = torch.amax(angle_difference * mask, dim=[1, 2, 3])
        sum_angle_difference = torch.sum(angle_difference * mask, dim=[1, 2, 3])

        metric_max_angle = torch.mean(max_angle_difference)
        metric_average_angle = torch.mean(sum_angle_difference / torch.sum(mask, dim=[1, 2, 3]))

        main_metric = metric_average_angle + \
                      (1 - iou_channels[0])/4 + \
                      (1 - iou_channels[1])/4 + \
                      (1 - iou_channels[2])/4 + \
                      (1 - iou_dilated_intersection)/4

        metrics = {
            f"{dataset_name}_iou_background": iou_channels[0],
            f"{dataset_name}_iou_ch": iou_channels[1],
            f"{dataset_name}_iou_overlap": iou_channels[2],
            f"{dataset_name}_iou_3category": iou_3category,
            f"{dataset_name}_iou_dilated_intersection": iou_dilated_intersection,
            f"{dataset_name}_average_angle": metric_average_angle,
            f"{dataset_name}_max_angle": metric_max_angle,
            f"{dataset_name}_main_metric": main_metric
        }

        return metrics






