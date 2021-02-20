import os
import urllib.request
import tarfile
import numpy as np
import torch
import torch.nn
import torch.nn.functional
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from typing import Sequence, Tuple, Union, Optional
from torch.utils.data import DataLoader

import networks
import datasets


def calculate_binary_iou_batch(prediction, label):
    """
    Calculates an iou core for binary tensors, maintaining the batch dimension.
    Defaults to a value of 0 if the union is 0
    :param prediction: binary tensor of shape (batch, 1, x, y)
    :param label: binary tensor of shape (batch, 1, x, y)
    :return: tensor of shape (batch)
    """
    assert prediction.shape[1] == 1
    assert label.shape[1] == 1
    intersection = torch.sum(torch.logical_and(prediction, label), dim=[1, 2, 3])
    union = torch.sum(torch.logical_or(prediction, label), dim=[1, 2, 3])
    iou = torch.div(intersection, union)
    iou[torch.logical_or(torch.isinf(iou), torch.isnan(iou))] = 0
    return iou


class ClassificationDataModule(pl.LightningDataModule):
    """
    A DataModule that loads all the datasets with defaults preset for classification purposes.
    It outputs the relevant training dataloader, and for validation and testing a tuple of dataloaders in the following
    order: (synthetic, real, original)
    """
    def __init__(self,
                 dataset_identifier: str,
                 cross_validation_i: int,
                 ):
        """
        :param dataset_identifier: Which dataset to train on. One of the following:
               'original' 'new_random', 'new_position', 'new_orientation', 'new_length'
        :param cross_validation_i: An integer in [0, 4) specifying which cross-validation split to select.
        """
        super().__init__()
        self.dataset_identifier = dataset_identifier
        self.cross_validation_i = cross_validation_i

        if self.dataset_identifier == 'original':
            self.train_on_original = True
            self.category_order = 'random'
        elif self.dataset_identifier == 'new_random':
            self.train_on_original = False
            self.category_order = 'random'
        elif self.dataset_identifier == 'new_position':
            self.train_on_original = False
            self.category_order = 'position'
        elif self.dataset_identifier == 'new_orientation':
            self.train_on_original = False
            self.category_order = 'orientation'
        elif self.dataset_identifier == 'new_length':
            self.train_on_original = False
            self.category_order = 'length'
        else:
            raise ValueError(f'{self.dataset_identifier} is not a valid dataset identifier')

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
        self.dataset_original_train = None
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
        if self.train_on_original:
            self.dataset_original_train = datasets.OriginalChromosomeDataset(self.filepath_original,
                                                                             train_subset,
                                                                             True,
                                                                             False,
                                                                             self.batchsize,
                                                                             dtype=self.dtype)
        self.dataset_original_val = datasets.OriginalChromosomeDataset(self.filepath_original,
                                                                       val_subset,
                                                                       True,
                                                                       False,
                                                                       self.batchsize,
                                                                       fix_random_seed=True,
                                                                       dtype=self.dtype)
        self.dataset_original_test = datasets.OriginalChromosomeDataset(self.filepath_original,
                                                                        test_subset,
                                                                        True,
                                                                        False,
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

        output_channels_list = ['dapi_cy3', '4_channel']

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
                                                                          test_slides,
                                                                          True,
                                                                          self.batchsize,
                                                                          self.test_batches_per_epoch,
                                                                          output_channels_list,
                                                                          self.category_order,
                                                                          fix_random_seed=True,
                                                                          dtype=self.dtype)
        # real dataset
        self.dataset_real_val = datasets.RealOverlappingChromosomes(self.filepath_real,
                                                                    True,
                                                                    (0, 0.5),
                                                                    separate_channels=False,
                                                                    half_resolution=True,
                                                                    output_categories=4,
                                                                    dtype=self.dtype)
        self.dataset_real_test = datasets.RealOverlappingChromosomes(self.filepath_real,
                                                                     True,
                                                                     (0.5, 1),
                                                                     separate_channels=False,
                                                                     half_resolution=True,
                                                                     output_categories=4,
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


class ClassificationModule(pl.LightningModule):
    def __init__(self, smaller_network: bool):
        """
        Module with hard coded parameters for everything.
        :param smaller_network: Whether to use the smaller network (Hu et al) or larger (Saleh et al)
        """
        super().__init__()
        self.save_hyperparameters()

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
                                     n_channels_out=4,
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
                                     n_channels_out=4,
                                     n_channels_start=64,
                                     depth_encoder=2,
                                     depth_decoder=2,
                                     n_resolutions=4,
                                     backbone_net=backbone_net,
                                     input_net=None,
                                     output_net=None,
                                     mode_add=False)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_step):
        batch_prediction, batch_label = self._step_core(batch)
        loss = torch.nn.functional.cross_entropy(batch_prediction, batch_label[:, 0, ...])
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
        return metrics

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
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def _step_core(self, batch):
        batch_in = batch[:, 0:1, ...]
        batch_label = batch[:, 1:2, ...].long()
        batch_prediction = self.net(batch_in)

        return batch_prediction, batch_label

    def _calculate_metrics(self, batch_prediction, batch_label, dataset_name):
        batch_prediction = batch_prediction.detach()
        batch_label = batch_label.detach()

        batch_label_background = torch.eq(batch_label, 0)
        batch_label_ch0 = torch.eq(batch_label, 1)
        batch_label_ch1 = torch.eq(batch_label, 2)
        batch_label_overlap = torch.eq(batch_label, 3)

        batch_prediction_class = torch.argmax(batch_prediction, dim=1, keepdim=True)

        batch_prediction_background = torch.eq(batch_prediction_class, 0)
        batch_prediction_ch0 = torch.eq(batch_prediction_class, 1)
        batch_prediction_ch1 = torch.eq(batch_prediction_class, 2)
        batch_prediction_overlap = torch.eq(batch_prediction_class, 3)

        batch_iou_background = calculate_binary_iou_batch(batch_prediction_background, batch_label_background)
        batch_iou_ch0 = calculate_binary_iou_batch(batch_prediction_ch0, batch_label_ch0)
        batch_iou_ch1 = calculate_binary_iou_batch(batch_prediction_ch1, batch_label_ch1)
        batch_iou_overlap = calculate_binary_iou_batch(batch_prediction_overlap, batch_label_overlap)

        batch_iou_ch0_switched = calculate_binary_iou_batch(batch_prediction_ch1, batch_label_ch0)
        batch_iou_ch1_switched = calculate_binary_iou_batch(batch_prediction_ch0, batch_label_ch1)

        switched_better = torch.lt(
            torch.mean(torch.stack([batch_iou_ch0, batch_iou_ch1], dim=0), dim=0),
            torch.mean(torch.stack([batch_iou_ch0_switched, batch_iou_ch1_switched], dim=0), dim=0)
        )
        standard_better = torch.logical_not(switched_better)

        batch_prediction_ch0_best = batch_prediction_ch0 * standard_better[:, None, None, None] + \
                                    batch_prediction_ch1 * switched_better[:, None, None, None]

        batch_prediction_ch1_best = batch_prediction_ch1 * standard_better[:, None, None, None] + \
                                    batch_prediction_ch0 * switched_better[:, None, None, None]

        batch_iou_ch0_best = calculate_binary_iou_batch(batch_prediction_ch0_best, batch_label_ch0)
        batch_iou_ch1_best = calculate_binary_iou_batch(batch_prediction_ch1_best, batch_label_ch1)

        batch_iou_ch0_or_ch1 = calculate_binary_iou_batch(
            torch.logical_or(batch_prediction_ch0, batch_prediction_ch1),
            torch.logical_or(batch_label_ch1, batch_label_ch1)
        )  # no need to use best, because OR is the same

        batch_iou_separated_ch0 = calculate_binary_iou_batch(
            torch.logical_or(batch_prediction_ch0_best, batch_prediction_overlap),
            torch.logical_or(batch_label_ch0, batch_label_overlap)
        )

        batch_iou_separated_ch1 = calculate_binary_iou_batch(
            torch.logical_or(batch_prediction_ch1_best, batch_prediction_overlap),
            torch.logical_or(batch_label_ch1, batch_label_overlap)
        )

        iou_background = torch.mean(batch_iou_background)
        iou_ch0_best = torch.mean(batch_iou_ch0_best)
        iou_ch1_best = torch.mean(batch_iou_ch1_best)
        iou_overlap = torch.mean(batch_iou_overlap)

        average_iou_classes = torch.mean(torch.stack([iou_background, iou_ch0_best, iou_ch1_best, iou_overlap]))
        average_iou_separated = torch.mean(torch.stack([batch_iou_separated_ch0, batch_iou_separated_ch1]))

        metrics = {dataset_name + '_iou_background': torch.mean(batch_iou_background),
                   dataset_name + '_iou_ch0': torch.mean(batch_iou_ch0_best),
                   dataset_name + '_iou_ch1': torch.mean(batch_iou_ch1_best),
                   dataset_name + '_iou_overlap': torch.mean(batch_iou_overlap),
                   dataset_name + '_average_iou_classes': average_iou_classes,
                   dataset_name + '_average_iou_separated': average_iou_separated}

        return metrics


def train(dataset_identifier: str,
          smaller_network: bool,
          cross_validation_i: int):

    run_name = os.path.join(f"{dataset_identifier}_{'snet' if smaller_network else 'lnet'}", f"cv{cross_validation_i}")

    max_epochs = 128
    early_stopping_patience = 8

    classification_module = ClassificationModule(smaller_network)
    classification_data_module = ClassificationDataModule(dataset_identifier, cross_validation_i)
    logger = pl_loggers.TensorBoardLogger('results/semantic_segmentation', name=run_name, default_hp_metric=False)

    if dataset_identifier == 'original':
        main_metric = 'val_original_iou/dataloader_idx_2'
    else:
        main_metric = 'val_synthetic_iou/dataloader_idx_0'

    early_stopping_callback = pl.callbacks.EarlyStopping(main_metric,
                                                         patience=early_stopping_patience,
                                                         mode='max')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor=main_metric, mode='max',
                                                       filename='best_{epoch}_{step}')

    trainer = pl.Trainer(logger=logger, gpus=1, max_epochs=max_epochs,
                         callbacks=[early_stopping_callback, checkpoint_callback])

    trainer.fit(classification_module, datamodule=classification_data_module)


def train_all():
    for smaller_network in (False, True):
        for dataset_identifier in ('original', 'new_random', 'new_position', 'new_orientation', 'new_length'):
            for cross_validation_i in (0, 1, 2, 3):
                train(dataset_identifier, smaller_network, cross_validation_i)


if __name__ == '__main__':
    train_all()
