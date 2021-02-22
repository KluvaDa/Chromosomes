import os
import urllib.request
import tarfile
import numpy as np
import torch
import torch.nn
import torch.nn.functional
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from math import pi


from typing import Sequence, Tuple, Union, Optional, List
from torch.utils.data import DataLoader

import networks
import datasets
from semantic_segmentation import calculate_binary_iou_batch
from clustering import Clustering


def angle_2_da_vector(angles: torch.Tensor) -> torch.Tensor:
    """
    Angles in radians to double-angle vector space; 0 radians -> (1, 0), pi/4 radians -> (0, 1)
    Args:
        angles: torch.Tenor of shape (batch, 1, x, y)
    Returns: torch tensor of shape (batch, 2, x, y)
    """
    double_angle = angles*2
    da_vectors_x = torch.cos(double_angle)
    da_vectors_y = torch.sin(double_angle)
    da_vectors = torch.cat([da_vectors_x, da_vectors_y], dim=1)
    return da_vectors


def da_vector_2_angle(vectors: torch.Tensor) -> torch.Tensor:
    """
    Double-angle vector space to angles in radians in range [0, pi); (1, 0) -> 0 radians, (0, 1) -> pi/4 radians
    Args:
        vectors: torch.Tensor of shape (batch, 2, x, y)
    Returns: torch.Tensor of shape (batch, 1, x, y)
    """
    double_angle = torch.atan2(vectors[:, 1:2, ...], vectors[:, 0:1, ...])
    double_angle = torch.remainder(double_angle, 2*pi)
    angle = double_angle / 2
    return angle


def angular_distance(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Calculates the angular distance between angles modulo pi (finding the smaller angle), maintaining tensor dimensions
    """
    prediction = torch.remainder(prediction, pi)
    label = torch.remainder(label, pi)
    return - torch.abs(torch.abs(prediction-label) - (pi/2)) + (pi/2)


def calculate_iou_separate_chromosomes(prediction_chromosomes,
                                       label_chromosomes):
    """
    Calculates the iou between the best matching chromosome pairs
    tensors of shape [channels, x, y]
    :param prediction_chromosomes: tensor of individual chromosomes as channels
    :param label_chromosomes: tensor of individual chromosomes as channels
    :return: iou value over batch
    """
    n_chromosomes_label = label_chromosomes.shape[0]
    n_chromosomes_prediction = prediction_chromosomes.shape[0]

    best_iou = torch.zeros((n_chromosomes_label,))

    for label_chromosomes_i in range(n_chromosomes_label):
        for prediction_chromosome_i in range(n_chromosomes_prediction):
            iou_batch = calculate_binary_iou_batch(
                prediction_chromosomes[None, prediction_chromosome_i: prediction_chromosome_i + 1, ...],
                label_chromosomes[None, label_chromosomes_i: label_chromosomes_i + 1, ...]
            )[0, ...]
            is_best_iou = iou_batch >= best_iou[label_chromosomes_i]
            if is_best_iou:
                best_iou[label_chromosomes_i] = iou_batch

    average_iou = torch.mean(best_iou)
    return average_iou


class InstanceSegmentationDataModule(pl.LightningDataModule):
    """
    A DataModule that loads all the datasets with defaults preset for instance segmentation purposes.
    Training set is only synthetic
    Validation is only synthetic
    Testing is (val_synthetic, test_synthetic, val_real, test_real, val_original, test_original)
    """

    def __init__(self, cross_validation_i: int,):
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
        return dataloader_synthetic

    def test_dataloader(self):
        dataloader_synthetic_val = DataLoader(self.dataset_synthetic_val,
                                              batch_size=None,
                                              num_workers=self.num_workers,
                                              pin_memory=self.num_workers > 0)
        dataloader_synthetic_test = DataLoader(self.dataset_synthetic_test,
                                               batch_size=None,
                                               num_workers=self.num_workers,
                                               pin_memory=self.num_workers > 0)
        dataloader_real_val = DataLoader(self.dataset_real_val,
                                         batch_size=None,
                                         num_workers=self.num_workers,
                                         pin_memory=self.num_workers > 0)
        dataloader_real_test = DataLoader(self.dataset_real_test,
                                          batch_size=None,
                                          num_workers=self.num_workers,
                                          pin_memory=self.num_workers > 0)
        dataloader_original_val = DataLoader(self.dataset_original_val,
                                             batch_size=None,
                                             num_workers=self.num_workers,
                                             pin_memory=self.num_workers > 0)
        dataloader_original_test = DataLoader(self.dataset_original_test,
                                              batch_size=None,
                                              num_workers=self.num_workers,
                                              pin_memory=self.num_workers > 0)

        return dataloader_synthetic_val, dataloader_synthetic_test,\
               dataloader_real_val, dataloader_real_test,\
               dataloader_original_val, dataloader_original_test


class InstanceSegmentationModule(pl.LightningModule):
    def __init__(self, smaller_network: bool):
        """
        Module with hard coded parameters for everything.
        :param smaller_network: Whether to use the smaller network (Hu et al) or larger (Saleh et al)
        """
        super().__init__()
        self.save_hyperparameters()

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
                                     n_channels_out=6,
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
                                     n_channels_out=6,
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
        batch_prediction_dilated_intersection = (batch_prediction[:, 3:4, ...] > 0).type(self.dtype).detach()
        batch_prediction_direction_angle = da_vector_2_angle(batch_prediction[:, 4:, ...]).detach()

        all_separate_chromosomes = []
        for prediction_3_category, prediction_dilated_intersection, prediction_direction_angle in \
            zip(batch_prediction_3category, batch_prediction_dilated_intersection, batch_prediction_direction_angle):

            separate_chromosomes = self.clustering.direction_2_separate_chromosomes(
                prediction_3_category.numpy(),
                prediction_dilated_intersection.numpy(),
                prediction_direction_angle.numpy()
            )
            all_separate_chromosomes.append(separate_chromosomes)

        return batch_prediction, all_separate_chromosomes

    def training_step(self, batch, batch_step):
        batch_in = batch[:, 0:1, ...]
        batch_label = batch[:, 1:, ...]

        batch_label_3_category_index = batch_label[:, 0:1, ...].long()
        batch_label_dilated_intersection = batch_label[:, 1:2, ...]
        batch_label_direction_angle = batch_label[:, 2:3, ...]
        # batch_label_chromosomes = batch_label[:, 3:5, ...]

        batch_prediction = self.net(batch_in)
        batch_prediction_3_category_channels = batch_prediction[:, 0:3, ...]
        batch_prediction_dilated_intersection = batch_prediction[:, 3:4, ...]
        batch_prediction_direction_da_vector = batch_prediction[:, 4:, ...]

        loss_3category = torch.nn.functional.cross_entropy(
            batch_prediction_3_category_channels,
            batch_label_3_category_index[:, 0, :, :]
        )
        loss_dilated_intersection = torch.nn.functional.binary_cross_entropy_with_logits(
            batch_prediction_dilated_intersection,
            batch_label_dilated_intersection
        )

        # only calculate direction loss on pixels that have unique chromosomes.
        batch_label_direction_da_vector = angle_2_da_vector(batch_label_direction_angle)
        loss_direction_nored = torch.nn.functional.smooth_l1_loss(
            batch_prediction_direction_da_vector,
            batch_label_direction_da_vector,
            reduction='none'
        )
        mask = torch.eq(batch_label_3_category_index, 1).type(self.dtype)
        loss_direction = torch.sum(loss_direction_nored * mask) / torch.sum(mask)

        loss = loss_3category + loss_dilated_intersection + loss_direction

        # metrics
        batch_prediction_3_category_index = torch.argmax(batch_prediction_3_category_channels, dim=1, keepdim=True)
        batch_prediction_direction_angle = da_vector_2_angle(batch_prediction_direction_da_vector)

        metrics = self._calculate_metrics_raw(
            batch_prediction_3_category_index,
            batch_prediction_dilated_intersection,
            batch_prediction_direction_angle,
            batch_label_3_category_index,
            batch_label_dilated_intersection,
            batch_label_direction_angle,
            'train'
        )

        metrics['loss'] = loss
        metrics['loss_3category'] = loss_3category
        metrics['loss_dilated_intersection'] = loss_dilated_intersection
        metrics['loss_direction'] = loss_direction

        self.log_dict(metrics, on_step=True)

        return loss

    def validation_step(self, batch, batch_step):
        batch_in = batch[:, 0:1, ...].detach()
        batch_label = batch[:, 1:, ...].detach()

        batch_prediction = self.net(batch_in)
        batch_prediction_3_category_channels = batch_prediction[:, 0:3, ...]
        batch_prediction_dilated_intersection = batch_prediction[:, 3:4, ...]
        batch_prediction_direction_representation = batch_prediction[:, 4:, ...]

        # raw metrics
        batch_prediction_3_category_index = torch.argmax(batch_prediction_3_category_channels, dim=1, keepdim=True)
        batch_prediction_direction_angle = da_vector_2_angle(batch_prediction_direction_representation)

        dataset_name = 'val_synthetic'
        batch_label_3_category_index = batch_label[:, 0:1, ...].long()
        batch_label_dilated_intersection = batch_label[:, 1:2, ...]
        batch_label_direction_angle = batch_label[:, 2:3, ...]

        metrics = self._calculate_metrics_raw(
            batch_prediction_3_category_index,
            batch_prediction_dilated_intersection,
            batch_prediction_direction_angle,
            batch_label_3_category_index,
            batch_label_dilated_intersection,
            batch_label_direction_angle,
            dataset_name
        )

        self.log_dict(metrics, on_epoch=True)

    def test_step(self, batch, batch_step, dataloader_idx):
        batch_in = batch[:, 0:1, ...].detach()
        batch_label = batch[:, 1:, ...].detach()

        batch_prediction = self.net(batch_in)
        batch_prediction_3_category_channels = batch_prediction[:, 0:3, ...]
        batch_prediction_dilated_intersection = batch_prediction[:, 3:4, ...]
        batch_prediction_direction_da_vector = batch_prediction[:, 4:, ...]

        # raw metrics
        batch_prediction_3_category_index = torch.argmax(batch_prediction_3_category_channels, dim=1, keepdim=True)
        batch_prediction_direction_angle = da_vector_2_angle(batch_prediction_direction_da_vector)

        if dataloader_idx == 0:
            dataset_name = 'val_synthetic'
            dataset_type = 'synthetic'
        elif dataloader_idx == 1:
            dataset_name = 'test_synthetic'
            dataset_type = 'synthetic'
        elif dataloader_idx == 2:
            dataset_name = 'val_real'
            dataset_type = 'real'
        elif dataloader_idx == 3:
            dataset_name = 'test_real'
            dataset_type = 'real'
        elif dataloader_idx == 4:
            dataset_name = 'val_original'
            dataset_type = 'original'
        elif dataloader_idx == 5:
            dataset_name = 'test_original'
            dataset_type = 'original'
        else:
            raise ValueError('dataloader_idx out of bounds')

        if dataset_type == 'synthetic':
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
        elif dataset_type == 'real':
            metrics = dict()
            batch_label_chromosomes = batch_label
        else:  # dataset_type == 'original'
            metrics = dict()
            batch_label_ch0 = torch.logical_or(torch.eq(batch_label, 1), torch.eq(batch_label, 3)).type(self.dtype)
            batch_label_ch1 = torch.logical_or(torch.eq(batch_label, 2), torch.eq(batch_label, 3)).type(self.dtype)
            batch_label_chromosomes = torch.cat([batch_label_ch0, batch_label_ch1], dim=1)

        all_iou_separate_chromosomes = []
        all_n_separate_chromosomes_difference = []
        for i_batch in range(batch.shape[0]):
            prediction_separate_chromosomes = self.clustering.direction_2_separate_chromosomes(
                batch_prediction_3_category_index[i_batch].detach().cpu().numpy(),
                batch_prediction_dilated_intersection[i_batch].detach().cpu().numpy(),
                batch_prediction_direction_angle[i_batch].detach().cpu().numpy()
            )
            prediction_separate_chromosomes = torch.from_numpy(prediction_separate_chromosomes).type_as(batch)
            n_prediction_separated_chromosomes = prediction_separate_chromosomes.shape[0]
            n_label_separated_chromosomes = batch_label_chromosomes.shape[1]
            n_predicted_chromosomes_difference = abs(n_prediction_separated_chromosomes - n_label_separated_chromosomes)
            all_n_separate_chromosomes_difference.append(torch.Tensor([n_predicted_chromosomes_difference]))

            iou_separate_chromosomes = calculate_iou_separate_chromosomes(
                prediction_separate_chromosomes,
                batch_label_chromosomes[i_batch, ...]
            )
            all_iou_separate_chromosomes.append(iou_separate_chromosomes)
        n_predicted_chromosomes_difference = torch.mean(torch.stack(all_n_separate_chromosomes_difference))
        iou_separate_chromosomes = torch.mean(torch.stack(all_iou_separate_chromosomes))

        metrics[f"{dataset_name}_n_chromosomes_difference"] = n_predicted_chromosomes_difference
        metrics[f"{dataset_name}_iou_separate_chromosomes"] = iou_separate_chromosomes

        self.log_dict(metrics, on_epoch=True)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def _calculate_metrics_raw(self,
                               batch_prediction_class,
                               batch_prediction_dilated_intersection,
                               batch_prediction_direction_angle,
                               batch_label_class,
                               batch_label_dilated_intersection,
                               batch_label_direction_angle,
                               dataset_name):
        """
        Only works on synthetic dataset. Tensors of shape [batch, channels, x, y]
        """
        batch_label_class = batch_label_class.detach()
        batch_label_dilated_intersection = batch_label_dilated_intersection.detach()
        batch_label_direction_angle = batch_label_direction_angle.detach()

        batch_prediction_class = batch_prediction_class.detach()
        batch_prediction_dilated_intersection = batch_prediction_dilated_intersection.detach()
        batch_prediction_direction_angle = batch_prediction_direction_angle.detach()

        # classes
        batch_label_background = torch.eq(batch_label_class, 0)
        batch_label_chromosome = torch.eq(batch_label_class, 1)
        batch_label_overlap = torch.eq(batch_label_class, 2)

        batch_prediction_background = torch.eq(batch_prediction_class, 0)
        batch_prediction_chromosome = torch.eq(batch_prediction_class, 1)
        batch_prediction_overlap = torch.eq(batch_prediction_class, 2)

        batch_iou_background = calculate_binary_iou_batch(batch_prediction_background, batch_label_background)
        batch_iou_chromosome = calculate_binary_iou_batch(batch_prediction_chromosome, batch_label_chromosome)
        batch_iou_overlap = calculate_binary_iou_batch(batch_prediction_overlap, batch_label_overlap)

        iou_background = torch.mean(batch_iou_background)
        iou_chromosome = torch.mean(batch_iou_chromosome)
        iou_overlap = torch.mean(batch_iou_overlap)

        average_iou_classes = torch.mean(torch.stack([iou_background, iou_chromosome, iou_overlap]))

        # dilated overlap
        batch_iou_dilated_overlap = calculate_binary_iou_batch(
            batch_prediction_dilated_intersection > 0, batch_label_dilated_intersection > 0)

        iou_dilated_overlap = torch.mean(batch_iou_dilated_overlap)

        # orientation
        angle_difference = angular_distance(batch_prediction_direction_angle, batch_label_direction_angle)
        mask = batch_label_chromosome.type(self.dtype)
        max_angle_difference = torch.amax(angle_difference * mask, dim=[1, 2, 3])
        sum_angle_difference = torch.sum(angle_difference * mask, dim=[1, 2, 3])

        metric_max_angle = torch.mean(max_angle_difference)
        metric_average_angle = torch.mean(sum_angle_difference / torch.sum(mask, dim=[1, 2, 3]))

        # main metric
        main_metric = metric_average_angle + \
                      (1 - iou_background)/4 + \
                      (1 - iou_chromosome)/4 + \
                      (1 - iou_overlap)/4 + \
                      (1 - iou_dilated_overlap)/4

        metrics = {
            f"{dataset_name}_iou_background": iou_background,
            f"{dataset_name}_iou_chromosome": iou_chromosome,
            f"{dataset_name}_iou_overlap": iou_overlap,
            f"{dataset_name}_average_iou_classes": average_iou_classes,
            f"{dataset_name}_iou_dilated_overlap": iou_dilated_overlap,
            f"{dataset_name}_average_angle": metric_average_angle,
            f"{dataset_name}_max_angle": metric_max_angle,
            f"{dataset_name}_main_metric": main_metric
        }

        return metrics


def train(smaller_network: bool,
          cross_validation_i: int):

    root_path = 'results/instance_segmentation'

    name = os.path.join(f"{'snet' if smaller_network else 'lnet'}", f"cv{cross_validation_i}")

    max_epochs = 128
    early_stopping_patience = 8

    instance_segmentation_module = InstanceSegmentationModule(smaller_network)
    instance_segmentation_data_module = InstanceSegmentationDataModule(cross_validation_i)

    logger = pl_loggers.TensorBoardLogger(root_path, name=name, default_hp_metric=False)

    main_metric = 'val_synthetic_main_metric'
    early_stopping_callback = pl.callbacks.EarlyStopping(main_metric,
                                                         patience=early_stopping_patience,
                                                         mode='min')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor=main_metric, mode='min',
                                                       filename='best_{epoch}_{step}')

    trainer = pl.Trainer(logger=logger, gpus=1, max_epochs=max_epochs,
                         callbacks=[early_stopping_callback, checkpoint_callback])

    trainer.fit(instance_segmentation_module, datamodule=instance_segmentation_data_module)


def train_all():
    for smaller_network in (False, True):
        for cross_validation_i in (0, 1, 2, 3):
            train(smaller_network, cross_validation_i)


if __name__ == '__main__':
    train_all()
