import os
import numpy as np
import skimage.morphology
from math import pi
import scipy.ndimage
import pickle
import h5py
import pytorch_lightning as pl

from typing import Sequence, Tuple, Union, Optional
from torch.utils.data import IterableDataset, get_worker_info

from representations import rep_2d_numpy


def load_pickle(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


class RealOverlappingChromosomes(IterableDataset):
    def __init__(self,
                 dataset_directory: str,
                 use_paired_cropped: bool,
                 subset: Optional[Tuple[Union[int, float], Union[int, float]]],
                 separate_channels: bool,
                 half_resolution: bool = False,
                 output_categories: Optional[int] = None,
                 dtype: np.ScalarType = np.float32):
        """
        Returns batches of size 1 with real images of intersecting chromosomes. This includes labels.
        channels are input (1 or 2 channels depending on separate_channels),
        followed by labels as either a 4 category index image, or 2 channels, one for each chromosome
        Args:
            dataset_directory: Location of the real_overlapping.pickle and/or real_overlapping_paired.pickle
            use_paired_cropped: Whether to load paired and cropped chromosomes (True) or full clusters (False)
            subset: Tuple with lower and upper proportion in the range [0, 1] to select a subset of images
            separate_channels: if dapi and cy3 are separate (True), or combined into one channels (False)
            half_resolution: Whether to halve the resolution of the image
            output_categories: 3 or 4 depending on how many categories to output as label. If None, ch0 and ch1 are
                               returned as channels instead. Must be None or 3 if not use_paired_cropped.
            dtype: dtype of the outputted numpy array.
        """
        self.dataset_directory = dataset_directory
        self.use_paired_cropped = use_paired_cropped
        self.subset = subset if subset is not None else [0, 1]
        self.separate_input_channels = separate_channels
        self.half_resolution = half_resolution
        self.output_categories = output_categories
        if output_categories is not None:
            assert output_categories in (3, 4)
        self.dtype = dtype

        if not self.use_paired_cropped:
            assert self.output_categories != 4

        if self.use_paired_cropped:
            self.images = load_pickle(os.path.join(self.dataset_directory, 'real_overlapping_paired.pickle'))
        else:
            self.images = load_pickle(os.path.join(self.dataset_directory, 'real_overlapping.pickle'))

        subset_indices = np.floor(np.array(self.subset) * len(self.images)).astype(np.int)
        self.all_image_indices = list(range(subset_indices[0], subset_indices[1]))

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            self.n_workers = 1
            self.worker_id = 0
        else:
            self.n_workers = worker_info.num_workers
            self.worker_id = worker_info.id

        self.worker_image_indices = self.all_image_indices[self.worker_id::self.n_workers]
        self.i_worker_image = 0
        return self

    def __next__(self):
        if self.i_worker_image >= len(self.worker_image_indices):
            raise StopIteration
        image_index = self.worker_image_indices[self.i_worker_image]
        self.i_worker_image += 1
        image = self.images[image_index]

        if self.output_categories is not None:
            chromosomes = image[2:]
            if self.output_categories == 3:
                categories = np.any(chromosomes, axis=0).astype(self.dtype) + \
                             (np.sum(chromosomes, axis=0) >= 2).astype(self.dtype)
            else:
                categories = chromosomes[0] + 2*chromosomes[1]
            image[2] = categories
            image = image[0:3]

        if self.separate_input_channels:
            pass
        else:
            image[0] = image[0] + image[1]
            image = np.delete(image, 1, 0)

        if self.half_resolution:
            image = image[:, ::2, ::2]
        image = image[None, ...]

        return image.astype(self.dtype)


class OriginalChromosomeDataset(IterableDataset):
    def __init__(self,
                 filepath: str,
                 data_subset_ranges: Sequence[Tuple[Union[float, int], Union[float, int]]],
                 segment_4_categories: bool,
                 shuffle_first: bool,
                 batchsize: int,
                 fix_random_seed: bool = False,
                 dtype: np.ScalarType = np.float32):
        """

        :param filepath: Where the h5 file with the 13434_overlapping_chrom_pairs_LowRes dataset is located
        :param data_subset_ranges: A list of tuples with high and low values between 0 and 1 to specify which parts of
                                   the dataset to use. e.g. [(0, 0.8)]
        :param segment_4_categories: If true, returns labels as: (0: background, 1: ch_0, 2: ch_1, 3: overlap)
                                     else, returns labels as: (0: background, 1: chromosome, 2: overlap)
        :param shuffle_first: Whether to shuffle the images in the dataset before selecting the data_subset_ranges
        :param batchsize: batchsize in dim 0
        :param dtype: dtype of the outputted numpy array.
        """
        super(OriginalChromosomeDataset).__init__()
        self.batchsize = batchsize
        self.dtype = dtype

        if fix_random_seed:
            self.rng = np.random.default_rng(0)
        else:
            self.rng = np.random.default_rng()

        # load dataset
        with h5py.File(filepath, 'r') as file:
            pairs = file['13434_overlapping_chrom_pairs_LowRes'][()]
        data = np.empty((pairs.shape[0], pairs.shape[3], pairs.shape[1], pairs.shape[2]), dtype=np.float32)
        data[:, 0, :, :] = np.clip(pairs[:, :, :, 0]/255.0, 0, 1)
        data[:, 1, :, :] = pairs[:, :, :, 1]

        # randomize order with a fixed seed
        if shuffle_first:
            np.random.default_rng().shuffle(data, axis=0)

        # select a subset of dataset
        n_images = data.shape[0]
        data_subsets = []
        for data_subset_range in data_subset_ranges:
            assert data_subset_range[0] < data_subset_range[1]
            assert 0 <= data_subset_range[0] < 1
            assert 0 < data_subset_range[1] <= 1
            i_start = int(n_images*data_subset_range[0])
            i_end = int(n_images*data_subset_range[1])
            data_subsets.append(data[i_start:i_end, ...])
        self.data = np.concatenate(data_subsets, axis=0)

        if not segment_4_categories:
            self.data -= self.data >= 2

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            self.n_workers = 1
            self.worker_id = 0
        else:
            self.n_workers = worker_info.num_workers
            self.worker_id = worker_info.id
        # select data subset for this dataloader
        self.data_for_worker = self.data[self.worker_id::self.n_workers]
        # shuffle order of data for next epoch
        self.rng.shuffle(self.data_for_worker)
        self.i_batch = 0
        return self

    def __next__(self):
        data_batch = self.data_for_worker[self.batchsize*self.i_batch:self.batchsize*(self.i_batch+1)]
        if len(data_batch) == 0:
            raise StopIteration
        self.i_batch += 1
        return data_batch.astype(self.dtype)


class SyntheticChromosomeDataset(IterableDataset):
    def __init__(self,
                 filepath_slides: str,
                 imsize: Sequence[int],
                 slides_select: Sequence[int],
                 half_resolution: bool,
                 batchsize: int,
                 epoch_batches: int,
                 output_channels_list: Sequence[str],
                 order: str = 'random',
                 fix_random_seed: bool = False,
                 dtype: np.ScalarType = np.float32):
        """
        Args:
            filepath_slides: Path to the pickle file with the data.
            imsize: tuple specifying the dimensions of the image
            slides_select: A list of slides to select chromosomes from randomly. 15 available (idx 0 to 14)
            half_resolution: whether to output full resolution images, or half
            batchsize: number of images in batch
            epoch_batches: number of batches before raising StopIteration
            order: 'random', 'position', 'orientation', 'length'
            fix_random_seed: Whether to output deterministic datasets
            dtype: dtype of the outputted numpy array.
            output_channels_list: A list of strings that specifies what the combine_function outputs. Each element in
                                  the list is a string that will correspond to a channel of the output. The following
                                  are the available channel strings:
                                  'dapi' - dapi image, overlapping chromosomes added
                                  'cy3' - cy3 image, overlapping chromosomes added
                                  'dapi_cy3' - dapi and cy3 images added, overlapping chromosomes added
                                  '3_category' - integer indices for (0: background, 1: unique, 2: intersection)
                                  '4_category' - integer indeces for (0: background, 1: ch_0, 2: ch_1, 3: intersection)
                                  'intersection' - the area of intersection of the two chromosomes
                                  'union' - the area with a single chromosome or the intersection
                                  'unique' - the area with a single chromosome
                                  'background' - the area without any chromosomes
                                  'ch_0' - the area with chromosome 0 (arbitrarily selected 0 and 1)
                                  'ch_1' - the area with chromosome 1 (arbitrarily selected 0 and 1)
                                  'ch_0_unique' - the area with chromosome 0 and not chromosome 1
                                  'ch_1_unique' - the area with chromosome 1 and not chromosome 0
                                  'skeleton' - the centreline of chromosome 0 and chromosome 1
                                  'skeleton_ch_0' - the centreline of chromosome 0
                                  'skeleton_ch_1' - the centreline of chromosome 0
                                  'direction' - the direction of the chromosome as an angle in range [0, pi],
                                                valid in unique areas, 0 where undefined
                                  'direction_ch_0' - the direction of chromosome 0 as an angle in range [0, pi],
                                                     0 where undefined
                                  'direction_ch_1' - the direction of chromosome 1 as an angle in range [0, pi]
                                                     0 where undefined
                                  'intersection_dilated' - intersection for chromosome areas that are dilated by 1 pixel
                                  'union_dilated' - union for chromosome areas that are dilated by 1 pixel
                                  'unique_dilated' - unique for chromosome areas that are dilated by 1 pixel
                                  'background_dilated' - background for chromosome areas that are dilated by 1 pixel
                                  'ch_0_dilated' - ch_0 for chromosome areas that are dilated by 1 pixel
                                  'ch_1_dilated' - ch_1 for chromosome areas that are dilated by 1 pixel
                                  'ch_0_dilated_unique' - ch_0 and not ch_1 for chromosomes that are dilated by 1 pixel
                                  'ch_1_dilated_unique' - ch_1 and not ch_0 for chromosomes that are dilated by 1 pixel
        """
        super(SyntheticChromosomeDataset, self).__init__()
        self.filepath_slides = filepath_slides
        self.imsize = imsize
        self.slides_select = slides_select
        self.half_resolution = half_resolution
        self.batchsize = batchsize
        self.epoch_batches = epoch_batches
        self.output_channels_list = output_channels_list
        self.order = order
        self.fix_random_seed = fix_random_seed
        self.dtype = dtype

        self.slides = load_pickle(self.filepath_slides)

        self.rng = np.random.default_rng()

    @staticmethod
    def _rotate_tangent(tangents, rotation):
        """ Rotate in degrees, tangents has channels 0,1 and 2,3 for two sets of vectors"""
        rotation_rad = rotation / 180 * pi
        rotation_sin = np.sin(rotation_rad)
        rotation_cos = np.cos(rotation_rad)
        tangents_rotated = np.empty_like(tangents)

        tangents_rotated[..., 0] = rotation_cos * tangents_rotated[..., 0] - rotation_sin * tangents_rotated[..., 1]
        tangents_rotated[..., 1] = rotation_sin * tangents_rotated[..., 0] + rotation_cos * tangents_rotated[..., 1]
        tangents_rotated[..., 2] = rotation_cos * tangents_rotated[..., 2] - rotation_sin * tangents_rotated[..., 3]
        tangents_rotated[..., 3] = rotation_sin * tangents_rotated[..., 2] + rotation_cos * tangents_rotated[..., 3]

        return tangents_rotated

    def _translate_image(self, image, translation):
        """ Translates the image from the centre of zeros with shape imsize """
        target_shape = np.array(self.imsize)
        image_shape = np.array(image.shape[-2:])
        translation = np.array(translation)

        translation_align_centre = (target_shape - image_shape) // 2

        target_lower = translation + translation_align_centre
        target_upper = translation + translation_align_centre + image_shape

        lower_out_of_bounds_correction = - np.minimum(0, target_lower)
        upper_out_of_bounds_correction = - np.maximum(0, target_upper - target_shape)

        translated_image = np.zeros(image.shape[:-2] + tuple(target_shape), dtype=self.dtype)

        slice_target_lower = target_lower + lower_out_of_bounds_correction
        slice_target_upper = target_upper + upper_out_of_bounds_correction

        slice_image_lower = lower_out_of_bounds_correction
        slice_image_upper = upper_out_of_bounds_correction + image_shape

        assert np.all(slice_target_upper - slice_target_lower == slice_image_upper - slice_image_lower)

        if np.any(slice_target_upper - slice_target_lower <= 0):
            # Out of range. Nothing to write
            return translated_image
        else:
            translated_image[...,
                             slice_target_lower[0]: slice_target_upper[0],
                             slice_target_lower[1]: slice_target_upper[1]] = \
                image[...,
                      slice_image_lower[0]: slice_image_upper[0],
                      slice_image_lower[1]: slice_image_upper[1]]

        return translated_image

    def combine_chromosomes(self, ch_0, ch_1):
        """
        Creates an output image by combining ch_0 and ch_1 to create the required outputs, which are specified in
        self.output_channel_list.
        Args:
            ch_0: Chromosome image of shape (5, ?, ?) with the following channels in dim 0:
                  (dapi, cy3, skeleton, segmentation, direction)
            ch_1: Second chromosome image of shape (5, ?, ?). Image dimensions do not have to be the same as in ch_0.

        Returns: Output image with channels in dim 0 corresponding to self.output_channel_list
        """
        # only calculate dilated segmentation if required
        ch_0_seg_dilated = None
        ch_1_seg_dilated = None
        output_channel_set = set(self.output_channels_list)
        if bool({'intersection_dilated', 'union_dilated', 'unique_dilated',
                 'background_dilated', 'ch_0_dilated', 'ch_1_dilated'} & output_channel_set):
            ch_0_seg_dilated = skimage.morphology.binary_dilation(ch_0[3], selem=np.ones((5, 5)))
            ch_1_seg_dilated = skimage.morphology.binary_dilation(ch_1[3], selem=np.ones((5, 5)))

        # calculate combined images
        output = []
        for output_channel in self.output_channels_list:
            if output_channel == 'dapi':
                output.append(ch_0[0] + ch_1[0])
            elif output_channel == 'cy3':
                output.append(ch_0[1] + ch_1[1])
            elif output_channel == 'dapi_cy3':
                output.append(ch_0[0] + ch_1[0] + ch_0[1] + ch_1[1])
            elif output_channel == '3_channel':
                indices = np.zeros_like(ch_0[3])
                indices += np.logical_or(ch_0[3], ch_1[3]).astype(self.dtype) + \
                           np.logical_and(ch_0[3], ch_1[3]).astype(self.dtype)
                output.append(indices)
            elif output_channel == '4_channel':
                indices = np.zeros_like(ch_0[3])
                indices += ch_0[3] + 2*ch_1[3]
                output.append(indices)
            elif output_channel == 'intersection':
                output.append(np.logical_and(ch_0[3], ch_1[3]).astype(self.dtype))
            elif output_channel == 'union':
                output.append(np.logical_or(ch_0[3], ch_1[3]).astype(self.dtype))
            elif output_channel == 'unique':
                output.append(np.logical_xor(ch_0[3], ch_1[3]).astype(self.dtype))
            elif output_channel == 'background':
                output.append(
                    np.logical_not(np.logical_or(ch_0[3], ch_1[3])).astype(self.dtype)
                )
            elif output_channel == 'ch_0':
                output.append(ch_0[3].astype(self.dtype))
            elif output_channel == 'ch_1':
                output.append(ch_1[3].astype(self.dtype))
            elif output_channel == 'ch_0_unique':
                output.append(
                    np.logical_and(ch_0[3], np.logical_not(ch_1[3])).astype(self.dtype)
                )
            elif output_channel == 'ch_1_unique':
                output.append(
                    np.logical_and(ch_1[3], np.logical_not(ch_0[3])).astype(self.dtype)
                )
            elif output_channel == 'skeleton':
                output.append(np.logical_and(ch_0[2], ch_1[2]).astype(self.dtype))
            elif output_channel == 'skeleton_ch_0':
                output.append(ch_0[2].astype(self.dtype))
            elif output_channel == 'skeleton_ch_1':
                output.append(ch_1[2].astype(self.dtype))
            elif output_channel == 'direction':
                direction = ch_0[4] * ch_0[3] + ch_1[4] * ch_1[3]
                direction *= np.logical_xor(ch_0[3], ch_1[3])
                direction = np.remainder(direction, pi)
                output.append(direction)
            elif output_channel == 'direction_ch_0':
                direction = ch_0[4] * ch_0[3]
                direction = np.remainder(direction, pi)
                output.append(direction)
            elif output_channel == 'direction_ch_1':
                direction = ch_1[4] * ch_1[3]
                direction = np.remainder(direction, pi)
                output.append(direction)
            elif output_channel == 'intersection_dilated':
                output.append(
                    np.logical_and(ch_0_seg_dilated, ch_1_seg_dilated).astype(self.dtype)
                )
            elif output_channel == 'union_dilated':
                output.append(np.logical_or(ch_0_seg_dilated, ch_1_seg_dilated).astype(self.dtype))
            elif output_channel == 'unique_dilated':
                output.append(np.logical_xor(ch_0_seg_dilated, ch_1_seg_dilated).astype(self.dtype))
            elif output_channel == 'background_dilated':
                output.append(np.logical_not(np.logical_or(ch_0_seg_dilated, ch_1_seg_dilated)).astype(self.dtype))
            elif output_channel == 'ch_0_dilated':
                output.append(ch_0_seg_dilated.astype(self.dtype))
            elif output_channel == 'ch_1_dilated':
                output.append(ch_1_seg_dilated.astype(self.dtype))
            elif output_channel == 'ch_0_dilated_unique':
                output.append(
                    np.logical_and(ch_0_seg_dilated, np.logical_not(ch_1_seg_dilated))
                )
            elif output_channel == 'ch_1_dilated_unique':
                output.append(
                    np.logical_and(ch_1_seg_dilated, np.logical_not(ch_0_seg_dilated))
                )
            else:
                raise ValueError(f"channel type {output_channel} not recognised")
        output = np.stack(output)
        return output

    def get_random_chromosome_pair(self):
        """ Select two random chromosomes and ensure they are not the same. """
        while True:
            slide_0_i = self.rng.choice(self.slides_select)
            slide_1_i = self.rng.choice(self.slides_select)
            ch_0_i = self.rng.integers(0, len(self.slides[slide_0_i]))
            ch_1_i = self.rng.integers(0, len(self.slides[slide_1_i]))
            # ch_0 and ch_1 must be different chromosomes
            if slide_0_i == slide_1_i and ch_0_i == ch_1_i:
                continue
            ch_0 = self.slides[slide_0_i][ch_0_i]
            ch_1 = self.slides[slide_1_i][ch_1_i]
            return ch_0, ch_1

    @staticmethod
    def rotate_chromosome(ch, rotation):
        """ Rotates the chromosomes by a rotation in degrees. Modifies the chromosome array inplace. """
        # ROTATE dapi, cy3, skeleton, segmentation, not angles, with spline interpolation
        ch_rotated = scipy.ndimage.rotate(ch[0:4], rotation, axes=(1, 2))
        # threshold segmentation after interpolation
        ch_rotated[3] = ch_rotated[3] > 0.5
        # skeletonize skeleton after interpolation
        ch_rotated[2] = skimage.morphology.skeletonize(ch_rotated[2] > 0.1)

        # ROTATE ANGLES
        # translate angles to doubleangle vectors (need empty batch dimensions 0)
        ch_doubleangle = rep_2d_numpy.angle_2_da_vector(np.expand_dims(ch[4:5], 0))
        # perform rotation
        ch_doubleangle = scipy.ndimage.rotate(ch_doubleangle, rotation, axes=(2, 3), mode='nearest')
        # translate doubleangle to angles
        ch_direction_rotated = rep_2d_numpy.da_vector_2_angle(ch_doubleangle)
        # rotate the angles themselves
        ch_direction_rotated += rotation * pi / 180
        # normalise angles to range [0, pi)
        ch_direction_rotated = np.remainder(ch_direction_rotated, pi)
        # zero out undefined values
        pass
        # remove batch dimension
        ch_direction_rotated = ch_direction_rotated[0]
        # combine ch_rotated and ch_direction_rotated
        ch = np.concatenate([ch_rotated, ch_direction_rotated], axis=0)
        return ch

    def enforce_order(self, ch_0, ch_1):
        """Determine which chromosome is 0 and which 1. Return them in the correct order."""
        if self.order == 'random':
            switch_chromosomes = self.rng.random() < 0.5
        elif self.order == 'position':
            # determine centre of mass and pick the chromosome with the large value in dimension 0 as chromosome 0
            centre_of_0 = scipy.ndimage.measurements.center_of_mass(ch_0[3])
            centre_of_1 = scipy.ndimage.measurements.center_of_mass(ch_1[3])
            switch_chromosomes = centre_of_0[0] < centre_of_1[0]
        elif self.order == 'orientation':
            # determine the average direction as a unit vector and pick the chromosome with the larger value in
            # dimension 1 as chromosome 0
            sum_0_cos = np.sum(np.cos(ch_0[4]) * ch_0[3])
            sum_0_sin = np.sum(np.sin(ch_0[4]) * ch_0[3])
            sum_1_cos = np.sum(np.cos(ch_1[4]) * ch_1[3])
            sum_1_sin = np.sum(np.sin(ch_1[4]) * ch_1[3])
            direction_0_sin = sum_0_sin / np.sqrt(sum_0_cos ** 2 + sum_0_sin ** 2)
            direction_1_sin = sum_1_sin / np.sqrt(sum_1_cos ** 2 + sum_1_sin ** 2)
            switch_chromosomes = direction_0_sin < direction_1_sin
        elif self.order == 'length':
            # pick the chromosomes with more pixels in the skeleton as chromosome 0
            length_0 = np.sum(ch_0[2])
            length_1 = np.sum(ch_1[2])
            switch_chromosomes = length_0 < length_1
        else:
            raise ValueError(f"self.order {self.order} is not supported."
                             f"Use 'random', 'position', 'orientation' or 'length'")

        if switch_chromosomes:
            return ch_1, ch_0
        else:
            return ch_0, ch_1

    def get_single(self):
        """
        Get a single chromosome image of intersecting chromosomes, where the channels of the image are specified by
        self.output_channels_list.
        The shape of the output is (len(self.output_channels_list), self.imsize[0], self.imsize[1])
        """
        # select random chromosomes
        ch_0, ch_1 = self.get_random_chromosome_pair()

        # randomly rotate chromosomes
        rotation_0 = self.rng.random() * 360
        rotation_1 = self.rng.random() * 360
        ch_0 = SyntheticChromosomeDataset.rotate_chromosome(ch_0, rotation_0)
        ch_1 = SyntheticChromosomeDataset.rotate_chromosome(ch_1, rotation_1)

        # keep trying translations until there is an intersection
        while True:
            translation = (self.rng.standard_normal(2) * np.array(self.imsize) / 6).astype(np.int)
            # translate chromosomes
            ch_0_translated = self._translate_image(ch_0, translation)
            ch_1_translated = self._translate_image(ch_1, -translation)

            # if no intersection, try different translation
            if not np.any(np.logical_and(ch_0_translated[3], ch_1_translated[3])):
                continue

            # enforce the order of ch_0 and ch_1
            ch_0_translated, ch_1_translated = self.enforce_order(ch_0_translated, ch_1_translated)

            output = self.combine_chromosomes(ch_0_translated, ch_1_translated)
            return output

    def __iter__(self):
        if self.fix_random_seed:
            self.rng = np.random.default_rng(seed=0)
        self.current_batch = 0
        worker_info = get_worker_info()
        if worker_info is None:
            self.n_workers = 1
        else:
            self.n_workers = worker_info.num_workers
        return self

    def __next__(self):
        self.current_batch += 1
        if self.current_batch > (self.epoch_batches // self.n_workers):
            raise StopIteration
        output = [self.get_single() for _ in range(self.batchsize)]
        output = np.stack(output, axis=0)
        if self.half_resolution:
            output = output[:, :, ::2, ::2]
        return np.stack(output, axis=0)



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    datasets = [
        # OriginalChromosomeDataset('data/Cleaned_LowRes_13434_overlapping_pairs.h5', [(0, 0.8)], True, True, 1),
        # SyntheticChromosomeDataset('data/separate.pickle', (128, 128), [0, 1, 2, 3], True, 1, 10,
        #                            ['dapi_cy3','3_channel'], 'length', True),
        RealOverlappingChromosomes('data', False, (0.2, 0.9), False, True, 3),
    ]

    from torch.utils.data import DataLoader

    images = []
    for dataset in datasets:
        # noinspection PyTypeChecker
        dataloader = iter(DataLoader(dataset, batch_size=None, num_workers=0))
        for _ in range(20):
            batch = next(dataloader)
            images.append(batch[0, 0, ...])
            for channel in range(batch.shape[1]):
                plt.close()
                plt.imshow(batch[0, channel])
                plt.show()
    print('Done')
