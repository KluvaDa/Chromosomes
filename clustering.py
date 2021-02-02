import os
import numpy as np
import math
from math import pi
import scipy.ndimage.morphology
from scipy import ndimage
import skimage.morphology

from typing import Sequence, Tuple, Union, Optional, List
import representations.rep_2d_numpy


class Clustering:
    """
    Separates direction-style network output into separate chromosomes. Work on a single image (no batch).
    The hyperparameters are saved in the __init__ funciton.
    """
    def __init__(self,
                 minimum_dilated_intersection_area: int = 8,
                 max_distance: int = 8,
                 merge_peaks_distance: int = 2,
                 minimum_clusters_area: int = 5,
                 minimum_adjacent_area: int = 5,
                 direction_sensitivity: float = 0.7,
                 cluster_grow_radius: Union[float, int] = 3.5,
                 max_chromosome_width: int = 26,
                 intersection_grow_radius: Union[float, int] = 3.5,
                 direction_local_weight: float = 0.5):
        """
        Save the hyperparameters.
        :param minimum_dilated_intersection_area: Delete intersections that are smaller
        :param max_distance: The distance that all values in the distance transform are rounded down to
        :param merge_peaks_distance: how far a local maximum has to be from another cluster before considered
                                     a separate class
        :param minimum_clusters_area: Removes clusters with fewer pixels than minimum_clusters.
        :param minimum_adjacent_area: How many touching pixels must two categories have to be considered for merging.
        :param direction_sensitivity: How similar must the direction be in order to merge two categories
        :param cluster_grow_radius: By how much to grow areas when checking for interconnectivity.
        :param max_chromosome_width: Don't join categories if their resulting width is larger than this width
        :param intersection_grow_radius: By how much to grow intersections when trying to merge them
        :param direction_local_weight: When determining direction similarity, what proportion should be used from
            a local neighbourhood of the intersection. (1-weight) used for whole channel direction similarity.

        """
        self.minimum_dilated_intersection_area = minimum_dilated_intersection_area
        self.max_distance = max_distance
        self.merge_peaks_distance = merge_peaks_distance
        self.minimum_clusters_area = minimum_clusters_area
        self.minimum_adjacent_area = minimum_adjacent_area
        self.direction_sensitivity = direction_sensitivity
        self.cluster_grow_radius = cluster_grow_radius
        self.max_chromosome_width = max_chromosome_width
        self.intersection_grow_radius = intersection_grow_radius
        self.direction_local_weight = direction_local_weight

    def direction_2_separate_chromosomes(self,
                                         index_3category: np.ndarray,
                                         dilated_intersection: np.ndarray,
                                         direction_angle: np.ndarray):
        """
        Transform the output of a direction style network to a classification
        :param index_3category: shape: (1, x, y) with indices 0: background, 1: chromosome, 2: intersection
        :param dilated_intersection: binary tensor of shape (1, x, y)
        :param direction_angle: float tensor of the angle defined in areas with index_3category == 1
        :return: A binary tensor of shape (chromosomes, x, y)
        """
        unique = np.equal(index_3category, 1)
        intersection = np.equal(index_3category, 2)
        dilated_intersection = np.greater(dilated_intersection, 0)
        direction_angle = np.mod(direction_angle, pi)
        da_vector = representations.rep_2d_numpy.angle_2_da_vector(direction_angle[np.newaxis, :, :, :])[0]

        # delete areas of intersection that are too small
        intersection, dilated_intersection = self.remove_small_intersection(intersection, dilated_intersection)

        dilated_unique = np.logical_and(unique, np.logical_not(dilated_intersection))

        distance = scipy.ndimage.morphology.distance_transform_cdt(unique)
        distance_dilated = scipy.ndimage.morphology.distance_transform_cdt(dilated_unique)

        clusters_dilated = self.distance_clustering(distance_dilated)
        clusters = self.distance_clustering_with_seed(distance, clusters_dilated)

        channels = cluster_idx_2_channels(clusters)

        # Merge channels not near intersections
        channels = self.merge_channels_not_near_intersections(channels, unique, intersection, da_vector)

        channels = remove_small_channels(channels, self.minimum_clusters_area)

        channels = self.merge_channels_across_intersection_assume_two(channels, intersection, da_vector)

        separate_chromosomes = combine_channels_and_intersection(channels, intersection)

        return separate_chromosomes

    def remove_small_intersection(self, intersection: np.ndarray, dilated_intersection: np.ndarray):
        """
        Deletes areas that are too small in the dilated intersection.
        :param intersection:
        :param dilated_intersection:
        """
        modified_intersection = intersection.copy()
        modified_dilated_intersection = dilated_intersection.copy()

        dilated_intersection = dilated_intersection.copy()
        dilated_intersection_clusters, num_dilated_intersection_clusters = ndimage.label(dilated_intersection)
        for i_dilated_intersection_clusters in range(1, num_dilated_intersection_clusters+1):
            dilated_intersection_cluster = dilated_intersection_clusters == i_dilated_intersection_clusters
            cluster_size = np.sum(dilated_intersection_cluster)
            if cluster_size < self.minimum_dilated_intersection_area:
                modified_dilated_intersection = np.logical_and(modified_dilated_intersection,
                                                               np.logical_not(dilated_intersection_cluster))
                modified_intersection = np.logical_and(modified_intersection,
                                                       np.logical_not(dilated_intersection_cluster))
        return modified_intersection, modified_dilated_intersection

    def distance_clustering(self, distance_image: np.ndarray) -> np.ndarray:
        """
        Perform iterative clustering based on the distance_image provided.
        Local maxima are turned into clusters unless they merge with another cluster within merge_peaks_distance.
        Clusters are grown to lower distances in the image.
        :param distance_image: integer np array of distances to cluster by
        Returns: A integer np array, where the values correspond to chromosome clusters (0 is the background)
        """

        assert distance_image.dtype == np.int or distance_image.dtype == np.int32
        distance_image_clipped = np.clip(distance_image, 0, self.max_distance)

        return self.distance_clustering_with_seed(distance_image_clipped, np.zeros_like(distance_image))

    def distance_clustering_with_seed(self,
                                      distance_image: np.ndarray,
                                      seed_clusters: np.ndarray,
                                      ) -> np.ndarray:
        """
        Perform iterative clustering based on the distance_image provided.
        Local maxima are turned into clusters unless they merge with another cluster within merge_peaks_distance.
        Clusters are grown to lower distances in the image.
        Args:
            distance_image: integer np array of distances to cluster by
            seed_clusters: The initial clusters to start with
        Returns: A integer np array, where the values correspond to chromosome clusters (0 is the background)
        """
        assert distance_image.dtype == np.int32
        assert seed_clusters.dtype == np.int32

        clusters = seed_clusters.copy()
        # ensure clusters have no gaps (in case we merged)
        clusters = remove_cluster_gaps(clusters)

        new_clusters_expiry = [0] * (np.max(clusters) + 1)  # idx 0 for background

        selem = np.ones((1, 3, 3))

        for current_distance in reversed(range(1, np.max(distance_image) + 1)):
            # grow clusters until all points in specific range are covered. Separate points will be added as new labels
            remaining_points = np.ones_like(clusters)
            while True:
                remaining_points_new = np.logical_and(distance_image == current_distance, clusters == 0)
                if np.all(remaining_points == remaining_points_new):
                    # no changes
                    break
                remaining_points = remaining_points_new

                # grow clusters
                for cluster_i in range(1, np.max(clusters) + 1):
                    grown_label = skimage.morphology.binary_dilation(clusters == cluster_i, selem=selem)
                    grown_label = np.logical_and(grown_label, remaining_points)
                    clusters[grown_label] = cluster_i

            # check whether to merge labels (because of local optimums based on merge_peaks_distance)
            clusters_to_check = np.where(np.array(new_clusters_expiry) > 0)[0]
            for cluster_to_check in clusters_to_check:
                grown_area = skimage.morphology.binary_dilation(clusters == cluster_to_check, selem=selem)
                clusters_in_grown_area = set(clusters[grown_area]) - {0}
                if len(clusters_in_grown_area) > 1:
                    cluster_to_merge_to = min(clusters_in_grown_area - {cluster_to_check})
                    clusters[clusters == cluster_to_check] = cluster_to_merge_to

            # ensure clusters have no gaps (in case we merged)
            clusters, new_clusters_expiry = remove_cluster_gaps(clusters, new_clusters_expiry)

            # reduce new_labels_expiry
            new_clusters_expiry = [max(0, value - 1) for value in new_clusters_expiry]

            # check for new clusters
            if np.any(remaining_points):
                new_clusters, n_new_clusters = ndimage.label(remaining_points)
                max_existing_cluster = np.max(clusters)
                for new_cluster_i in range(1, n_new_clusters + 1):
                    cluster_i = max_existing_cluster + new_cluster_i
                    clusters[new_clusters == new_cluster_i] = cluster_i
                    new_clusters_expiry.append(self.merge_peaks_distance)

        return clusters

    def merge_channels_not_near_intersections(self,
                                              channels: np.ndarray,
                                              union: np.ndarray,
                                              intersection: np.ndarray,
                                              da_vector: np.ndarray):
        """
        Merges adjacent clusters (represented as channels), only if they are far from any intersections.
        Ensures that adjacent clusters are not adjacent next to an intersection
        Ensures that adjacent clusters touch with a sufficient number of pixels
        Ensures that the average direction of the clusters is not too different
        Ensures that the max width of the clusters is not too big after joining
        Args:
            channels: np array, channel first, channels represent clusters (no background)
            union: 2D np array, boolean
            intersection: 2D np array, boolean
            da_vector: np array, channel first, double-angle direction representation
        Returns: np array, channel first, channels represent clusters (no background)
        """

        channels = sort_channels_by_size(channels, large_first=False)  # combine small clusters first

        channels_grown = skimage.morphology.binary_dilation(
            channels,
            selem=get_disk(self.cluster_grow_radius)[None, :, :]
        )

        channels_to_delete = []
        for channel_1 in range(channels.shape[0]):
            for channel_2 in range(channel_1 + 1, channels.shape[0]):
                channel_grown_intersection = np.logical_and(channels_grown[channel_1], channels_grown[channel_2])
                channel_intersection = np.logical_and(channel_grown_intersection, union)

                # if intersection is too small then continue
                sum_channel_intersection = np.sum(channel_intersection)
                if sum_channel_intersection / 2 < self.minimum_adjacent_area:
                    continue

                # if channel_grown_intersection is close to an intersection then continue
                if np.any(np.logical_and(channel_grown_intersection, intersection)):
                    continue

                # if direction discrepancy in intersection is too big the continue
                average_da_vector = calculate_average_da_vector(channel_intersection[0], da_vector, False)
                magnitude_average_da_vector = np.sqrt(average_da_vector[0] ** 2 + average_da_vector[1] ** 2)
                if magnitude_average_da_vector < self.direction_sensitivity:
                    continue

                # ensure distance is not too big after joining
                combined_categories = np.logical_or(channels[channel_1], channels[channel_2])
                if np.max(ndimage.distance_transform_cdt(combined_categories)) > self.max_chromosome_width / 2:
                    continue

                # combine channel_1 and channel_2 to channel_2 (if combined to channel_1 some comparisons could get lost)
                channels[channel_2] = combined_categories
                channels[channel_1] = 0
                channels_to_delete.append(channel_1)
        channels = np.delete(channels, channels_to_delete, axis=0)
        return channels

    def merge_channels_across_intersection_assume_two(self,
                                                      channels: np.ndarray,
                                                      intersection: np.ndarray,
                                                      da_vector: np.ndarray):
        """
        Merges clusters (expressed as channels) across intersections.
        Finds channels that have the most similar direction to merge.

        Args:
            channels: np array, channel first, channels represent clusters (no background)
            intersection: 2D np array, boolean
            da_vector: np array, channel first, double-angle direction representation
        Returns: np array, channel first, channels represent clusters (no background)
        """

        # ensure that there are intersections:
        if np.sum(intersection) == 0:
            return channels

        # separate intersection into channels (if multiple intersections)
        grown_intersection = skimage.morphology.binary_dilation(
            intersection,
            selem=get_disk(self.intersection_grow_radius)[None, :, :]
        )
        grown_intersection_clusters, _ = ndimage.label(grown_intersection)

        grown_intersection_channels = cluster_idx_2_channels(grown_intersection_clusters)
        intersection_channels = np.logical_and(grown_intersection_channels, np.expand_dims(intersection, axis=0))

        for intersection_channel_i in range(intersection_channels.shape[0]):
            channels_to_delete = []
            # find channels in intersection surroundings
            channels_in_intersection_surrounding = np.logical_and(
                channels,
                grown_intersection_channels[intersection_channel_i: intersection_channel_i + 1]
            )
            channel_idxes_in_intersection_surrounding = np.any(channels_in_intersection_surrounding, axis=(1, 2))
            channel_idxes_in_intersection_surrounding = np.where(channel_idxes_in_intersection_surrounding)[0]

            # calculate channel angle
            average_da_vectors = {}
            average_da_intersection_vectors = {}
            for channel in channel_idxes_in_intersection_surrounding:
                average_da_intersection = calculate_average_da_vector(
                    channels_in_intersection_surrounding[channel],
                    da_vector,
                    True
                )
                average_da = calculate_average_da_vector(
                    channels[channel],
                    da_vector,
                    True
                )
                average_da_intersection_vectors[channel] = average_da_intersection
                average_da_vectors[channel] = average_da

            while len(channel_idxes_in_intersection_surrounding) - len(channels_to_delete) > 2:

                best_similarity = 0
                best_channel_1 = None
                best_channel_2 = None
                for channel_1 in channel_idxes_in_intersection_surrounding:
                    for channel_2 in channel_idxes_in_intersection_surrounding:
                        if channel_2 <= channel_1:
                            continue
                        if channel_2 in channels_to_delete:
                            continue
                        if channel_1 in channels_to_delete:
                            continue

                        # angle similarity
                        average_da_intersection = (
                            (average_da_intersection_vectors[channel_1][0] + average_da_intersection_vectors[channel_2][
                                0]) / 2,
                            (average_da_intersection_vectors[channel_1][1] + average_da_intersection_vectors[channel_2][
                                1]) / 2
                        )

                        average_da = (
                            (average_da_vectors[channel_1][0] + average_da_vectors[channel_2][0]) / 2,
                            (average_da_vectors[channel_1][1] + average_da_vectors[channel_2][1]) / 2
                        )

                        intersection_direction_similarity = np.sqrt(average_da_intersection[0] ** 2 +
                                                                    average_da_intersection[1] ** 2)

                        direction_similarity = np.sqrt(average_da[0] ** 2 +
                                                       average_da[1] ** 2)

                        weighted_direction_similarity = intersection_direction_similarity * self.direction_local_weight + \
                                                        direction_similarity * (1 - self.direction_local_weight)

                        if weighted_direction_similarity > best_similarity:
                            best_similarity = weighted_direction_similarity
                            best_channel_1 = channel_1
                            best_channel_2 = channel_2

                # combine channels
                combined_channels = np.logical_or(channels[best_channel_1], channels[best_channel_2])
                channels[best_channel_1] = combined_channels
                channels[best_channel_2] = 0
                channels_to_delete.append(best_channel_2)

                channels_in_intersection_surrounding = np.logical_and(
                    channels,
                    grown_intersection_channels[intersection_channel_i: intersection_channel_i + 1]
                )

                average_da_intersection = calculate_average_da_vector(
                    channels_in_intersection_surrounding[best_channel_1],
                    da_vector,
                    True
                )
                average_da = calculate_average_da_vector(
                    channels[best_channel_1],
                    da_vector,
                    True
                )
                average_da_intersection_vectors[best_channel_1] = average_da_intersection
                average_da_vectors[best_channel_1] = average_da

            channels = np.delete(channels, channels_to_delete, axis=0)

        return channels


def remove_cluster_gaps(clusters: np.ndarray,
                        new_clusters_expiry: Optional[List[int]] = None) \
        -> Union[np.ndarray, Tuple[np.ndarray, List[int]]]:
    """
    Takes an image with indices corresponding to clusters and removes any indices that have no entries
    Args:
        clusters: int np array of indices
        new_clusters_expiry: Optional expiry list that will also get adjusted if categories are deleted
    Returns: the adjusted clusters image (or the clusters image and the new_clusters_expiry)
    """
    i_cluster = 0
    while i_cluster <= np.max(clusters):
        if i_cluster not in clusters:
            clusters[clusters > i_cluster] -= 1
            if new_clusters_expiry is not None:
                del new_clusters_expiry[i_cluster]
        else:
            i_cluster += 1

    if new_clusters_expiry is not None:
        return clusters, new_clusters_expiry
    else:
        return clusters


def remove_small_channels(channels: np.ndarray, min_area: int) -> np.ndarray:
    """
    Removes channels that have fewer than min_area points
    :param channels: np array, channel first, channels represent clusters (no background)
    :param min_area: minimum number of pixels for a channel to be kept
    Returns: modified channels array
    """
    channels_to_delete = []
    for i_channel, channel in enumerate(channels):
        if np.sum(channel) < min_area:
            channels_to_delete.append(i_channel)

    channels = np.delete(channels, channels_to_delete, axis=0)
    return channels


def calculate_average_da_vector(mask: np.ndarray, da_vector: np.ndarray, normalise: bool = False) \
        -> Tuple[float, float]:
    """
    Returns the mean da_vector in a given area.
    Args:
        mask: np array of shape (x, y), defining the area to average in
        da_vector: np array, channel first, double-angle direction representation
        normalise: Whether to normalise the magnitude of the resultant vector.
    Returns: A vector as a tuple.
    """
    average_da_vector = (np.sum(da_vector[0][mask]), np.sum(da_vector[1][mask]))

    if normalise:
        magnitude = np.sqrt(average_da_vector[0] ** 2 + average_da_vector[1] ** 2)
    else:
        magnitude = np.sum(mask)

    if magnitude == 0:
        raise ValueError("magnitude is 0")

    average_da_vector = (average_da_vector[0] / magnitude, average_da_vector[1] / magnitude)

    return average_da_vector


def combine_channels_and_intersection(channels: np.ndarray,
                                      intersection: np.ndarray) -> np.ndarray:
    """
    Adds areas of intersection to channels that are adjacent to them
    Args:
        channels: np array, channel first, channels represent clusters (no background)
        intersection: 2D np array, boolean
    Returns: np array, channel first, like channels but also contains adjacent intersections
    """
    # ensure that there are intersections:
    if np.sum(intersection) == 0:
        return channels

    # separate intersection into channels (if multiple intersections)
    grown_intersection = skimage.morphology.binary_dilation(intersection, selem=np.ones((1, 3, 3)))
    grown_intersection_clusters, _ = ndimage.label(grown_intersection)

    grown_intersection_channels = cluster_idx_2_channels(grown_intersection_clusters)
    intersection_channels = np.logical_and(grown_intersection_channels, intersection)

    for intersection_channel_i in range(intersection_channels.shape[0]):
        # find channels in intersection surroundings
        channels_in_intersection_surrounding = np.logical_and(
            channels,
            grown_intersection_channels[intersection_channel_i: intersection_channel_i + 1]
        )
        channel_idxes_in_intersection_surrounding = np.any(channels_in_intersection_surrounding, axis=(1, 2))
        channel_idxes_in_intersection_surrounding = np.where(channel_idxes_in_intersection_surrounding)[0]
        for channel_idx in channel_idxes_in_intersection_surrounding:
            channels[channel_idx] = np.logical_or(channels[channel_idx],
                                                  intersection_channels[intersection_channel_i])
    return channels


def sort_channels_by_size(channels: np.ndarray, large_first: bool = False) -> np.ndarray:
    """ Sort clusters (represented as channels) by their size
    Args:
        channels: np array, channel first, channels represent clusters (no background)
        large_first: whether to sort in reverse order
    """
    sum_channels = np.sum(channels, axis=(1, 2))
    channels_order = np.argsort(sum_channels)
    if large_first:
        channels_order = np.array(list(reversed(channels_order)))
    channels = channels[channels_order]
    return channels


def cluster_idx_2_channels(clusters: np.ndarray) -> np.ndarray:
    """
    Changes cluster indices to channels. Clusters have 0 for background, which is not transformed to a channel
    Args:
        clusters: 2D np array of int indices
    Returns: channels first np array of channels
    """
    if np.max(clusters) == 0:
        return np.zeros((0, clusters.shape[1], clusters.shape[2]))
    channels = np.concatenate([clusters == idx for idx in range(1, np.max(clusters) + 1)], axis=0)
    return channels


def cluster_channels_2_idx(channels: np.ndarray) -> np.ndarray:
    """
    Changes cluster channels to indices. Index 0 is added for background
    Args:
        channels: channels first np array of channels
    Returns: 2D np array of int indices (0 for background with no channel value
    """
    indices = np.argmax(channels, axis=0) + 1
    indices[np.logical_not(np.any(channels, axis=0))] = 0  # add index 0 for background
    return indices


def get_disk(radius: Union[int, float]) -> np.ndarray:
    """ Returns a numpy boolean array with a disk (filled circle). """
    radius_floor = math.floor(radius)
    y, x = np.ogrid[-radius_floor:radius_floor + 1, -radius_floor:radius_floor + 1]
    disk = x ** 2 + y ** 2 <= radius ** 2
    return disk
