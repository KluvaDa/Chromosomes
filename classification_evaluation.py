import torch
import pytorch_lightning as pl
import os
import pathlib
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import pandas as pd

from tensorboard.backend.event_processing import event_accumulator
from typing import Dict

import classification
import tensorboard_reader


def interpret_dirname(dirname: str):
    # deduce parameters
    (train_on_original, segment_4_categories, smaller_network, shuffle_first, category_order) = \
        dirname.split('_')
    train_on_original = True if train_on_original == 'original' else False
    segment_4_categories = True if segment_4_categories == '4category' else False
    smaller_network = True if smaller_network == 'snet' else False
    shuffle_first = True if shuffle_first == 'shuffled' else False
    # category order unchanged

    return train_on_original, segment_4_categories, smaller_network, shuffle_first, category_order


def load_module(dirpath: str, cross_validation: int):
    """
    Loads the pytorch lightning module. Always loads version_0.
    :param dirpath: String path
    :param cross_validation: Which cross validation run to use.
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
    classification_module = classification.ClassificationModule.load_from_checkpoint(checkpoint)
    return classification_module


def evaluate(root_path: str, run_name: str, i_cv: int, clustering=None):
    """
    Runs validation and testing and returns the metrics dictionary
    :param root_path: Path to where the runs are saved
    :param run_name: Name of the run
    :param i_cv: The cross validation run in (0, 1, 2, 3) to evaluate
    :param clustering: a clustering object that overrides the default clustering parameters
    """
    train_on_original, segment_4_categories, smaller_network, shuffle_first, category_order = \
        interpret_dirname(run_name)

    dirpath = os.path.join(root_path, run_name)
    data_module = classification.ClassificationDataModule(train_on_original=train_on_original,
                                                          cross_validation_i=i_cv,
                                                          segment_4_categories=segment_4_categories,
                                                          shuffle_first=False,
                                                          category_order=category_order)
    module = load_module(dirpath, i_cv)
    trainer = pl.Trainer(gpus=1)
    test_metrics = trainer.test(module, datamodule=data_module)
    test_metrics = pd.DataFrame(test_metrics)
    test_metrics = test_metrics.mean()  # no repetition, should merely flatten
    return test_metrics

def evaluate_average_cv(root_path: str, run_name: str):
    """
    Finds the metrics averaged over the cross-validation runs
    :param root_path: Path to where the runs are saved
    :param run_name: Name of the run
    """
    all_cv_metrics = []
    for i_cv in (0, 1, 2, 3):
        test_metrics = evaluate(root_path, run_name, i_cv)
        all_cv_metrics.append(test_metrics)
    average_metrics = pd.concat(all_cv_metrics)
    average_metrics = average_metrics.groupby(average_metrics.index).mean()
    return average_metrics


def evaluate_all():
    """
    Evaluates all runs and saves the results as as csv file in results/instance_segmentation_test_metrics.csv
    """
    root_path = 'results/classification_good'
    run_names = os.listdir(root_path)
    all_metrics = dict()
    for run_name in run_names:
        if os.path.isdir(os.path.join(root_path, run_name)):
            metrics = evaluate_average_cv(root_path, run_name)
            all_metrics[run_name] = metrics
    all_metrics = pd.DataFrame(all_metrics)
    all_metrics.to_csv('results/classification_test_metrics.csv')


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
        renamed_metrics = {metric_name.split('/')[0]: value for metric_name, value in metrics.items()}
        all_metrics[run_name] = renamed_metrics
    return all_metrics


def scalars_averaged_over_epochs(scalars, metric_names):
    max_epoch = int(np.max(scalars['epoch'][1]))
    step_epochs = dict(zip(scalars['epoch'][0], scalars['epoch'][1]))

    metrics = pd.DataFrame(0, index=range(max_epoch + 1), columns=metric_names)
    for metric_name in metric_names:
        metric_steps = scalars[metric_name][0]
        metric_epoch = np.array([int(step_epochs[metric_step]) for metric_step in metric_steps])

        metric_values = scalars[metric_name][1]
        for epoch_i in range(max_epoch+1):
            average_metric = np.average(metric_values[np.where(metric_epoch == epoch_i)])
            metrics.loc[epoch_i, metric_name] = average_metric

    return metrics


def save_metrics_as_csv():
    """
    Reads metrics from tensorboard logs, finds metrics at best performance and averages them out over cross-validation.
    The resulting metrics are saved in 'results/classification_metrics.csv'
    """
    root_path = 'results/classification'
    all_metrics = load_tensorboard(root_path)
    dataframe = pd.DataFrame(all_metrics).transpose()
    dataframe.to_csv('results/classification_metrics.csv')


def graph_metrics(root_path, run_name, metric_names):
    scalars_list = tensorboard_reader.read_scalars_in_directory(os.path.join(root_path, run_name))
    epoch_metrics_list = []
    for scalars in scalars_list:
        scalars = {name.split('/')[0]: value for name, value in scalars.items()}
        epoch_metrics = scalars_averaged_over_epochs(scalars, metric_names)
        epoch_metrics_list.append(epoch_metrics)
    epoch_metrics = pd.concat(epoch_metrics_list)
    epoch_metrics = epoch_metrics.groupby(epoch_metrics.index).mean()
    epoch_metrics.plot(color=['#3182bd', '#6baed6', '#9ecae1', '#c6dbef',
                              '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2',
                              '#31a354', '#74c476', '#a1d99b', '#c7e9c0'
                              ])


def save_images(root_path, run_name, n_images, cross_validation=0):
    train_on_original, segment_4_categories, smaller_network, shuffle_first, category_order = \
        interpret_dirname(run_name)

    classification_module = load_module(os.path.join(root_path, run_name), cross_validation=0)
    data_module = classification.ClassificationDataModule(train_on_original=train_on_original,
                                                          cross_validation_i=cross_validation,
                                                          segment_4_categories=segment_4_categories,
                                                          shuffle_first=False,
                                                          category_order=category_order)
    data_module.setup()
    dataloader_synthetic, dataloader_real, dataloader_original = data_module.val_dataloader()
    iterator_synthetic = iter(dataloader_synthetic)
    iterator_real = iter(dataloader_real)
    iterator_original = iter(dataloader_original)
    data_iterators = {'synthetic': iterator_synthetic, 'real': iterator_real, 'original': iterator_original}
    for dataset_name, dataset_iterator in data_iterators.items():
        image_i = 0
        while image_i < n_images:
            batch = next(dataset_iterator)
            if image_i >= n_images:
                break
            batch_in = batch[:, 0:1, ...]
            batch_label = batch[:, 1:, ...]
            batch_prediction = torch.argmax(classification_module(batch_in), dim=1, keepdim=True)

            batch_in = batch_in.detach().cpu().numpy()
            batch_label = batch_label.detach().cpu().numpy()
            batch_prediction = batch_prediction.detach().cpu().numpy()

            for image_in, image_label, image_prediction in zip(batch_in, batch_label, batch_prediction):
                if image_i >= n_images:
                    break
                plt.clf()

                plt.subplot(131)
                plt.imshow(image_in[0], cmap='gray', vmin=0, vmax=1)
                plt.axis('off')

                plt.subplot(132)
                plt.imshow(image_label[0].astype(int), cmap='tab10', vmin=0, vmax=9)
                plt.axis('off')

                plt.subplot(133)
                plt.imshow(image_prediction[0], cmap='tab10', vmin=0, vmax=9)
                plt.axis('off')

                plt.savefig(os.path.join(root_path, run_name, f"{dataset_name}_cv{cross_validation}_{image_i:03}.png"),
                            bbox_inches='tight',
                            dpi=200)

                image_i += 1


def save_all_images(n_images):
    root_path = 'results/classification'
    run_names = os.listdir(root_path)
    for run_name in run_names:
        if len(run_name.split('_')) == 5:
            save_images(root_path, run_name, n_images, 0)


if __name__ == '__main__':
    # save_metrics_as_csv()

    evaluate_all()

    # save_all_images(10)

    # root_path = 'results/classification_old'
    #
    # metrics_names = ['val_original_iou_background',
    #                  'val_original_iou_ch0_best_match',
    #                  'val_original_iou_ch1_best_match',
    #                  'val_original_iou_overlap',
    #                  'val_synthetic_iou_background',
    #                  'val_synthetic_iou_ch0_best_match',
    #                  'val_synthetic_iou_ch1_best_match',
    #                  'val_synthetic_iou_overlap',
    #                  'val_real_iou_background',
    #                  'val_real_iou_ch0_best_match',
    #                  'val_real_iou_ch1_best_match',
    #                  'val_real_iou_overlap'
    #                  ]
    #
    #
    # # graph_metrics(root_path, 'original_4category_snet_shuffled_random', metrics_names)
    # # plt.show()
    # # graph_metrics(root_path, 'original_4category_lnet_shuffled_random', metrics_names)
    # # plt.show()
    # # graph_metrics(root_path, 'original_4category_snet_ordered_random', metrics_names)
    # # plt.show()
    # # graph_metrics(root_path, 'original_4category_lnet_ordered_random', metrics_names)
    # # plt.show()
    #
    #
    # all_metrics = load_tensorboard(root_path)
    # df = pd.DataFrame(all_metrics).transpose()
    #
    # original_metric_names = ['val_original_iou_background',
    #                          'val_original_iou_ch0_best_match',
    #                          'val_original_iou_ch1_best_match',
    #                          'val_original_iou_overlap']
    # synthetic_metric_names = ['val_synthetic_iou_background',
    #                           'val_synthetic_iou_ch0_best_match',
    #                           'val_synthetic_iou_ch1_best_match',
    #                           'val_synthetic_iou_overlap']
    # real_metric_names = ['val_real_iou_background',
    #                      'val_real_iou_ch0_best_match',
    #                      'val_real_iou_ch1_best_match',
    #                      'val_real_iou_overlap']
    #
    # run_names = [('original_4category_snet_ordered_random', 'Trained on Original Dataset, Ordered, Small Net'),
    #              ('original_4category_snet_shuffled_random', 'Trained on Original Dataset, Shuffled, Small Net'),
    #              ('original_4category_lnet_ordered_random', 'Trained on Original Dataset, Ordered, Large Net'),
    #              ('original_4category_lnet_shuffled_random', 'Trained on Original Dataset, Shuffled, Large Net'),
    #              ('synthetic_4category_snet_ordered_random', 'Trained on Synthetic Dataset, Random Ordering, Small Net'),
    #              ('synthetic_4category_snet_ordered_position', 'Trained on Synthetic Dataset, Positional Ordering, Small Net'),
    #              ('synthetic_4category_snet_ordered_length', 'Trained on Synthetic Dataset, Length-Based Ordering, Small Net'),
    #              ('synthetic_4category_snet_ordered_orientation', 'Trained on Synthetic Dataset, Orientation-Based Ordering, Small Net'),
    #              ('synthetic_4category_lnet_ordered_random', 'Trained on Synthetic Dataset, Random Ordering, Large Net'),
    #              ('synthetic_4category_lnet_ordered_position', 'Trained on Synthetic Dataset, Positional Ordering, Large Net'),
    #              ('synthetic_4category_lnet_ordered_length', 'Trained on Synthetic Dataset, Length-Based Ordering, Large Net'),
    #              ('synthetic_4category_lnet_ordered_orientation', 'Trained on Synthetic Dataset, Orientation-Based Ordering, Large Net'),
    #              ]
    #
    # for run_name, title in run_names:
    #     original = df.loc[run_name, original_metric_names].to_numpy()
    #     synthetic = df.loc[run_name, synthetic_metric_names].to_numpy()
    #     real = df.loc[run_name, real_metric_names].to_numpy()
    #     data = np.stack([original, synthetic, real])
    #     visualisation_df = pd.DataFrame(data,
    #                                         index=['original', 'synthetic', 'real'],
    #                                         columns=['background', 'chromosome 0', 'chromosome 1', 'overlap'])
    #     visualisation_df.plot.bar(rot=0).legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #     plt.title(title)
    #     plt.savefig(f'results/{run_name}.png', bbox_inches='tight')
    #
    #
    #
    # # df.loc[['original_4category_snet_shuffled_random',
    # #         'original_4category_lnet_shuffled_random',
    # #         'original_4category_snet_ordered_random',
    # #         'original_4category_lnet_ordered_random'],
    # #        ['val_original_iou_background',
    # #         'val_original_iou_ch0_best_match',
    # #         'val_original_iou_ch1_best_match',
    # #         'val_original_iou_overlap',
    # #         'val_synthetic_iou_background',
    # #         'val_synthetic_iou_ch0_best_match',
    # #         'val_synthetic_iou_ch1_best_match',
    # #         'val_synthetic_iou_overlap'
    # #         ]].plot.bar(color=['#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2'])
    # #
    # # plt.savefig('results/original_4category.png', bbox_inches='tight')
    # #
    # # df.loc[['original_4category_snet_shuffled_random',
    # #         'original_4category_lnet_shuffled_random',
    # #         'original_4category_snet_ordered_random',
    # #         'original_4category_lnet_ordered_random'],
    # #        ['val_synthetic_iou_background',
    # #         'val_synthetic_iou_ch0_best_match',
    # #         'val_synthetic_iou_ch1_best_match',
    # #         'val_synthetic_iou_overlap']].plot.bar()
    # #
    # # plt.savefig('results/original_4category_original.png', bbox_inches='tight')
    #
    # # runs = os.listdir(root_path)
    # # run = runs[0]
    # #
    # # classification_module = load_module(os.path.join(root_path, runs[0]))
    # # print(classification_module.segment_4_categories)
