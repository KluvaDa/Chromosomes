from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pathlib
from typing import Dict, Tuple, List


def read_scalars(path: str) -> Dict[str, np.ndarray]:
    """
    Reads scalar values from a tensorboard event as numpy arrays
    Args:
        :param path: Path to a tensorboard event
    :return: {'scalar_name': np.ndarray([step], [value], [wall_time])}
    """
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    scalar_names = ea.Tags()['scalars']
    scalars = dict()
    for scalar_name in scalar_names:
        scalar_data_step = [scalar.step for scalar in ea.Scalars(scalar_name)]
        scalar_data_value = [scalar.value for scalar in ea.Scalars(scalar_name)]
        scalar_data_wall_time = [scalar.wall_time for scalar in ea.Scalars(scalar_name)]
        scalars[scalar_name] = np.array([scalar_data_step, scalar_data_value, scalar_data_wall_time])
    return scalars


def read_scalars_in_directory(path: str) -> List[Dict[str, np.ndarray]]:
    """
    Loads all tensorboard events in a directory. Does not discriminate between the events and returns them in a list.
    Args:
        path: Path to the directory with tensorboard events
    Returns: A list of dictionaries
    """
    directory = pathlib.Path(path)
    events_data = []
    for event in directory.glob('**/events.out.tfevents.*'):
        event_path = event.as_posix()
        event_data = read_scalars(event_path)
        events_data.append(event_data)
    return events_data


def find_best_metrics(scalars, main_metric_name, maximise=False) -> Dict[str, float]:
    """
    Finds metrics at the step when the main_metric is optimised
    Args:
        scalars: A dict of tensorboard scalars (from read_scalars)
        main_metric_name: the name of the main metric to use
        maximise: whether to maximise the metric (True) or minimise (False)
    Returns: Dict[str, float]
    """
    assert main_metric_name in scalars, f"'{main_metric_name}' does not exist in the scalars dict"
    if maximise:
        best_index = np.argmax(scalars[main_metric_name][1])
    else:
        best_index = np.argmin(scalars[main_metric_name][1])
    best_step = scalars[main_metric_name][0][best_index]
    best_metrics = dict.fromkeys(scalars, 0.0)
    for metric_name in best_metrics.keys():
        metric_index = np.argmin(np.abs(scalars[metric_name][0] - best_step))  # find the entry with the closest step
        metric_value = scalars[metric_name][1][metric_index]
        best_metrics[metric_name] = metric_value
    return best_metrics


def find_best_metrics_averaged(scalars_list, main_metric_name, maximise=False) -> Dict[str, float]:
    """
    Averages the best metrics over multiple runs. Only metric values that exist in each list are kept.
    Args:
        scalars_list: A list of scalars returned by read_scalars_in_directory
        main_metric_name: the name of the main metric to use
        maximise: whether to maximise the metric (True) or minimise (False)
    Returns: Dict[str, float]
    """
    best_metrics_list = []
    for scalars in scalars_list:
        best_metrics = find_best_metrics(scalars, main_metric_name, maximise)
        best_metrics_list.append(best_metrics)
    metric_names = set.intersection(*(set(scalars.keys()) for scalars in best_metrics_list))
    average_best_metrics = dict.fromkeys(metric_names, 0.0)
    for metric_name in metric_names:
        average_best_metric = np.average([best_metrics[metric_name] for best_metrics in best_metrics_list])
        average_best_metrics[metric_name] = average_best_metric
    return average_best_metrics


def save_dict(data_dict: Dict[str, float],
              path: str) -> None:
    """
    Saves a text file with the data from the dictionary
    Args:
        data_dict: The data to save
        path: the path to the file to write to
    """
    with open(path, 'w') as f:
        for key in sorted(data_dict):
            f.write(f"{key},{data_dict[key]}\n")
    return


def read_dict(path) -> Dict[str, float]:
    """
    Loads a text file saved by save_dict.
    Expected file format: "str,float\n"
    Args:
        path: Path to text file
    Returns: dictionary with values
    """
    data_dict = dict()
    with open(path, 'r') as f:
        for line in f:
            key, value = line.split(',')
            data_dict[key] = float(value)
    return data_dict
