import argparse
import json
import os
import sys
from pprint import pprint

import numpy as np
from easydict import EasyDict as edict
import torch


def parse_args():
    """
    Parse the arguments of the program
    :return: (config_args)
    :rtype: tuple
    """
    # Create a parser
    parser = argparse.ArgumentParser(description="MobileNet-V2 PyTorch Implementation")
    parser.add_argument('--version', action='version', version='%(prog)s 0.0.1')
    parser.add_argument('--config', default=None, type=str, help='Configuration file')

    # Parse the arguments
    args = parser.parse_args()

    # Parse the configurations from the config json file provided
    try:
        if args.config is not None:
            with open(args.config, 'r') as config_file:
                config_args_dict = json.load(config_file)
        else:
            print("Add a config file using \'--config file_name.json\'", file=sys.stderr)
            exit(1)

    except FileNotFoundError:
        print("ERROR: Config file not found: {}".format(args.config), file=sys.stderr)
        exit(1)
    except json.decoder.JSONDecodeError:
        print("ERROR: Config file is not a proper JSON file!", file=sys.stderr)
        exit(1)

    config_args = edict(config_args_dict)

    pprint(config_args)
    print("\n")

    return config_args


def create_experiment_dirs(exp_dir):
    """
    Create Directories of a regular tensorflow experiment directory
    :param exp_dir:
    :return summary_dir, checkpoint_dir:
    """
    experiment_dir = os.path.realpath(
        os.path.join(os.path.dirname(__file__))) + "/experiments/" + exp_dir + "/"
    summary_dir = experiment_dir + 'summaries/'
    checkpoint_dir = experiment_dir + 'checkpoints/'

    dirs = [summary_dir, checkpoint_dir]
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        print("Experiment directories created!")
        # return experiment_dir, summary_dir, checkpoint_dir
        return experiment_dir, summary_dir, checkpoint_dir
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def calc_dataset_stats(dataset, axis=0, ep=1e-7):
    data = []
    for i in dataset:
        data.append(i[0])
    data = torch.stack(data)
    return (torch.mean(data, dim=axis) / 255.0).tolist(), (
            torch.std(data + ep, dim=axis) / 255.0).tolist()

def calc_dataset_stats(dataset, axis=0, ep=1e-7):
    rolling_mean = 0
    variance = 0
    n = len(dataset)
    for i in dataset:
        rolling_mean += torch.mean(i[0])
    rolling_mean /= n
    for i in dataset:
        variance += torch.mean(torch.sum(i[0] - rolling_mean.tolist()) ** 2)
    variance /= (n - 1)
    variance ** 0.5
    return rolling_mean.tolist() / 255, variance.tolist() / 255

class AverageTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
