import os
import sys
import random
import argparse
from tqdm import tqdm
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import pickle as pkl
from tqdm import tqdm
from diffuser.utils import set_seed
from diffuser.models.vae import VAE
import torch.nn.functional as F
from train_vae import main as train_vae_main


import design_bench

from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
from design_bench.datasets.discrete.tf_bind_10_dataset import TFBind10Dataset
from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset
from design_bench.datasets.continuous.superconductor_dataset import SuperconductorDataset

import torch
import numpy as np

TASKNAME2FULL = {
    'dkitty': DKittyMorphologyDataset,
    'ant': AntMorphologyDataset,
    'tfbind8': TFBind8Dataset,
    'tfbind10': TFBind10Dataset,
    'superconductor': SuperconductorDataset,
}

TASKNAME2TASK = {
    'dkitty': 'DKittyMorphology-Exact-v0',
    'ant': 'AntMorphology-Exact-v0',
    'tfbind8': 'TFBind8-Exact-v0',
    'tfbind10': 'TFBind10-Exact-v0',
    'superconductor': 'Superconductor-RandomForest-v0',
}

TASKNAME2MAX_SAMPLES = {
    'dkitty': 10004,
    'ant': 10004,
    'tfbind8': 32898,
    'tfbind10': 50000,
    'superconductor': 17014,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, choices=list(TASKNAME2TASK.keys()), default='tfbind8')
    parser.add_argument('--frac', type=float, default=1.0)
    parser.add_argument('--sigma', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    set_seed(args.seed)

    task = design_bench.make(TASKNAME2TASK[args.task],
                                dataset_kwargs=dict(
                                max_samples=int(TASKNAME2MAX_SAMPLES[args.task] * args.frac),
                                distribution=None,
                                min_percentile=0)
                            )
    fully_observed_task = TASKNAME2FULL[args.task]()

    
    
    
    
    task.map_to_logits()
    data_x = torch.from_numpy(task.x.reshape(task.x.shape[0], -1)).float()
    data_y = torch.from_numpy(task.y).float()
    
    print(data_x.shape[1])