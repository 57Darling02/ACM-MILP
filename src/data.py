import os
import pickle
import numpy as np

import pandas as pd
import torch
from torch import FloatTensor, LongTensor
from torch.utils.data import BatchSampler
from torch_geometric.data import Data, Dataset


class BipartiteGraph(Data):
    masked_cons_idx: int
    bias_label: FloatTensor
    degree_label: LongTensor
    logits_label: FloatTensor
    connected_vars_idx: LongTensor
    weights_label: FloatTensor

    def __init__(
        self,
        x_constraints: FloatTensor = None,
        edge_index: LongTensor = None,
        edge_attr: FloatTensor = None,
        x_variables: FloatTensor = None,
        **kwargs
    ):
        """
        Bipartite graph data object, each representing a MILP instance.
        """
        super().__init__(**kwargs)
        self.x_constraints = x_constraints
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.x_variables = x_variables
        if x_constraints is not None:
            self.num_nodes = len(x_constraints) + len(x_variables)

    def __inc__(self, key, value, store, *args, **kwargs):
        if key == "edge_index":
            return torch.tensor(
                [[self.x_constraints.size(0)], [self.x_variables.size(0)]]
            )
        if key == "masked_cons_idx":
            return self.x_constraints.size(0)
        elif key == "connected_vars_idx":
            return self.x_variables.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    @property
    def num_variables(self):
        return len(self.x_variables)

    @property
    def num_constraints(self):
        return len(self.x_constraints)


def list_relative_files(root_dir: str, suffix: str):
    files = []
    for current_root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith(suffix):
                continue
            rel_dir = os.path.relpath(current_root, root_dir)
            if rel_dir == ".":
                files.append(filename)
            else:
                files.append(os.path.join(rel_dir, filename))
    files.sort()
    return files


class InstanceDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 community_dir: str,
                 solving_results_path: str = None,
                 num_instances: int = None,
                 ):
        """
        Dataset for instances.

        Args:
            data_dir: path to the directory saving the instances
            community_dir: path to the community information
            solving_results_path: path to the csv file saving the solving results
            num_instances: number of instances to use
        """
        super().__init__()
        self.data_dir = data_dir
        self.community_dir = community_dir
        self.files = list_relative_files(data_dir, ".pkl")
        community_files = list_relative_files(community_dir, ".npy")
        community_map = {
            os.path.splitext(file)[0]: file for file in community_files
        }
        self.solving_path = solving_results_path

        if self.solving_path:
            self.solving_results = pd.read_csv(
                solving_results_path).set_index("instance")
            self.solving_time_mean = self.solving_results.loc[:, "solving_time"].mean(
            )
            self.solving_time_std = self.solving_results.loc[:, "solving_time"].std(
            )
            self.num_nodes_mean = self.solving_results.loc[:, "num_nodes"].mean(
            )
            self.num_nodes_std = self.solving_results.loc[:, "num_nodes"].std()

        if num_instances:
            self.files = self.files[:num_instances]

        missing_community = [
            file for file in self.files
            if os.path.splitext(file)[0] not in community_map
        ]
        if missing_community:
            raise FileNotFoundError(
                f"Missing community files for {len(missing_community)} samples, "
                f"first missing sample: {missing_community[0]}"
            )
        self.community_files = [
            community_map[os.path.splitext(file)[0]]
            for file in self.files
        ]

    def len(self):
        return len(self.files)

    def get(self, index):
        file = os.path.join(self.data_dir, self.files[index])
        with open(file, "rb") as f:
            data = pickle.load(f)
        community_file = os.path.join(self.community_dir, self.community_files[index])
        community_info = np.load(community_file, allow_pickle=True)
        x_constraints, edge_index, edge_attr, x_variables = data

        if isinstance(community_info[0], np.ndarray):
            community_info = [list(arr) for arr in community_info]

        # add positional encoding for x_constraints
        num_constraints = x_constraints.shape[0]
        cons_pos_encoding = np.arange(num_constraints) / num_constraints
        x_constraints = np.concatenate([x_constraints, cons_pos_encoding[:, None]], axis=1)

        assert x_constraints.shape[1] == 2

        return BipartiteGraph(
            x_constraints=FloatTensor(x_constraints),
            edge_index=LongTensor(edge_index),
            edge_attr=FloatTensor(edge_attr),
            x_variables=FloatTensor(x_variables),
            community_info=community_info,
        )


class TrainSampler(BatchSampler):
    def __init__(self, dataset: InstanceDataset, batch_size: int, repeat_size: int, total_steps: int):
        """
        Sampler for training ACMMILP.

        Args:
            dataset: training dataset
            batch_size: batch size, number of instances in each batch
            repeat_size: number of times to repeat randomly masking constraints of the same instance. We use this to increase the training difficulty
            total_steps: number of total training steps
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.repeat_size = repeat_size
        self.total_steps = total_steps

    def __iter__(self):
        all_batches = torch.randint(
            0, len(self.dataset), (self.total_steps, self.batch_size))
        for i in range(self.total_steps):
            batch = all_batches[i]
            batch = batch.repeat_interleave(self.repeat_size)
            batch = batch.tolist()
            yield batch

    def __len__(self):
        return self.total_steps
