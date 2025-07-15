#!/usr/bin/env python
# -*- coding:utf-8 _*-
import json
import os
import pickle
import gzip
import yaml
import numpy as np

from .ts_dataset import TimeSeriesDataset


class GeneralDataset(TimeSeriesDataset):
    def __init__(self, data_path, normalization_method=None):
        self.data = read_file_by_extension(data_path)
        if normalization_method is not None:
            for idx in range(self.data):
                if isinstance(self.data[idx],dict):
                    self.data[idx]["sequence"] = normalization_method(self.data[idx]["sequence"])
                else:
                    self.data[idx] = normalization_method(self.data[idx])
        self.num_tokens = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, seq_idx):
        seq = self.data[seq_idx]
        if isinstance(seq, dict):
            seq = seq['sequence']
        return seq

    def get_range(self,start,end,max_channel=None):
        assert start < end, "start must be less than end"
        if max_channel is None or len(self.data) <= max_channel:
            data =  [(seq['sequence'] if isinstance(seq, dict) else seq)[start:end]
                for seq in self.data]
        elif max_channel is not None and len(self.data) > max_channel:
            # Randomly select max_channel indices from 0 to self.num_sequences-1 without replacement, sorted
            selected_indices = np.sort(np.random.choice(len(self.data), max_channel, replace=False))
            data =  [(seq['sequence'] if isinstance(seq, dict) else seq)[start:end]
                    for seq in self.data[selected_indices]]
        else:
            raise ValueError(f"Invalid max_channel value: {max_channel}")

        return np.array(data)

    def get_num_tokens(self):
        if self.num_tokens is None:
            self.num_tokens = sum([len(seq) for seq in self])
        return self.num_tokens

    def get_sequence_length_by_idx(self, seq_idx):
        seq = self[seq_idx]
        return len(seq)

    @staticmethod
    def is_valid_path(data_path):
        if os.path.exists(data_path) and os.path.isfile(data_path):
            parts = data_path.split('.')
            if len(parts) == 0:
                return False
            suffix = parts[-1]
            if suffix in ('json', 'jsonl', 'npy', 'npy.gz', 'pkl'):
                return True
            else:
                return False
        else:
            return False


def read_file_by_extension(fn):
    if fn.endswith('.json'):
        with open(fn, encoding='utf-8') as file:
            data = json.load(file)
    elif fn.endswith('.jsonl'):
        data = read_jsonl_to_list(fn)
    elif fn.endswith('.yaml'):
        data = load_yaml_file(fn)
    elif fn.endswith('.npy'):
        data = np.load(fn, allow_pickle=True)
    elif fn.endswith('.npz'):
        data = np.load(fn, allow_pickle=True)
    elif fn.endswith('.npy.gz'):
        with gzip.GzipFile(fn, 'r') as file:
            data = np.load(file, allow_pickle=True)
    elif fn.endswith('.pkl') or fn.endswith('.pickle'):
        data = load_pkl_obj(fn)
    else:
        raise RuntimeError(f'Unknown file extension: {fn}')
    return data


def read_jsonl_to_list(jsonl_fn):
    with open(jsonl_fn, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file.readlines()]


def load_yaml_file(fn):
    if isinstance(fn, str):
        with open(fn, 'r', encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config
    else:
        return fn


def load_pkl_obj(fn):
    out_list = []
    with open(fn, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                out_list.append(data)
            except EOFError:
                break
    if len(out_list) == 0:
        return None
    elif len(out_list) == 1:
        return out_list[0]
    else:
        return out_list




