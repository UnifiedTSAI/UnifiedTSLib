#!/usr/bin/env python
# -*- coding:utf-8 _*-
import os
import numpy as np
import pandas as pd

from .ts_dataset import TimeSeriesDataset
from .general_dataset import GeneralDataset
from .binary_dataset import BinaryDataset, InvertedBinaryDataset


class ChannelMixingDataset(TimeSeriesDataset):

    def __init__(self, configs):
        self.data_folder = configs["data_path"]
        normalization_method = configs["normalization_method"]
        self.datasets = []
        self.splited_datasets = []
        self.num_tokens = None
        self.max_channel = configs['max_channel']

        self.context_length = configs["context_length"]
        self.prediction_length = configs["prediction_length"]
        self.window_size = self.context_length + self.prediction_length
        self.sliding_steps = configs["sliding_steps"]
        self.inner_batch_ratio = configs["inner_batch_ratio"]

        if normalization_method is None:
            self.normalization_method = None
        elif isinstance(normalization_method, str):
            if normalization_method.lower() == 'max':
                self.normalization_method = max_scaler
            elif normalization_method.lower() == 'zero':
                self.normalization_method = zero_scaler
            else:
                raise ValueError(f'Unknown normalization method: {normalization_method}')
        else:
            self.normalization_method = normalization_method
        if self.data_folder.lower().endswith('.xlsx'):
            df = pd.read_excel('sundial_data_summary_filtered_100k.xlsx')
            folders = df['folder'].tolist()
            for folder in folders:
                if BinaryDataset.is_valid_path(folder):
                    ds = BinaryDataset(folder)
                    if len(ds) > 0:
                        self.datasets.append(ds)
                elif GeneralDataset.is_valid_path(folder):
                    ds = GeneralDataset(folder)
                    if len(ds) > 0:
                        self.datasets.append(ds)
        if BinaryDataset.is_valid_path(self.data_folder):
            ds = BinaryDataset(self.data_folder)
            if len(ds) > 0:
                self.datasets.append(ds)
        elif InvertedBinaryDataset.is_valid_path(self.data_folder):
            ds = InvertedBinaryDataset(self.data_folder)
            if len(ds) > 0:
                self.datasets.append(ds)
        elif GeneralDataset.is_valid_path(self.data_folder):
            ds = GeneralDataset(self.data_folder, normalization_method=self.normalization_method)
            if len(ds) > 0:
                self.datasets.append(ds)
        else:
            # walk through the data_folder
            for root, dirs, files in os.walk(self.data_folder):
                for file in files:
                    fn_path = os.path.join(root, file)
                    if file != BinaryDataset.meta_file_name and GeneralDataset.is_valid_path(fn_path):
                        ds = GeneralDataset(fn_path)
                        if len(ds) > 0:
                            self.datasets.append(ds)
                for sub_folder in dirs:
                    folder_path = os.path.join(root, sub_folder)
                    if BinaryDataset.is_valid_path(folder_path):
                        ds = BinaryDataset(folder_path)
                        if len(ds) > 0:
                            self.datasets.append(ds)
                    elif InvertedBinaryDataset.is_valid_path(folder_path):
                        ds = InvertedBinaryDataset(folder_path)
                        if len(ds) > 0:
                            self.datasets.append(ds)

        self.cumsum_batches = [0]
        self.dataset_length = []
        self.dataset_inner_batchsize = []

        max_channel = 0
        for ds in self.datasets:
            max_channel = max(max_channel, min(self.max_channel, len(ds)))
            self.dataset_length.append(ds.get_sequence_length_by_idx(0))
        self.max_channel = max_channel

        for idx, ds in enumerate(self.datasets):
            inner_batch_size = max_channel // min(self.max_channel, len(ds)) * self.inner_batch_ratio

            if (self.dataset_length[idx] - self.window_size + self.sliding_steps) > (
                    self.sliding_steps * inner_batch_size):
                num_inner_batches = (self.dataset_length[idx] - self.window_size + self.sliding_steps) // (
                            self.sliding_steps * inner_batch_size)
            else:
                num_inner_batches = 1
                inner_batch_size = (self.dataset_length[
                                        idx] - self.window_size + self.sliding_steps) // self.sliding_steps
            self.dataset_inner_batchsize.append(inner_batch_size)
            print(num_inner_batches)
            self.cumsum_batches.append(
                self.cumsum_batches[-1] + num_inner_batches
            )

        self.item_length = max_channel * self.window_size

        self.num_batches = self.cumsum_batches[-1]

    def __len__(self):
        return self.num_batches

    def __getitem__(self, seq_idx):
        if seq_idx >= self.cumsum_batches[-1]:
            raise ValueError(f'Index out of the dataset length: {seq_idx} >= {self.cumsum_batches[-1]}')
        elif seq_idx < 0:
            raise ValueError(f'Index out of the dataset length: {seq_idx} < 0')

        dataset_idx = binary_search(self.cumsum_batches, seq_idx)

        dataset_offset = seq_idx - self.cumsum_batches[dataset_idx]
        current_num_channel = min(self.max_channel, len(self.datasets[dataset_idx]))

        inner_batch_size = self.dataset_inner_batchsize[dataset_idx]

        start_idx = dataset_offset * self.sliding_steps
        end_idx = start_idx + self.window_size + (inner_batch_size - 1) * self.sliding_steps

        data = self.datasets[dataset_idx].get_range(start_idx, end_idx, max_channel=self.max_channel)

        window_data_list = []

        for i in range(inner_batch_size):
            window_data_list.append(data[:, i * self.sliding_steps:self.window_size + i * self.sliding_steps])

        flatten_seq = np.concatenate(window_data_list).flatten().astype(np.float32)

        n_pad = self.item_length - inner_batch_size * self.window_size * current_num_channel
        if n_pad > 0:
            flatten_seq = np.pad(flatten_seq, (0, n_pad), 'constant', constant_values=0)

        return {
            'input_ids': flatten_seq,
            'prediction_length': self.prediction_length,
            'context_length': self.context_length,
            'inner_batchsize': inner_batch_size,
            'num_channel': current_num_channel,
            'dataset_idx': dataset_idx
        }

    def get_num_tokens(self):
        if self.num_tokens is None:
            self.num_tokens = sum([ds.get_num_tokens() for ds in self.datasets])

        return self.num_tokens


def zero_scaler(seq):
    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)
    origin_dtype = seq.dtype
    # std_val = seq.std(dtype=np.float64)
    std_val = seq.std()
    if std_val == 0:
        normed_seq = seq
    else:
        # mean_val = seq.mean(dtype=np.float64)
        mean_val = seq.mean()
        normed_seq = (seq - mean_val) / std_val

    return normed_seq.astype(origin_dtype)


def max_scaler(seq):
    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)
    origin_dtype = seq.dtype
    # max_val = np.abs(seq).max(dtype=np.float64)
    max_val = np.abs(seq).max()
    if max_val == 0:
        normed_seq = seq
    else:
        normed_seq = seq / max_val

    return normed_seq.astype(origin_dtype)


def binary_search(sorted_list, value):
    low = 0
    high = len(sorted_list) - 1
    best_index = -1

    while low <= high:
        mid = (low + high) // 2
        if sorted_list[mid] <= value:
            best_index = mid
            low = mid + 1
        else:
            high = mid - 1

    return best_index

