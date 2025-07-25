#!/usr/bin/env python
# -*- coding:utf-8 _*-
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from Modeling.datasets.general_dataset import GeneralDataset
from Modeling.datasets.channel_mixing_dataset import BinaryDataset
from Modeling.utils.log_util import log_in_local_rank_0


class BenchmarkEvalDataset(Dataset):

    def __init__(self, csv_path, context_length: int, prediction_length: int):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length

        df = pd.read_csv(csv_path)

        base_name = os.path.basename(csv_path).lower()
        if 'etth' in base_name:
            border1s = [0, 12 * 30 * 24 - context_length, 12 * 30 * 24 + 4 * 30 * 24 - context_length]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif 'ettm' in base_name:
            border1s = [0, 12 * 30 * 24 * 4 - context_length, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - context_length]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            num_train = int(len(df) * 0.7)
            num_test = int(len(df) * 0.2)
            num_vali = len(df) - num_train - num_test
            border1s = [0, num_train - context_length, len(df) - num_test - context_length]
            border2s = [num_train, num_train + num_vali, len(df)]

        start_dt = df.iloc[border1s[2]]['date']
        eval_start_dt = df.iloc[border1s[2] + context_length]['date']
        end_dt = df.iloc[border2s[2] - 1]['date']
        log_in_local_rank_0(f'>>> Split test data from {start_dt} to {end_dt}, '
                            f'and evaluation start date is: {eval_start_dt}')

        cols = df.columns[1:]
        df_values = df[cols].values

        train_data = df_values[border1s[0]:border2s[0]]
        test_data = df_values[border1s[2]:border2s[2]]

        # scaling
        scaler = StandardScaler()
        scaler.fit(train_data)
        scaled_test_data = scaler.transform(test_data)

        # assignment
        self.hf_dataset = scaled_test_data.transpose(1, 0)
        self.num_sequences = len(self.hf_dataset)
        # 1 for the label
        self.window_length = self.context_length + self.prediction_length

        self.sub_seq_indexes = []
        for seq_idx, seq in enumerate(self.hf_dataset):
            n_points = len(seq)
            if n_points < self.window_length:
                continue
            for offset_idx in range(self.window_length, n_points):
                self.sub_seq_indexes.append((seq_idx, offset_idx))

    def __len__(self):
        return len(self.sub_seq_indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        seq_i, offset_i = self.sub_seq_indexes[idx]
        seq = self.hf_dataset[seq_i]

        window_seq = np.array(seq[offset_i - self.window_length: offset_i], dtype=np.float32)

        return {
            'inputs': np.array(window_seq[: self.context_length], dtype=np.float32),
            'labels': np.array(window_seq[-self.prediction_length:], dtype=np.float32),
        }


class GeneralEvalDataset(Dataset):

    def __init__(self, data_path, context_length: int, prediction_length: int, onfly_norm: bool = False):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.onfly_norm = onfly_norm
        self.window_length = self.context_length + self.prediction_length
        self.dataset = GeneralDataset(data_path)

        self.sub_seq_indexes = []
        for seq_idx, seq in enumerate(self.dataset):
            n_points = len(seq)
            if n_points < self.window_length:
                continue
            for offset_idx in range(self.window_length, n_points):
                self.sub_seq_indexes.append((seq_idx, offset_idx))

    def __len__(self):
        return len(self.sub_seq_indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        seq_i, offset_i = self.sub_seq_indexes[idx]
        seq = self.dataset[seq_i]

        window_seq = np.array(seq[offset_i - self.window_length: offset_i], dtype=np.float32)

        inputs = np.array(window_seq[: self.context_length], dtype=np.float32)
        labels = np.array(window_seq[-self.prediction_length:], dtype=np.float32)

        if self.onfly_norm:
            mean_ = inputs.mean()
            std_ = inputs.std()
            if std_ == 0:
                std_ = 1
            inputs = (inputs - mean_) / std_
            labels = (labels - mean_) / std_

        return {
            'inputs': np.array(window_seq[: self.context_length], dtype=np.float32),
            'labels': np.array(window_seq[-self.prediction_length:], dtype=np.float32),
        }

class ChannelEvalDataset(Dataset):

    def __init__(self, data_path, context_length: int, prediction_length: int, onfly_norm: bool = False):
        super().__init__()
        self.data_path = data_path
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.onfly_norm = onfly_norm
        self.window_length = self.context_length + self.prediction_length
        if BinaryDataset.is_valid_path(self.data_path):
            self.dataset = BinaryDataset(data_path)
        elif GeneralDataset.is_valid_path(self.data_path):
            self.dataset = GeneralDataset(data_path)
        else:
            raise ValueError('Invalid dataset path: {}, a single dataset should be specified not a folder. '.format(data_path))

    def __len__(self):
        return self.dataset.get_sequence_length_by_idx(0)-self.window_length+1

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        data = self.dataset.get_range(idx,idx+self.window_length)
        inputs = data[:,:self.context_length]
        labels = data[:,self.context_length:]
        if self.onfly_norm:
            for idx in range(inputs.size(0)):
                mean_ = inputs[idx].mean()
                std_ = inputs[idx].std()
                if std_ == 0:
                    std_ = 1
                inputs[idx] = (inputs[idx] - mean_) / std_
                labels[idx] = (labels[idx] - mean_) / std_
        return {
            'inputs': inputs.T,# (T,N)
            'labels': labels.T,
        }

