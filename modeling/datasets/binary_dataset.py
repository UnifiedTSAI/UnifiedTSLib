#!/usr/bin/env python
# -*- coding:utf-8 _*-
import json
import os
import numpy as np

from .ts_dataset import TimeSeriesDataset


class BinaryDataset(TimeSeriesDataset):
    meta_file_name = 'meta.json'
    bin_file_name_template = 'data-{}-of-{}.bin'

    def __init__(self, data_path):
        if not self.is_valid_path(data_path):
            raise ValueError(f'Folder {data_path} is not a valid TimeMoE dataset.')

        self.data_path = data_path

        # load meta file
        meta_file_path = os.path.join(data_path, self.meta_file_name)
        try:
            self.meta_info = load_json_file(meta_file_path)
        except Exception as e:
            print(f'Error when loading file {meta_file_path}: {e}')
            raise e

        self.num_sequences = self.meta_info['num_sequences']
        self.dtype = self.meta_info['dtype']
        self.seq_infos = self.meta_info['scales']
        for seq_info in self.seq_infos:
            assert seq_info['length'] == self.seq_infos[0]['length'],"every sequency should have same length, but {} does not.".format(data_path)

        # process the start index for each file
        self.file_start_idxes = []
        s_idx = 0
        for fn, length in sorted(self.meta_info['files'].items(), key=lambda x: int(x[0].split('-')[1])):
            self.file_start_idxes.append(
                (os.path.join(data_path, fn), s_idx, length)
            )
            s_idx += length
        self.num_tokens = s_idx

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, seq_idx):
        seq_info = self.seq_infos[seq_idx]
        read_info_list = self._get_read_infos_by_offset_length(seq_info['offset'], seq_info['length'])
        out = []
        for fn, offset_in_file, length in read_info_list:
            out.append(self._read_sequence_in_file(fn, offset_in_file, length))

        if len(out) == 1:
            sequence = out[0]
        else:
            sequence = np.concatenate(out, axis=0)

        if 'mean' in seq_info and 'std' in seq_info:
            return sequence * seq_info['std'] + seq_info['mean']
        else:
            return sequence

    def get_num_tokens(self):
        return self.num_tokens

    def get_sequence_length_by_idx(self, seq_idx):
        return self.seq_infos[seq_idx]['length']

    def _get_read_infos_by_offset_length(self, offset, length):
        # just use naive search
        binary_read_info_list = []
        end_offset = offset + length
        for fn, start_idx, fn_length in self.file_start_idxes:
            end_idx = start_idx + fn_length
            if start_idx <= offset < end_idx:
                if end_offset <= end_idx:
                    binary_read_info_list.append([fn, offset - start_idx, length])
                    break
                else:
                    binary_read_info_list.append([fn, offset - start_idx, end_idx - offset])
                    length = end_offset - end_idx
                    offset = end_idx
        return binary_read_info_list

    def _read_sequence_in_file(self, fn, offset_in_file, length):
        sentence = np.empty(length, dtype=self.dtype)
        with open(fn, mode='rb', buffering=0) as file_handler:
            file_handler.seek(offset_in_file * sentence.itemsize)
            file_handler.readinto(sentence)
        return sentence

    def get_range(self, start,end, max_channel=None):
        assert start < end,"start must be less than end"
        target_length = end-start
        read_info_list = []

        if max_channel is not None and self.num_sequences>max_channel:
            # Randomly select max_channel indices from 0 to self.num_sequences-1 without replacement, sorted
            selected_indices = np.sort(np.random.choice(self.num_sequences, max_channel, replace=False))
        else:
            # Select all indices from 0 to self.num_sequences-1
            selected_indices = np.arange(self.num_sequences)
        for idx in selected_indices:
            seq_info = self.seq_infos[idx]
            read_info_list.extend(self._get_read_infos_by_offset_length(seq_info['offset']+start, target_length))


        out = self._read_sequences_batch(read_info_list)
        sequence = []
        if len(out) == 1:
            sequence = out
        else:
            accu = []
            for out_item in out:
                if len(out_item)+len(accu) <= target_length:
                    accu.extend(out_item)
                if len(accu)==target_length:
                    sequence.append(accu)
                    accu = []

        if 'mean' in self.seq_infos[0] and 'std' in self.seq_infos[0]:
            for idx in range(sequence):
                sequence[idx] = list(np.array(sequence[idx]) * self.seq_infos[idx]['std'] + self.seq_infos[idx]['mean'])

        return np.array(sequence)

    def _read_sequences_batch(self, plans):
        indexed_plans = list(enumerate(plans))

        from collections import defaultdict
        file_groups = defaultdict(list)
        for idx, (fn, offset, length) in indexed_plans:
            file_groups[fn].append((offset, length, idx))


        results = [None] * len(plans)
        dtype = self.dtype
        itemsize = np.dtype(dtype).itemsize

        for fn in file_groups:
            with open(fn, 'rb', buffering=0) as f:
                for offset, length, orig_idx in file_groups[fn]:
                    f.seek(offset * itemsize)
                    sentence = np.empty(length, dtype=dtype)
                    bytes_read = f.readinto(sentence)
                    if bytes_read != sentence.nbytes:
                        raise IOError(f"Read failed for {fn} at offset {offset}")
                    results[orig_idx] = list(sentence)

        return results

    @staticmethod
    def is_valid_path(data_path):
        if (os.path.exists(data_path)
                and os.path.isdir(data_path)
                and os.path.exists(os.path.join(data_path, 'meta.json'))
        ):
            # Load meta.json and verify that 'inverted' field is not True
            meta_file_path = os.path.join(data_path, 'meta.json')
            try:
                meta_info = load_json_file(meta_file_path)
                if meta_info['inverted'] is True:
                    return False
            except Exception as e:
                pass
            for sub in os.listdir(data_path):
                if os.path.isfile(os.path.join(data_path, sub)) and sub.endswith('.bin'):
                    return True
        return False

class InvertedBinaryDataset(TimeSeriesDataset):
    meta_file_name = 'meta.json'
    bin_file_name_template = 'data-{}-of-{}.bin'

    def __init__(self, data_path):
        if not self.is_valid_path(data_path):
            raise ValueError(f'Folder {data_path} is not a valid TimeMoE dataset.')

        self.data_path = data_path

        # load meta file
        meta_file_path = os.path.join(data_path, self.meta_file_name)
        try:
            self.meta_info = load_json_file(meta_file_path)
        except Exception as e:
            print(f'Error when loading file {meta_file_path}: {e}')
            raise e

        self.num_sequences = self.meta_info['num_sequences']
        self.dtype = self.meta_info['dtype']
        self.time_point_infos = self.meta_info['scales']
        for time_point_info in self.time_point_infos:
            assert time_point_info['length'] == self.time_point_infos[0]['length'], "every sequency should have same length, but {} does not.".format(data_path)

        # process the start index for each file
        self.file_start_idxes = []
        s_idx = 0
        for fn, length in sorted(self.meta_info['files'].items(), key=lambda x: int(x[0].split('-')[1])):
            self.file_start_idxes.append(
                (os.path.join(data_path, fn), s_idx, length)
            )
            s_idx += length
        self.num_tokens = s_idx

    def __len__(self):
        return self.num_sequences

    def get_num_tokens(self):
        return self.num_tokens

    def get_sequence_length_by_idx(self, seq_idx):
        return len(self.time_point_infos)

    def _get_read_infos_by_offset_length(self, offset, length):
        # just use naive search
        binary_read_info_list = []
        end_offset = offset + length
        for fn, start_idx, fn_length in self.file_start_idxes:
            end_idx = start_idx + fn_length
            if start_idx <= offset < end_idx:
                if end_offset <= end_idx:
                    binary_read_info_list.append([fn, offset - start_idx, length])
                    break
                else:
                    binary_read_info_list.append([fn, offset - start_idx, end_idx - offset])
                    length = end_offset - end_idx
                    offset = end_idx
        return binary_read_info_list

    def _read_sequence_in_file(self, fn, offset_in_file, length):
        sentence = np.empty(length, dtype=self.dtype)
        with open(fn, mode='rb', buffering=0) as file_handler:
            file_handler.seek(offset_in_file * sentence.itemsize)
            file_handler.readinto(sentence)
        return sentence

    def get_range(self, start,end, max_channel=None):
        assert start < end,"start must be less than end"
        target_length = end-start

        if max_channel is not None and self.num_sequences>max_channel:
            # Randomly select max_channel indices from 0 to self.num_sequences-1 without replacement, sorted
            selected_indices = np.sort(np.random.choice(self.num_sequences, max_channel, replace=False))
        else:
            # Select all indices from 0 to self.num_sequences-1
            selected_indices = np.arange(self.num_sequences)

        read_info_list = self._get_read_infos_by_offset_length(self.time_point_infos[start]['offset'], target_length*self.num_sequences)

        out = self._read_sequences_batch(read_info_list)
        time_points = []

        if len(out) == 1:
            time_points = out[0]
        else:
            time_points = out[0]
            for idx in range(1,len(out)):
                out_item = out[idx]
                if len(out_item)+len(time_points) <= target_length*self.num_sequences:
                    time_points.extend(out_item)

        assert len(time_points)==target_length*self.num_sequences,"the length of time_points is not equal to target_length*self.num_sequences"

        data = np.array(time_points).reshape(target_length, self.num_sequences) # shape (T,N)
        data = data.T[selected_indices,:]

        return data

    def _read_sequences_batch(self, plans):
        indexed_plans = list(enumerate(plans))

        from collections import defaultdict
        file_groups = defaultdict(list)
        for idx, (fn, offset, length) in indexed_plans:
            file_groups[fn].append((offset, length, idx))


        results = [None] * len(plans)
        dtype = self.dtype
        itemsize = np.dtype(dtype).itemsize

        for fn in file_groups:
            with open(fn, 'rb', buffering=0) as f:
                for offset, length, orig_idx in file_groups[fn]:
                    f.seek(offset * itemsize)
                    sentence = np.empty(length, dtype=dtype)
                    bytes_read = f.readinto(sentence)
                    if bytes_read != sentence.nbytes:
                        raise IOError(f"Read failed for {fn} at offset {offset}")
                    results[orig_idx] = list(sentence)

        return results

    @staticmethod
    def is_valid_path(data_path):
        if (os.path.exists(data_path)
                and os.path.isdir(data_path)
                and os.path.exists(os.path.join(data_path, 'meta.json'))
        ):
            # Load meta.json and verify that 'inverted' field is True
            meta_file_path = os.path.join(data_path, 'meta.json')
            try:
                meta_info = load_json_file(meta_file_path)
                if meta_info['inverted'] is False:
                    return False
            except Exception as e:
                print(f'Error when loading file {meta_file_path}: {e}')
                return False
            for sub in os.listdir(data_path):
                if os.path.isfile(os.path.join(data_path, sub)) and sub.endswith('.bin'):
                    return True
        return False


def load_json_file(fn):
    with open(fn, encoding='utf-8') as file:
        data = json.load(file)
        return data


def save_json_file(obj, fn):
    with open(fn, 'w', encoding='utf-8') as file:
        json.dump(obj, file)

