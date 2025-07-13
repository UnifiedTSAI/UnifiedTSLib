#!/usr/bin/env python
# -*- coding:utf-8 _*-
import json
import os
import argparse
import numpy as np
import logging
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm

from transformers import AutoModelForCausalLM


from time_mixer_pp.datasets.benchmark_dataset import ChannelEvalDataset


def setup_nccl(rank, world_size, master_addr='127.0.0.1', master_port=9899):
    dist.init_process_group("nccl", init_method='tcp://{}:{}'.format(master_addr, master_port), rank=rank,
                            world_size=world_size)


def count_num_tensor_elements(tensor):
    n = 1
    for s in tensor.shape:
        n = n * s
    return n


# ------------------ Metrics ------------------
class SumEvalMetric:
    def __init__(self, name, init_val: float = 0.0):
        self.name = name
        self.value = init_val

    def push(self, preds, labels, **kwargs):
        self.value += self._calculate(preds, labels, **kwargs)

    def _calculate(self, preds, labels, **kwargs):
        pass


class MSEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum((preds - labels) ** 2)


class MAEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum(torch.abs(preds - labels))


class UnifiedTS:
    def __init__(self, model_path, device, context_length, prediction_length, **kwargs):
        from time_mixer_pp.models.modeling_UnifiedTS import UnifiedTSForPrediction,UnifiedTSConfig
        self.model = UnifiedTSForPrediction.from_pretrained(
            model_path,  
            # device_map="auto",
            torch_dtype="auto"
        )
        self.device = device
        self.model = self.model.to(self.device) 
        self.prediction_length = prediction_length
        self.model.eval()

    def predict(self, batch):
       
        inputs = batch['inputs'].to(self.device)
        labels = batch['labels'].to(self.device)
        outputs = self.model(input_ids=inputs, labels=labels)

        preds = outputs.logits if isinstance(outputs.logits, list) else [outputs.logits]
        labels_slices = [labels[:, :pl] for pl in self.model.config.pred_len]
        return preds, labels_slices


def evaluate(args):
    batch_size = args.batch_size
    context_length = args.context_length
    prediction_lengths = [96, 192, 336, 720]  
    

    master_addr = os.getenv('MASTER_ADDR', '127.0.0.1')
    master_port = os.getenv('MASTER_PORT', 9899)
    world_size = int(os.getenv('WORLD_SIZE') or 1)
    rank = int(os.getenv('RANK') or 0)
    local_rank = int(os.getenv('LOCAL_RANK') or 0)
    
   
    if torch.cuda.is_available():
        try:
            setup_nccl(rank, world_size, master_addr, master_port)
            device = f"cuda:{local_rank}"
            is_dist = True
        except Exception as e:
            print(f'Setup nccl fail: {e}, fallback to cpu')
            device = 'cpu'
            is_dist = False
    else:
        device = 'cpu'
        is_dist = False


    model = UnifiedTS(
        args.model_path,
        device,
        context_length=context_length,
        prediction_length=prediction_lengths,  
        channel_mixing=args.channel_mixing
    )

  
    global_metrics = {pl: {'mse': 0.0, 'mae': 0.0, 'count': 0} for pl in prediction_lengths}

    for pl in prediction_lengths:
      
        dataset = ChannelEvalDataset(
            args.data,
            context_length=context_length,
            prediction_length=pl
        )

        sampler = DistributedSampler(dataset, shuffle=False) if is_dist else None
        test_dl = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
            drop_last=False
        )

        local_mse = 0.0
        local_mae = 0.0
        local_count = 0

        with torch.no_grad():
            for batch in tqdm(test_dl, desc=f"Testing pl={pl}"):
            
                all_preds, all_labels = model.predict(batch)
               
                for pred, label, pred_len in zip(all_preds, all_labels, model.model.config.pred_len):
                    if pred_len == pl:
                        pred = pred.squeeze(-1)
                        label = label.squeeze(-1)
                        
                        
                        local_mse += torch.sum((pred - label) ** 2).item()
                        local_mae += torch.sum(torch.abs(pred - label)).item()
                        local_count += pred.numel()
                        break


        if is_dist:
            local_stats = torch.tensor([local_mse, local_mae, local_count]).to(device)
            global_stats = [torch.zeros_like(local_stats) for _ in range(world_size)]
            dist.all_gather(global_stats, local_stats)
            
            global_mse = sum([s[0].item() for s in global_stats])
            global_mae = sum([s[1].item() for s in global_stats])
            global_count = sum([s[2].item() for s in global_stats])
        else:
            global_mse, global_mae, global_count = local_mse, local_mae, local_count

     
        if global_count > 0:
            global_metrics[pl]['mse'] = global_mse / global_count
            global_metrics[pl]['mae'] = global_mae / global_count
            global_metrics[pl]['count'] = global_count

 
    if rank == 0:
        result = {
            'model': args.model_path,
            'data': args.data,
            'context_length': context_length,
            'metrics': {}
        }
        
        for pl in prediction_lengths:
            if global_metrics[pl]['count'] > 0:
                result['metrics'].update({
                    f'mse_{pl}': global_metrics[pl]['mse'],
                    f'mae_{pl}': global_metrics[pl]['mae']
                })
        
        logging.info(json.dumps(result, indent=2))
        print("\nFinal Results:")
        for pl in prediction_lengths:
            print(f"PredLen {pl}: MSE={global_metrics[pl]['mse']:.4f}, MAE={global_metrics[pl]['mae']:.4f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Timemixerpp Evaluate')
    parser.add_argument(
        '--model_path',
        type=str,
        default='logs/timemixerpp',
        help='Model path'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='timemixerpp',
        help='Model path'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Benchmark data path'
    )

    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=32,
        help='Batch size of evaluation'
    )
    parser.add_argument(
        '--context_length', '-c',
        type=int,
        default=96,
        help='Context length'
    )
    parser.add_argument(
        '--channel_mixing',
        type=bool,
        default=False
    )
    args = parser.parse_args()
    evaluate(args)
