<div align="center">
<img src="./pic/logo.svg" height = "200" alt="" align=center />

# UnifiedTSLib: A Unified Time Series Foundation Model Training Architecture
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2410.16032v5&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2410.16032v5)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/aeiiou/TimeMixerPP_50M)
[![License: MIT](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tutorial](https://img.shields.io/badge/Tutorial-Finetune_%26_Eval-important)](https://github.com/kwuking/UnifiedTSLib/blob/main/tutorial/finetune_and_eval.ipynb)
</div>
UnifiedTSLib is a collection of popular time series analysis models implemented in the Hugging Face Transformers style. This library provides easy-to-use, standardized interfaces for training, fine-tuning, and evaluating various state-of-the-art time series forecasting models, making it convenient to apply and benchmark them on your own datasets.

### 🌟Key Features
- Implementation of [TimeMixer++](https://arxiv.org/pdf/2410.16032v5), [TimeMixer](https://arxiv.org/pdf/2405.14616), [iTransformer](https://arxiv.org/pdf/2310.06625), [TimesNet](https://arxiv.org/pdf/2210.02186), and [Autoformer](https://arxiv.org/pdf/2106.13008) (with more being added continuously).
- Supports data parallel training with models saved in Hugging Face format🤗.
- Features a channel-mixing time series pre-training framework that balances batch size and channel count across datasets to enhance computational stability and reduce bandwidth waste caused by padding.
- Inherits [Time-MoE](https://github.com/Time-MoE/Time-MoE)'s disk-based single-sequence reading capability to avoid memory overflow during large-scale data training (300B+ time points), and accelerates disk reading of all sequences within a specified range in Channel Mixing mode.

### 🚀 Usage

#### 1. Install Dependencies
Make sure you have Python 3.8+ installed. Install the required packages with:
```
pip install -r requirements.txt
```
#### 2. Prepare Data
Prepare your time series dataset and place it in the appropriate directory (e.g., `data/train/`, `data/val/`). Supported formats include `.jsonl`, `.csv`, or `.bin`, etc.
#### 3. Train (Pre-train or Fine-tune) a Model
You can either pre-train a model from scratch or fine-tune a model based on pretrained weights.
- Fine-tuning: Use a pretrained model as the starting point (`-m` specifies the model path).
- Pre-training: Train a model from scratch by adding the `--from_scratch` flag and omitting the `-m` argument.
Fine-tune example:
```
python torch_dist_run.py main.py \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --channel_mixing True \
    --inner_batch_ratio 1 \
    --model_name timemixerpp \
    -o logs/timemixerpp_traffic_finetune \
    -d data/train/data_electricity_train/ \
    -m /tiger/UnifiedTSLib/logs/timemixerpp_1 \
    --val_data_path data/val/data_electricity_validation/
```
Pre-train example:
```
python torch_dist_run.py main.py \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --channel_mixing True \
    --inner_batch_ratio 1 \
    --model_name timemixerpp \
    --from_scratch \
    -o logs/timemixerpp_pretrain \
    -d data/train/ \
```
Parameter explanations:
1. `--channel_mixing True`: Enables channel mixing strategy during training.
2. `--micro_batch_size 1`: Sets the micro batch size per GPU to 1. For datasets with a large number of channels, it is recommended to use 1.
3. `--inner_batch_ratio 1`: Sets the inner batch size for the dataset with the largest number of channels. Recommended value is 1.
4. `--model_name Timemixerpp`: Specifies the model architecture to use.
5. `--from_scratch`: (Optional) If set, the model will be trained from scratch without loading pretrained weights.
6. `-m`: (Optional) Path to the pretrained model directory. Omit this argument when pre-training from scratch.
#### 4. Evaluate a Model
You can evaluate a trained model using `eval_model.py` as follows:
```
python eval_model.py \
    -d datasets_pretrain/test/data_etth1_test.jsonl \
    --channel_mixing True \
    --batch_size 512 \
    --model_path logs/UnifiedTS/Timemixerpp \
    --model_name timemixerpp
```
Parameter explanations:
- `-d`: Path to the evaluation dataset.
- `--channel_mixing True`: Enables channel mixing during evaluation.
- `--batch_size`: Batch size for evaluation.
- `--model_path`: Path to the trained model directory.
- `--model_name`: Name of the model architecture.


## 📝 Citation

If you find this useful for your research, please consider citing the associated [paper](https://arxiv.org/abs/2410.16032):

```bibtex
@inproceedings{Wang2025TimeMixer++,
  title={Timemixer++: A general time series pattern machine for universal predictive analysis},
  author={Wang, Shiyu and Li, Jiawei and Shi, Xiaoming and Ye, Zhou and Mo, Baichuan and Lin, Wenze and Ju, Shengtong and Chu, Zhixuan and Jin, Ming},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}

@inproceedings{shi2024timemoe,
  title={Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts},
  author={Xiaoming Shi and Shiyu Wang and Yuqi Nie and Dianqi Li and Zhou Ye and Qingsong Wen and Ming Jin},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}

@inproceedings{wang2023timemixer,
  title={TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting},
  author={Wang, Shiyu and Wu, Haixu and Shi, Xiaoming and Hu, Tengge and Luo, Huakun and Ma, Lintao and Zhang, James Y and ZHOU, JUN},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

## 📃 License

This project is licensed under the Apache-2.0 License.
