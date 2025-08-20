from __future__ import print_function
import copy
import os
import time

import torch
import yaml
from gxl_ai_utils.utils import utils_file
from torch.utils.data import DataLoader
from cn2an import an2cn
from wenet.dataset.dataset import Dataset
from wenet.utils.config import override_config
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.common import TORCH_NPU_AVAILABLE  # noqa just ensure to check torch-npu
import logging
import  sys

import torch

from common_utils.utils4infer import convert_numbers_in_string, load_model_and_tokenizer, get_test_conf, do_format_shard_manifest4one

from patches import modelling_qwen2_infer_gpu  # 打patch
from tts.cosyvoice.utils.file_utils import load_wav

from cn2an import an2cn
import re

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test_data_path', type=str, help='config path')
parser.add_argument('--infer_res_path', type=str,  help='data type')
parser.add_argument('--gpu_id', type=int, help='gpu id')
parser.add_argument('--data_type', type=str, help='task type')

args = parser.parse_args()



config_path = './conf/ct_config.yaml'
test_data_path = args.test_data_path
infer_res_path = args.infer_res_path
gpu_id = args.gpu_id
data_type = args.data_type
print(f'test_data_path: {test_data_path}, infer_res_path: {infer_res_path}, gpu_id: {gpu_id}, data_type: {data_type}')

if data_type == "shards":
    test_data_path = do_format_shard_manifest4one(test_data_path)
dtype = torch.float32
# export CUDA_VISIBLE_DEVICES=6
device = torch.device(f'cuda:{gpu_id}')
configs, test_conf = get_test_conf(config_path)

checkpoint_path = "/home/A02_tmpdata2/ckpt/osum_chat_new_start_0810/epoch3_s2t_t2s_t2t_s2s_hq_language_think/step_27499.pt"
config_path = "conf/ct_config.yaml"
cosyvoice_model_path = "/home/A02_tmpdata2/ckpt/cosyvoice1-300M/CosyVoice-300M-25Hz"

prompt_wav_path = "./tts/assert/prompt.wav"
prompt_audio_cache = {"拟人": load_wav(prompt_wav_path, 22050)}
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
model, tokenizer = load_model_and_tokenizer(checkpoint_path, config_path, device)
model.eval()

def do_asr(model, feat, feat_lens):  # 增加 model 参数
    print(f'feat shape: {feat.shape}, feat_lens: {feat_lens}')
    start_time = time.time()
    res_text = model.generate(wavs=feat, wavs_len=feat_lens, prompt = "将这段音频的语音内容详细记录为文字稿。", cache_implementation="static")[0]
    end_time = time.time()
    print(f"S2T4Chat think 推理消耗时间: {end_time - start_time:.2f} 秒")
    return res_text

test_dataset = Dataset(data_type,
                       test_data_path,
                       tokenizer,
                       test_conf,
                       partition=False)

test_data_loader = DataLoader(test_dataset,
                              batch_size=None,
                              num_workers=5)

infer_dict = {}
with torch.no_grad():
    # logging.info(f'utt_num: {utt_num}')
    for batch_idx, batch in enumerate(test_data_loader):
        keys = batch["keys"]
        feats = batch["feats"].to(device).to(torch.bfloat16)
        feats_lengths = batch["feats_lengths"].to(device)
        txts = batch["txts"]
        batch_size = feats.size(0)
        res_text = do_asr(model, feats, feats_lengths)
        true_txt = txts[0]
        res_text = convert_numbers_in_string(res_text)
        true_txt = convert_numbers_in_string(true_txt)
        key = keys[0]
        print(f'{key}\t {res_text}\t {true_txt}')
        infer_dict[key] = res_text
        if batch_idx % 100 == 0:
            utils_file.write_dict_to_scp(infer_dict, infer_res_path)
    utils_file.write_dict_to_scp(infer_dict, infer_res_path)




