import copy
import os
import random
import re

import yaml
from cn2an import an2cn
from gxl_ai_utils.utils import utils_file
from wenet.utils.init_tokenizer import init_tokenizer
from gxl_ai_utils.config.gxl_config import GxlNode
from wenet.utils.init_model import init_model
import logging
import librosa
import torch
import torchaudio



def load_model_and_tokenizer(checkpoint_path, config_path, device:torch.device=torch.device('cuda')):
    """
    封装了加载模型和分词器的逻辑
    Args:
        checkpoint_path (str): 模型权重文件路径
        config_path (str): 模型配置文件路径
        device (torch.device): 加载模型的设备
    Returns:
        model: 加载好的模型
        tokenizer: 加载好的分词器
    """
    print(f"正在从以下路径加载模型: {checkpoint_path}")
    args = GxlNode({"checkpoint": checkpoint_path})
    configs = utils_file.load_dict_from_yaml(config_path)
    model, configs = init_model(args, configs)
    model = model.to(device).to(torch.bfloat16)
    model.eval()  # 设置为评估模式
    tokenizer = init_tokenizer(configs)
    print(f"模型 {checkpoint_path} 加载完成并移动到 {device}")
    return model, tokenizer

def token_list2wav(token_list, prompt_speech, wav_path, cosyvoice):
    token_list = [int(i) for i in token_list]
    j = cosyvoice.inference_zero_shot_gz_22k(
        '收到好友从远方寄来的生日礼物。',
        '希望你以后能够做的比我还好呦。', prompt_speech, stream=False, token_list=token_list)
    utils_file.makedir_for_file(wav_path)
    torchaudio.save(wav_path, j['tts_speech'],cosyvoice.sample_rate)
    print(f'语音合成完成，保存到 {wav_path}')
    return wav_path

def do_resample(input_wav_path):
    """..."""
    waveform, sample_rate = torchaudio.load(input_wav_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)
    return waveform, 16000


def get_feat_from_wav_path(input_wav_path, device:torch.device=torch.device('cuda')):
    """..."""
    waveform, sample_rate = do_resample(input_wav_path)
    waveform = waveform.squeeze(0)
    window = torch.hann_window(400)
    stft = torch.stft(waveform, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = torch.from_numpy(librosa.filters.mel(sr=sample_rate, n_fft=400, n_mels=80))
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    feat = log_spec.transpose(0, 1)
    feat_lens = torch.tensor([feat.shape[0]], dtype=torch.int64).to(device)
    feat = feat.unsqueeze(0).to(device)
    feat = feat.to(torch.bfloat16)
    return feat, feat_lens



def do_format_shard_manifest4one(input_shards_path, tmp_file_path=None):
    if tmp_file_path is None:
        tmp_file_path = f'~/.cache/.temp/{random.randint(10000, 99999)}.txt'
    data_path_i = input_shards_path
    utils_file.logging_info(f'path:{data_path_i} ')
    final_data_list_i = utils_file.load_list_file_clean(data_path_i)
    # 判断数据类型
    if "combines_list.txt" in data_path_i:
        print(f'是 combine类型的数据')
        tar_root_path = data_path_i.replace('combines_list.txt', 'combines_tar_root.txt')
        if not os.path.exists(tar_root_path):
            utils_file.logging_error(
                f'combine_list.txt:{data_path_i} 对应的 combines_tar_root.txt:{tar_root_path} 不存在')
            return
        tar_root = utils_file.load_first_row_clean(tar_root_path)
        if tar_root.endswith('/'):
            tar_root = tar_root[:-1]
        utils_file.logging_info(f' tar_root:{tar_root}')
        new_final_data_list_i = []
        for data_path_j in final_data_list_i:
            # "combine_path|shard_path"
            tmp_lines = f'{data_path_j}|{tar_root}/{utils_file.do_get_file_pure_name_from_path(data_path_j)}.tar'
            new_final_data_list_i.append(tmp_lines)
    else:
        print(f'不是 combine类型的数据,是传统shard类型的数据')
        new_final_data_list_i = [f'-|{data_path_j}' for data_path_j in final_data_list_i]

    utils_file.logging_info(f'true load num is : {len(new_final_data_list_i)}')
    utils_file.write_list_to_file(new_final_data_list_i, tmp_file_path)
    return tmp_file_path



def convert_numbers_in_string(s):
    # 正则表达式匹配数字（支持整数、小数、负数）
    pattern = r'-?\d+\.?\d*'

    def replace_func(match):
        num_str = match.group()
        try:
            # 尝试转换数字
            return an2cn(num_str)
        except ValueError:
            # 若转换失败（如非有效数字），返回原内容
            return num_str
    # 替换字符串中所有匹配的数字
    return re.sub(pattern, replace_func, s)

def get_test_conf(config_path):
    with open(config_path, 'r', encoding='utf-8') as fin:
        print(f"加载配置文件 {config_path}")
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    configs['dataset_conf']['filter_conf']['filter_no_extra_info'] = False
    test_conf = copy.deepcopy(configs['dataset_conf'])

    # test_conf['filter_conf']['max_length'] = 3000 # whisper最长处理30s 102400
    test_conf['filter_conf']['min_length'] = 10
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 1
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['filter_conf']['filter_no_extra_info'] = False
    test_conf['filter_conf']['max_seq_len'] = 102400
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['spec_trim'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    test_conf['cycle'] = 1
    test_conf['list_shuffle'] = True
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
    elif 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = 1
    test_conf['split_num'] = 1
    test_conf['multi_num'] = 1
    test_conf['other_filter_conf'] = {}
    test_conf['data_recover'] = False
    return configs, test_conf


