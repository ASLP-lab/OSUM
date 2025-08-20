import base64
import json
import time

import gradio as gr
import os

import sys

import torch
import torchaudio
from torch import dtype

from gxl_ai_utils.utils import utils_file
from wenet.utils.init_tokenizer import init_tokenizer
from gxl_ai_utils.config.gxl_config import GxlNode
from wenet.utils.init_model import init_model
import logging
import librosa
import torch
import torchaudio
import numpy as np

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
config_path = "conf/config_llm_huawei_base-version.yaml"
checkpoint_path = "***"
args = GxlNode({
    "checkpoint": checkpoint_path,
})
configs = utils_file.load_dict_from_yaml(config_path)
model, configs = init_model(args, configs)
gpu_id = 0
model = model.cuda(gpu_id)
tokenizer = init_tokenizer(configs)
print(model)
resample_rate = 16000

def do_resample(input_wav_path, output_wav_path):
    """"""
    print(f'input_wav_path: {input_wav_path}, output_wav_path: {output_wav_path}')
    waveform, sample_rate = torchaudio.load(input_wav_path)
    # 检查音频的维度
    num_channels = waveform.shape[0]
    # 如果音频是多通道的，则进行通道平均
    if num_channels > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = torchaudio.transforms.Resample(
        orig_freq=sample_rate, new_freq=16000)(waveform)
    utils_file.makedir_for_file(output_wav_path)
    torchaudio.save(output_wav_path, waveform, 16000)


def do_decode(input_wav_path, input_prompt):
    # input_prompt = TASK_PROMPT_MAPPING.get(input_prompt, "未知任务类型")
    print(f"wav_path: {input_wav_path}, prompt:{input_prompt}")
    timestamp_ms = int(time.time() * 1000)
    now_file_tmp_path_resample = f'./.cache/.temp/{timestamp_ms}_resample.wav'
    do_resample(input_wav_path, now_file_tmp_path_resample)
    input_wav_path = now_file_tmp_path_resample
    waveform, sample_rate = torchaudio.load(input_wav_path)
    waveform = waveform.squeeze(0)  # (channel=1, sample) -> (sample,)
    print(f'wavform shape: {waveform.shape}, sample_rate: {sample_rate}')
    window = torch.hann_window(400)
    stft = torch.stft(waveform,
                      400,
                      160,
                      window=window,
                      return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = torch.from_numpy(
        librosa.filters.mel(sr=sample_rate,
                            n_fft=400,
                            n_mels=80))
    mel_spec = filters @ magnitudes

    # NOTE(xcsong): https://github.com/openai/whisper/discussions/269
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    feat = log_spec.transpose(0, 1)
    feat_lens = torch.tensor([feat.shape[0]], dtype=torch.int64).to(gpu_id)
    feat = feat.unsqueeze(0).to(gpu_id)
    # feat = feat.half()
    # feat_lens = feat_lens.half()
    res_text = model.generate(wavs=feat, wavs_len=feat_lens, prompt=input_prompt)[0]
    print("识别结果：", res_text)
    return res_text


if __name__ == "__main__":
    input_wav_path = "***"
    input_prompt = "将这段音频的语音内容详细记录为文字稿。" # for asr, more prompt refer to ../conf/prompt_config.yaml
    res_text_list = do_decode(input_wav_path, input_prompt)
    print(res_text_list)

