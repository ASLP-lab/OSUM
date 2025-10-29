#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import os
import sys
import time
import datetime
from common_utils.utils4infer import get_feat_from_wav_path, load_model_and_tokenizer, token_list2wav

# 添加路径
sys.path.insert(0, '.')
sys.path.insert(0, './tts')
sys.path.insert(0, './tts/third_party/Matcha-TTS')
from patches import modelling_qwen2_infer_gpu
from tts.cosyvoice.cli.cosyvoice import CosyVoice
from tts.cosyvoice.utils.file_utils import load_wav

def test_multi_turn_conversation():
    """测试多轮对话功能"""
    
    # 配置参数
    CHECKPOINT_PATH = ""
    CONFIG_PATH = "./conf/ct_config.yaml"
    COSYVOICE_MODEL_PATH = "./CosyVoice-300M-25Hz"
    # 2399不错，就最后一句最后有点瑕疵
    GPU_ID = 0
    device = torch.device(f'cuda:{GPU_ID}')

    test_audio_files = [
        "./1.wav",
        "./2.wav",
        "./3.wav",
        "./4.wav",
    ]

    reference_audio = "./tts/assert/prompt2.wav"
    model, tokenizer = load_model_and_tokenizer(CHECKPOINT_PATH, CONFIG_PATH, device=device)
    cosyvoice = CosyVoice(COSYVOICE_MODEL_PATH, gpu_id=GPU_ID)
    
    prompt_speech = load_wav(reference_audio, 22050)
    output_dir = "./multiturn_output"
    os.makedirs(output_dir, exist_ok=True)
    
    for round_num, audio_file in enumerate(test_audio_files, 1):
        try:
            feat, feat_lens = get_feat_from_wav_path(audio_file)
            output_text, text_res, speech_res = model.generate_s2s_no_stream_multi_turn(
                wavs=feat, 
                wavs_len=feat_lens
            )
            print(f"text: {output_text[0]}")
            speech_tokens = speech_res[0].tolist()[1:]
            
            # 合成音频
            if len(speech_tokens) > 0:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                output_audio_path = os.path.join(output_dir, f"round_{round_num}_{timestamp}.wav")
                try:
                    token_list2wav(speech_tokens, prompt_speech, output_audio_path, cosyvoice)
                    if os.path.exists(output_audio_path):
                        print(f"audio: {output_audio_path}")
                    else:
                        print("audio: failed")
                except Exception as tts_error:
                    print(f"audio: {str(tts_error)}")
        except Exception as e:
            print(f"error: {str(e)}")
            break

if __name__ == "__main__":        
    test_multi_turn_conversation()
