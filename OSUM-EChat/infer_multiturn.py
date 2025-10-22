#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# conda activate  /home/work_nfs16/qjshao/anaconda3/envs/osum/
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
from patches import modelling_qwen2_infer_gpu  # 打patch
from tts.cosyvoice.cli.cosyvoice import CosyVoice
from tts.cosyvoice.utils.file_utils import load_wav

def test_multi_turn_conversation():
    """测试多轮对话功能"""
    
    # 配置参数
    # 2799 is the best model in test4
    CHECKPOINT_PATH = "/home/work_nfs19/sywang/code/OSUM_tmp/OSUM-EChat/exp/humdial_emotion/test4/step_3199.pt"
    # CHECKPOINT_PATH = "/home/work_nfs19/sywang/code/OSUM_tmp/OSUM-EChat/exp/temp/step_2399.pt"
    CONFIG_PATH = "./conf/ct_config.yaml"
    COSYVOICE_MODEL_PATH = "/home/work_nfs16/xlgeng/code/CosyVoice/pretrained_models/CosyVoice-300M-25Hz"
    # 2399不错，就最后一句最后有点瑕疵
    GPU_ID = 0
    device = torch.device(f'cuda:{GPU_ID}')
    
    # 测试音频文件路径（请根据实际情况修改这些路径）
    test_audio_files = [
        "/home/work_nfs19/sywang/code/icassp/data/test/中文/split_audio_任务二-5/0001/0006/0006_turn_01_0.00_12.87.wav",
        "/home/work_nfs19/sywang/code/icassp/data/test/中文/split_audio_任务二-5/0001/0006/0006_turn_02_15.59_20.54.wav",
        "/home/work_nfs19/sywang/code/icassp/data/test/中文/split_audio_任务二-5/0001/0006/0006_turn_03_23.26_30.17.wav",
        "/home/work_nfs19/sywang/code/icassp/data/test/中文/split_audio_任务二-5/0001/0006/0006_turn_04_32.89_43.44.wav",
        "/home/work_nfs19/sywang/code/icassp/data/test/中文/split_audio_任务二-5/0001/0006/0006_turn_06_58.75_62.30.wav",
    ]

    # 参考音频（用于TTS合成）
    reference_audio = "./tts/assert/拟人.wav"  # 可以根据需要修改
    
    print("=" * 60)
    print("开始加载模型...")
    print("=" * 60)
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(CHECKPOINT_PATH, CONFIG_PATH, device=device)
    print("模型加载完成！")
    
    # 加载CosyVoice TTS模型
    print("加载CosyVoice TTS模型...")
    cosyvoice = CosyVoice(COSYVOICE_MODEL_PATH, gpu_id=GPU_ID)
    print("CosyVoice TTS模型加载完成！")
    
    # 加载参考音频
    prompt_speech = load_wav(reference_audio, 22050)
    print(f"参考音频加载完成: {reference_audio}")
    
    # 创建输出目录
    output_dir = "./multiturn_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    print("\n" + "=" * 60)
    print("开始多轮对话测试")
    print("=" * 60)
    
    # 开始多轮对话测试
    for round_num, audio_file in enumerate(test_audio_files, 1):
        print(f"\n--- 第 {round_num} 轮对话 ---")
        print(f"输入音频: {audio_file}")
        
        # 检查音频文件是否存在
        if not os.path.exists(audio_file):
            print(f"警告: 音频文件 {audio_file} 不存在，跳过此轮")
            continue
        
        try:
            # 获取音频特征
            feat, feat_lens = get_feat_from_wav_path(audio_file)
            print(f"音频特征形状: {feat.shape}, 长度: {feat_lens}")
            
            # 开始计时
            start_time = time.time()
            
            # 调用多轮对话生成
            output_text, text_res, speech_res = model.generate_s2s_no_stream_multi_turn(
                wavs=feat, 
                wavs_len=feat_lens
            )
            
            # 结束计时
            end_time = time.time()
            
            # 输出结果
            print(f"文本回复: {output_text[0]}")
            
            # 处理语音token
            speech_tokens = speech_res[0].tolist()[1:]
            
            # 合成音频
            if len(speech_tokens) > 0:
                print("开始合成音频...")
                tts_start_time = time.time()
                
                # 生成输出音频文件名
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                output_audio_path = os.path.join(output_dir, f"round_{round_num}_{timestamp}.wav")
                
                # 使用token_list2wav合成音频
                try:
                    token_list2wav(speech_tokens, prompt_speech, output_audio_path, cosyvoice)
                    tts_end_time = time.time()
                
                    
                    # 检查生成的音频文件
                    if os.path.exists(output_audio_path):
                        file_size = os.path.getsize(output_audio_path)
                        print(f"生成音频文件大小: {file_size} bytes")
                    else:
                        print("警告: 音频文件生成失败")
                        
                except Exception as tts_error:
                    print(f"音频合成失败: {str(tts_error)}")
                    import traceback
                    traceback.print_exc()
            else:
                print("没有生成语音token，跳过音频合成")
                
        except Exception as e:
            print(f"第 {round_num} 轮对话出错: {str(e)}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n" + "=" * 60)
    print("多轮对话测试完成")
    print(f"所有输出文件保存在: {output_dir}")
    print("=" * 60)



if __name__ == "__main__":        
    test_multi_turn_conversation()
