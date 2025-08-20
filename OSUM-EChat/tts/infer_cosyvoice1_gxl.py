import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('.')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

import json
gpu_id = 3
cosyvoice = CosyVoice('/mnt/sfs/asr/ckpt/cosyvoice1/CosyVoice-300M-25Hz/CosyVoice-300M-25Hz', gpu_id=gpu_id)
index = 0
while True:
    print('开始生成程序')
    list_str = input("请输入一个JSON格式的列表：")
    prompt_str = input("请输入一个音色示例：")
    prompt_speech_16k = load_wav(prompt_str, 16000)
    
    if list_str == "exit":
        print("退出程序！")
        break
    try:
        token_list = json.loads(list_str)
        token_list = [int(i) for i in token_list]
        #list_str = list_str.strip()
        #token_list = [int(i) for i in list_str.split(' ')]
        print(f"index: {index}, token_list:",token_list)
        j = cosyvoice.inference_zero_shot_gxl('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False, token_list = token_list)
        import os
        os.makedirs('data/output_data', exist_ok=True)
        torchaudio.save('data/output_data/test2cosyvoice1-25hz_{}_gxl.wav'.format(index), j['tts_speech'], cosyvoice.sample_rate)
        index += 1
    except Exception as e:
        print("输入的字符串不是一个JSON格式的列表！")
        print(e)
        continue
