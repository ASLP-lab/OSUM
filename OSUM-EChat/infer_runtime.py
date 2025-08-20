import ast
import logging
import os
import  sys
import time
import traceback
import torch

from common_utils.utils4infer import get_feat_from_wav_path, load_model_and_tokenizer, token_list2wav

sys.path.insert(0, '.')
sys.path.insert(0, './tts')
sys.path.insert(0, './tts/third_party/Matcha-TTS')
from patches import modelling_qwen2_infer_gpu  # 打patch
from tts.cosyvoice.cli.cosyvoice import CosyVoice
from tts.cosyvoice.utils.file_utils import load_wav


is_npu = False
try:
    import torch_npu
except ImportError:
    is_npu = False
    print("torch_npu is not available. if you want to use npu, please install it.")



gpu_id=0
device = torch.device(f'cuda:{gpu_id}')
checkpoint_path = "./language_think_final.pt"
config_path = "conf/ct_config.yaml"
cosyvoice_model_path = "./CosyVoice-300M-25Hz"

prompt_wav_path = "./tts/assert/prompt.wav"
prompt_audio_cache = {"拟人": load_wav(prompt_wav_path, 22050)}
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

model, tokenizer = load_model_and_tokenizer(checkpoint_path, config_path, device)
cosyvoice = CosyVoice(cosyvoice_model_path, gpu_id=gpu_id)


def do_s2t_speech_understanding(model, input_wav_path, input_prompt):  # 增加 model 参数
    model.eval()
    feat, feat_lens = get_feat_from_wav_path(input_wav_path)
    print(f'feat shape: {feat.shape}, feat_lens: {feat_lens}')
    if is_npu: torch_npu.npu.synchronize()
    start_time = time.time()
    res_text = model.generate(wavs=feat, wavs_len=feat_lens, prompt=input_prompt, cache_implementation="static")[0]
    if is_npu: torch_npu.npu.synchronize()
    end_time = time.time()
    print(f"S2T 推理消耗时间: {end_time - start_time:.2f} 秒")
    return res_text


def do_s2t_chat_no_think(model, input_wav_path):  # 增加 model 参数
    model.eval()
    feat, feat_lens = get_feat_from_wav_path(input_wav_path)
    print(f'feat shape: {feat.shape}, feat_lens: {feat_lens}')
    if is_npu: torch_npu.npu.synchronize()
    start_time = time.time()
    res_text = model.generate4chat(wavs=feat, wavs_len=feat_lens, cache_implementation="static")[0]
    if is_npu: torch_npu.npu.synchronize()
    end_time = time.time()
    print(f"S2T4Chat 推理消耗时间: {end_time - start_time:.2f} 秒")
    return res_text

def do_s2t_chat_think(model, input_wav_path):  # 增加 model 参数
    model.eval()
    feat, feat_lens = get_feat_from_wav_path(input_wav_path)
    print(f'feat shape: {feat.shape}, feat_lens: {feat_lens}')
    if is_npu: torch_npu.npu.synchronize()
    start_time = time.time()
    res_text = model.generate4chat_think(wavs=feat, wavs_len=feat_lens, cache_implementation="static")[0]
    if is_npu: torch_npu.npu.synchronize()
    end_time = time.time()
    print(f"S2T4Chat think 推理消耗时间: {end_time - start_time:.2f} 秒")
    return res_text


def do_t2s(model, text_for_tts):  # 增加 model 参数
    model.eval()
    if is_npu: torch_npu.npu.synchronize()
    start_time = time.time()
    res_tensor = model.generate_tts(device=device, text=text_for_tts)[0]
    res_token_list = res_tensor.tolist()
    res_text = res_token_list[:-1]
    if is_npu: torch_npu.npu.synchronize()
    end_time = time.time()
    print(f"T2S 推理消耗时间: {end_time - start_time:.2f} 秒")
    return res_text


def do_t2t_chat(model, question_txt):  # 增加 model 参数
    model.eval()
    if is_npu: torch_npu.npu.synchronize()
    start_time = time.time()
    print(f'开始t2t推理, question_txt: {question_txt}')
    res_text = model.generate_text2text(device=device, text=question_txt)[0]
    if is_npu: torch_npu.npu.synchronize()
    end_time = time.time()
    print(f"T2T 推理消耗时间: {end_time - start_time:.2f} 秒")
    return res_text


def do_s2s(model, input_wav_path):  # 增加 model 参数
    model.eval()
    feat, feat_lens = get_feat_from_wav_path(input_wav_path)
    print(f'feat shape: {feat.shape}, feat_lens: {feat_lens}')
    if is_npu: torch_npu.npu.synchronize()
    start_time = time.time()
    output_text, text_res, speech_res = model.generate_s2s_no_stream_with_repetition_penalty(wavs=feat, wavs_len=feat_lens)
    if is_npu: torch_npu.npu.synchronize()
    end_time = time.time()
    print(f"S2S 推理消耗时间: {end_time - start_time:.2f} 秒")
    return f'{output_text[0]}|{str(speech_res[0].tolist()[1:])}'

def do_s2s_think(model, input_wav_path):  # 增加 model 参数
    model.eval()
    feat, feat_lens = get_feat_from_wav_path(input_wav_path)
    print(f'feat shape: {feat.shape}, feat_lens: {feat_lens}')
    if is_npu: torch_npu.npu.synchronize()
    start_time = time.time()
    output_text, text_res, speech_res = model.generate_s2s_no_stream_think_with_repetition_penalty(wavs=feat, wavs_len=feat_lens)
    if is_npu: torch_npu.npu.synchronize()
    end_time = time.time()
    print(f"S2S 推理消耗时间: {end_time - start_time:.2f} 秒")
    return f'{output_text[0]}|{str(speech_res[0].tolist()[1:])}'


print("开始预热模型...")
warmup_wav_path = "./tts/assert/hq_1.wav"
warmup_prompt = "将这段音频的语音内容详细记录为文字稿。"
print(f"正在预热 ...")
try:
    # 使用重构后的 do_s2t 函数进行预热，传入对应的模型
    res_text = do_s2t_speech_understanding(model, warmup_wav_path, warmup_prompt)
    print(f'预热完成。ASR推理结果: {res_text}')
except Exception as e:
    traceback.print_exc()
    print(f"预热 时发生错误: {e}")

res = do_s2s_think(model, warmup_wav_path)
print(f'S2S think推理结果: {res}')
token_list = ast.literal_eval(res.split('|')[1])
output_wav_path = "./tmp/s2s_think_output1.wav"
os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
token_list2wav(token_list, prompt_audio_cache["拟人"], output_wav_path,cosyvoice=cosyvoice)


res = do_s2s(model, warmup_wav_path)
print(f'S2S no think推理结果: {res}')
token_list = ast.literal_eval(res.split('|')[1])
output_wav_path = "./tmp/s2s_output.wav"
os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
token_list2wav(token_list, prompt_audio_cache["拟人"], output_wav_path,cosyvoice=cosyvoice)


res = do_s2t_chat_think(model, warmup_wav_path)
print(f'S2t think推理结果: {res}')


res = do_s2t_chat_no_think(model, warmup_wav_path)
print(f'S2T no think推理结果: {res}')


res = do_t2s(model, "你好，我是你的语音助手OSUM-EChat。")
print(f't2S推理结果: {res}')
output_wav_path = "./tmp/t2s_output.wav"
os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
token_list2wav(res, prompt_audio_cache["拟人"], output_wav_path,cosyvoice=cosyvoice)


res = do_s2t_speech_understanding(model, warmup_wav_path, "将这段音频的语音内容详细记录为文字稿。")
print(f'S2t asr推理结果: {res}')


res = do_t2t_chat(model, "嗯那对于你而言你觉得如果嗯在未来你心中所期待的婚礼会是什么样子的")
print(f't2t结果: {res}')
