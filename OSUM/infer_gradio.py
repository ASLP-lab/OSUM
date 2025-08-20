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

# 将图片转换为 Base64
with open("common_utils/实验室.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

# 自定义CSS样式
custom_css = """
/* 自定义CSS样式 */
"""

# 任务提示映射
TASK_PROMPT_MAPPING = {
    "ASR (Automatic Speech Recognition)": "执行语音识别任务，将音频转换为文字。",
    "SRWT (Speech Recognition with Timestamps)": "请转录音频内容，并为每个英文词汇及其对应的中文翻译标注出精确到0.1秒的起止时间，时间范围用<>括起来。",
    "VED (Vocal Event Detection)(类别:laugh，cough，cry，screaming，sigh，throat clearing，sneeze，other)": "请将音频转录为文字记录，并在记录末尾标注<音频事件>标签，音频事件共8种：laugh，cough，cry，screaming，sigh，throat clearing，sneeze，other。",
    "SER (Speech Emotion Recognition)(类别:sad，anger，neutral，happy，surprise，fear，disgust，和other)": "请将音频内容转录成文字记录，并在记录末尾标注<情感>标签，情感共8种：sad，anger，neutral，happy，surprise，fear，disgust，和other。",
    "SSR (Speaking Style Recognition)(类别:新闻科普，恐怖故事，童话故事，客服，诗歌散文，有声书，日常口语，其他)": "请将音频内容进行文字转录，并在最后添加<风格>标签，标签共8种：新闻科普、恐怖故事、童话故事、客服、诗歌散文、有声书、日常口语、其他。",
    "SGC (Speaker Gender Classification)(类别:female,male)": "请将音频转录为文本，并在文本结尾处标注<性别>标签，性别为female或male。",
    "SAP (Speaker Age Prediction)(类别:child、adult和old)": "请将音频转录为文本，并在文本结尾处标注<年龄>标签，年龄划分为child、adult和old三种。",
    "STTC (Speech to Text Chat)": "首先将语音转录为文字，然后对语音内容进行回复，转录和文字之间使用<开始回答>分割。"
}

gpu_id = 4
def init_model_my():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    config_path = "conf/config_llm_huawei_base-version.yaml"
    checkpoint_path = "***"
    args = GxlNode({
        "checkpoint": checkpoint_path,
    })
    configs = utils_file.load_dict_from_yaml(config_path)
    model, configs = init_model(args, configs)
    model = model.cuda(gpu_id)
    tokenizer = init_tokenizer(configs)
    print(model)
    return model, tokenizer

model, tokenizer = init_model_my()

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

def true_decode_fuc(input_wav_path, input_prompt):
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

def do_decode(input_wav_path, input_prompt):
    print(f'input_wav_path= {input_wav_path}, input_prompt= {input_prompt}')
    # 省略处理逻辑
    output_res = true_decode_fuc(input_wav_path, input_prompt)
    return output_res

def save_to_jsonl(if_correct, wav, prompt, res):
    data = {
        "if_correct": if_correct,
        "wav": wav,
        "task": prompt,
        "res": res
    }
    with open("results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def handle_submit(input_wav_path, input_prompt):
    output_res = do_decode(input_wav_path, input_prompt)
    return output_res

def download_audio(input_wav_path):
    if input_wav_path:
        # 返回文件路径供下载
        return input_wav_path
    else:
        return None

# 创建Gradio界面
with gr.Blocks(css=custom_css) as demo:
    # 添加标题
    gr.Markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: center; text-align: center;">
            <h1 style="font-family: 'Arial', sans-serif; color: #014377; font-size: 32px; margin-bottom: 0; display: inline-block; vertical-align: middle;">
                OSUM Speech Understanding Model Test
            </h1>
        </div>
        """
    )

    # 添加音频输入和任务选择
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(label="录音", type="filepath")
        with gr.Column(scale=1, min_width=300):  # 给输出框设置最小宽度，确保等高对齐
            output_text = gr.Textbox(label="输出结果", lines=8, placeholder="生成的结果将显示在这里...", interactive=False)

    # 添加任务选择和自定义输入框
    with gr.Row():
        task_dropdown = gr.Dropdown(
            label="任务",
            choices=list(TASK_PROMPT_MAPPING.keys()) + ["自主输入文本"],  # 新增选项
            value="ASR (Automatic Speech Recognition)"
        )
        custom_prompt_input = gr.Textbox(label="自定义任务提示", placeholder="请输入自定义任务提示...", visible=False)  # 新增文本输入框

    # 添加按钮（下载按钮在左边，开始处理按钮在右边）
    with gr.Row():
        download_button = gr.DownloadButton("下载音频", variant="secondary", elem_classes=["button-height", "download-button"])
        submit_button = gr.Button("开始处理", variant="primary", elem_classes=["button-height", "submit-button"])

    # 添加确认组件
    with gr.Row(visible=False) as confirmation_row:
        gr.Markdown("请判断结果是否正确：")
        confirmation_buttons = gr.Radio(
            choices=["正确", "错误"],
            label="",
            interactive=True,
            container=False,
            elem_classes="confirmation-buttons"
        )
        save_button = gr.Button("提交反馈", variant="secondary")

    # 添加底部内容
    with gr.Row():
        # 底部内容容器
        with gr.Column(scale=1, min_width=800):  # 设置最小宽度以确保内容居中
            gr.Markdown(
                f"""
                <div style="position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); display: flex; align-items: center; justify-content: center; gap: 20px;">
                    <div style="text-align: center;">
                        <p style="margin: 0;"><strong>Audio, Speech and Language Processing Group (ASLP@NPU),</strong></p>
                        <p style="margin: 0;"><strong>Northwestern Polytechnical University</strong></p>
                    </div>
                    <img src="data:image/png;base64,{encoded_string}" alt="OSUM Logo" style="height: 80px; width: auto;">
                </div>
                """
            )

    # 绑定事件
    def show_confirmation(output_res, input_wav_path, input_prompt):
        return gr.update(visible=True), output_res, input_wav_path, input_prompt

    def save_result(if_correct, wav, prompt, res):
        save_to_jsonl(if_correct, wav, prompt, res)
        return gr.update(visible=False)

    def handle_submit(input_wav_path, task_choice, custom_prompt):
        if task_choice == "自主输入文本":
            input_prompt = custom_prompt  # 使用用户输入的自定义文本
        else:
            input_prompt = TASK_PROMPT_MAPPING.get(task_choice, "未知任务类型")  # 使用预定义的提示
        output_res = do_decode(input_wav_path, input_prompt)
        return output_res

    task_dropdown.change(
        fn=lambda choice: gr.update(visible=choice == "自主输入文本"),
        inputs=task_dropdown,
        outputs=custom_prompt_input
    )

    submit_button.click(
        fn=handle_submit,
        inputs=[audio_input, task_dropdown, custom_prompt_input],
        outputs=output_text
    ).then(
        fn=show_confirmation,
        inputs=[output_text, audio_input, task_dropdown],
        outputs=[confirmation_row, output_text, audio_input, task_dropdown]
    )

    download_button.click(
        fn=download_audio,
        inputs=[audio_input],
        outputs=[download_button]  # 输出到 download_button
    )

    save_button.click(
        fn=save_result,
        inputs=[confirmation_buttons, audio_input, task_dropdown, output_text],
        outputs=confirmation_row
    )

# 启动Gradio界面
demo.launch()