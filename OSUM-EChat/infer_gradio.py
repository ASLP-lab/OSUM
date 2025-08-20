import ast
import base64
import datetime
import json
import logging
import gradio as gr
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



# 在文件开头添加参数解析
CHECKPOINT_PATH_A="./language_think_final.pt"
CHECKPOINT_PATH_B="./tag_think_final.pt"
CONFIG_PATH = "./conf/ct_config.yaml"
NAME_A="language_think"
NAME_B="tag_think"
cosyvoice_model_path = "./CosyVoice-300M-25Hz"



gpu_id=0
device = torch.device(f'cuda:{gpu_id}')

print("开始加载模型 A...")
model_a, tokenizer_a = load_model_and_tokenizer(CHECKPOINT_PATH_A, CONFIG_PATH, device=device)

print("\n开始加载模型 B...")
if CHECKPOINT_PATH_B is not None:
    model_b, tokenizer_b = load_model_and_tokenizer(CHECKPOINT_PATH_B, CONFIG_PATH, device=device)
else:
    model_b, tokenizer_b = None, None
loaded_models = {
    NAME_A: {"model": model_a, "tokenizer": tokenizer_a},
    NAME_B: {"model": model_b, "tokenizer": tokenizer_b},
} if model_b is not None else {
    NAME_A: {"model": model_a, "tokenizer": tokenizer_a},
}
print("\n所有模型已加载完毕。")

cosyvoice = CosyVoice(cosyvoice_model_path, gpu_id=gpu_id)

# 将图片转换为 Base64
with open("./tts/assert/实验室.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

# 任务映射
TASK_PROMPT_MAPPING = {
    "empathetic_s2s_dialogue with think": "THINK",
    "empathetic_s2s_dialogue no think": "s2s_no_think",
    "empathetic_s2t_dialogue with think": "s2t_think",
    "empathetic_s2t_dialogue no think": "s2t_no_think",
    "ASR (Automatic Speech Recognition)": "转录这段音频中的语音内容为文字。",
    "SRWT (Speech Recognition with Timestamps)": "请识别音频内容，并对所有英文词和中文字进行时间对齐，标注格式为<>，时间精度0.1秒。",
    "VED (Vocal Event Detection)(类别:laugh，cough，cry，screaming，sigh，throat clearing，sneeze，other)": "请将音频转化为文字，并在末尾添加相关音频事件标签，标签格式为<>。",
    "SER (Speech Emotion Recognition)(类别:sad，anger，neutral，happy，surprise，fear，disgust，和other)": "请将音频内容转录成文字记录，并在记录末尾标注情感标签,以<>表示。",
    "SSR (Speaking Style Recognition)(类别:新闻科普，恐怖故事，童话故事，客服，诗歌散文，有声书，日常口语，其他)": "请将音频中的讲话内容转化为文字，并在结尾处注明风格标签，用<>表示。",
    "SGC (Speaker Gender Classification)(类别:female,male)": "请将音频转录为文字，并在文本末尾标注性别标签，标签格式为<>。",
    "SAP (Speaker Age Prediction)(类别:child、adult和old)": "请将这段音频转录成文字，并在末尾加上年龄标签，格式为<>。",
    "STTC (Speech to Text Chat)": "首先将语音转录为文字，然后对语音内容进行回复，转录和文字之间使用<开始回答>分割。",
    "Only Age Prediction(类别:child、adult和old)": "请根据音频分析发言者的年龄并输出年龄标签，标签格式为<>。",
    "Only Gender Classification(类别:female,male)": "根据下述音频内容判断说话者性别，返回性别标签，格式为<>.",
    "Only Style Recognition(类别:新闻科普，恐怖故事，童话故事，客服，诗歌散文，有声书，日常口语，其他)": "对于以下音频，请直接判断风格并返回风格标签，标签格式为<>。",
    "Only Emotion Recognition(类别:sad，anger，neutral，happy，surprise，fear，disgust，和other)": "请鉴别音频中的发言者情感并标出，标签格式为<>。",
    "Only  Event Detection(类别:laugh，cough，cry，screaming，sigh，throat clearing，sneeze，other)": "对音频进行标签化，返回音频事件标签，标签格式为<>。",
    "ASR+AGE+GENDER": '请将这段音频进行转录，并在转录完成的文本末尾附加<年龄> <性别>标签。',
    "AGE+GENDER": "请识别以下音频发言者的年龄和性别.",
    "ASR+STYLE+AGE+GENDER": "请对以下音频内容进行转录，并在文本结尾分别添加<风格>、<年龄>、<性别>标签。",
    "STYLE+AGE+GENDER": "请对以下音频进行分析，识别说话风格、说话者年龄和性别。",
    "ASR with punctuations": "需对提供的语音文件执行文本转换，同时为转换结果补充必要的标点。",
    "ASR EVENT AGE GENDER": "请将以下音频内容进行转录，并在转录完成的文本末尾分别附加<音频事件>、<年龄>、<性别>标签。",
    "ASR EMOTION AGE GENDER": "请将下列音频内容进行转录，并在转录文本的末尾分别添加<情感>、<年龄>、<性别>标签。",
}

prompt_audio_choices = [
    {"name": "拟人",
     "value": "./tts/assert/prompt.wav"},
]

prompt_audio_cache = {}
for item in prompt_audio_choices:
    prompt_audio_cache[item["value"]] = load_wav(item["value"], 22050)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')



def do_s2t(model, input_wav_path, input_prompt, profile=False):  # 增加 model 参数
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


def do_s2t4chat(model, input_wav_path, input_prompt, profile=False):  # 增加 model 参数
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

def do_s2t4chat_think(model, input_wav_path, input_prompt, profile=False):  # 增加 model 参数
    model.eval()
    feat, feat_lens = get_feat_from_wav_path(input_wav_path)
    print(f'feat shape: {feat.shape}, feat_lens: {feat_lens}')
    if is_npu: torch_npu.npu.synchronize()
    start_time = time.time()
    res_text = model.generate4chat_think(wavs=feat, wavs_len=feat_lens, cache_implementation="static")[0]
    if is_npu: torch_npu.npu.synchronize()
    end_time = time.time()
    print(f"S2T4Chat 推理消耗时间: {end_time - start_time:.2f} 秒")
    return res_text


def do_t2s(model, input_prompt, text_for_tts, profile=False):  # 增加 model 参数
    model.eval()
    if is_npu: torch_npu.npu.synchronize()
    start_time = time.time()
    res_tensor = model.generate_tts(device=device, text=text_for_tts, )[0]
    res_token_list = res_tensor.tolist()
    res_text = res_token_list[:-1]
    if is_npu: torch_npu.npu.synchronize()
    end_time = time.time()
    print(f"T2S 推理消耗时间: {end_time - start_time:.2f} 秒")
    return res_text


def do_t2t(model, question_txt, profile=False):  # 增加 model 参数
    model.eval()
    if is_npu: torch_npu.npu.synchronize()
    start_time = time.time()
    print(f'开始t2t推理, question_txt: {question_txt}')
    res_text = model.generate_text2text(device=device, text=question_txt)[0]
    if is_npu: torch_npu.npu.synchronize()
    end_time = time.time()
    print(f"T2T 推理消耗时间: {end_time - start_time:.2f} 秒")
    return res_text


def do_s2s(model, input_wav_path, input_prompt, profile=False):  # 增加 model 参数
    model.eval()
    feat, feat_lens = get_feat_from_wav_path(input_wav_path)
    print(f'feat shape: {feat.shape}, feat_lens: {feat_lens}')
    if is_npu: torch_npu.npu.synchronize()
    start_time = time.time()
    output_text, text_res, speech_res = model.generate_s2s_no_stream_with_repetition_penalty(wavs=feat, wavs_len=feat_lens,)
    if is_npu: torch_npu.npu.synchronize()
    end_time = time.time()
    print(f"S2S 推理消耗时间: {end_time - start_time:.2f} 秒")
    return f'{output_text[0]}|{str(speech_res[0].tolist()[1:])}'

def do_s2s_think(model, input_wav_path, input_prompt, profile=False):  # 增加 model 参数
    model.eval()
    feat, feat_lens = get_feat_from_wav_path(input_wav_path)
    print(f'feat shape: {feat.shape}, feat_lens: {feat_lens}')
    if is_npu: torch_npu.npu.synchronize()
    start_time = time.time()
    output_text, text_res, speech_res = model.generate_s2s_no_stream_think_with_repetition_penalty(wavs=feat, wavs_len=feat_lens,)
    if is_npu: torch_npu.npu.synchronize()
    end_time = time.time()
    print(f"S2S 推理消耗时间: {end_time - start_time:.2f} 秒")
    return f'{output_text[0]}|{str(speech_res[0].tolist()[1:])}'

def true_decode_fuc(model, tokenizer, input_wav_path, input_prompt):  # 增加 model 和 tokenizer 参数
    print(f"wav_path: {input_wav_path}, prompt:{input_prompt}")
    if input_wav_path is None and not input_prompt.endswith(("_TTS", "_T2T")):
        print("音频信息未输入，且不是T2S或T2T任务")
        return "错误：需要音频输入"

    if input_prompt.endswith("_TTS"):
        text_for_tts = input_prompt.replace("_TTS", "")
        prompt = "恳请将如下文本转换为其对应的语音token，力求生成最为流畅、自然的语音。"
        res_text = do_t2s(model, prompt, text_for_tts)
    elif input_prompt.endswith("_self_prompt"):
        prompt = input_prompt.replace("_self_prompt", "")
        res_text = do_s2t(model, input_wav_path, prompt)
    elif input_prompt.endswith("_T2T"):
        question_txt = input_prompt.replace("_T2T", "")
        res_text = do_t2t(model, question_txt)
    elif input_prompt in ["识别语音内容，并以文字方式作出回答。",
                          "请推断对这段语音回答时的情感，标注情感类型，撰写流畅自然的聊天回复，并生成情感语音token。",
                          "s2s_no_think"]:
        res_text = do_s2s(model, input_wav_path, input_prompt)
    elif input_prompt == "THINK":
        res_text = do_s2s_think(model, input_wav_path, input_prompt)
    elif input_prompt == "s2t_no_think":
        res_text = do_s2t4chat(model, input_wav_path, input_prompt)
    elif input_prompt == "s2t_think":
        res_text = do_s2t4chat_think(model, input_wav_path, input_prompt)
    else:
        res_text = do_s2t(model, input_wav_path, input_prompt)
        res_text = res_text.replace("<youth>", "<adult>").replace("<middle_age>", "<adult>").replace("<middle>",
                                                                                                     "<adult>")

    print("识别结果为：", res_text)
    return res_text


def do_decode(model, tokenizer, input_wav_path, input_prompt):  # 增加 model 和 tokenizer 参数
    print(f'使用模型进行推理: input_wav_path={input_wav_path}, input_prompt={input_prompt}')
    output_res = true_decode_fuc(model, tokenizer, input_wav_path, input_prompt)
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


def download_audio(input_wav_path):
    return input_wav_path if input_wav_path else None


def get_wav_from_token_list(input_list, prompt_speech):
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    wav_path = f"./tmp/{time_str}.wav"
    return token_list2wav(input_list, prompt_speech, wav_path, cosyvoice)


# --- Gradio 界面 ---
with gr.Blocks() as demo:
    gr.Markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: center; text-align: center;">
            <h1 style="font-family: 'Arial', sans-serif; color: #014377; font-size: 32px; margin-bottom: 0; display: inline-block; vertical-align: middle;">
                OSUM Speech Understanding Model Test
            </h1>
        </div>
        """
    )

    # ### --- 关键修改：添加模型选择器 --- ###
    with gr.Row():
        model_selector = gr.Radio(
            choices=list(loaded_models.keys()),  # 从加载的模型字典中获取选项
            value=NAME_A,  # 默认值
            label="选择推理模型",
            interactive=True
        )

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            audio_input = gr.Audio(label="录音", sources=["microphone", "upload"], type="filepath", visible=True)
        with gr.Column(scale=1, min_width=300):
            output_text = gr.Textbox(label="输出结果", lines=6, placeholder="生成的结果将显示在这里...",
                                     interactive=False)

    with gr.Row():
        task_dropdown = gr.Dropdown(label="任务",
                                    choices=list(TASK_PROMPT_MAPPING.keys()) + ["自主输入文本", "TTS任务", "T2T任务"],
                                    value="empathetic_s2s_dialogue with think")
        prompt_speech_dropdown = gr.Dropdown(label="参考音频（prompt_speech）",
                                             choices=[(item["name"], item["value"]) for item in prompt_audio_choices],
                                             value=prompt_audio_choices[0]["value"], visible=True)
        custom_prompt_input = gr.Textbox(label="自定义任务提示", placeholder="请输入自定义任务提示...", visible=False)
        tts_input = gr.Textbox(label="TTS输入文本", placeholder="请输入TTS任务的文本...", visible=False)
        t2t_input = gr.Textbox(label="T2T输入文本", placeholder="请输入T2T任务的文本...", visible=False)

    audio_player = gr.Audio(label="播放音频", type="filepath", interactive=False)

    with gr.Row():
        download_button = gr.DownloadButton("下载音频", variant="secondary",
                                            elem_classes=["button-height", "download-button"])
        submit_button = gr.Button("开始处理", variant="primary", elem_classes=["button-height", "submit-button"])

    with gr.Row(visible=False) as confirmation_row:
        # ... (确认组件不变)
        gr.Markdown("请判断结果是否正确：")
        confirmation_buttons = gr.Radio(choices=["正确", "错误"], label="", interactive=True, container=False,
                                        elem_classes="confirmation-buttons")
        save_button = gr.Button("提交反馈", variant="secondary")

    # ... (底部内容不变)
    with gr.Row():
        with gr.Column(scale=1, min_width=800):
            gr.Markdown(f"""...""")  # 省略底部HTML


    def show_confirmation(output_res, input_wav_path, input_prompt):
        return gr.update(visible=True), output_res, input_wav_path, input_prompt


    def save_result(if_correct, wav, prompt, res):
        save_to_jsonl(if_correct, wav, prompt, res)
        return gr.update(visible=False)


    # handle_submit 函数现在接收 `selected_model_name` 参数
    def handle_submit(selected_model_name, input_wav_path, task_choice, custom_prompt, tts_text, t2t_text,
                      prompt_speech):
        # 1. 根据选择的模型名称，从字典中获取对应的模型和分词器
        print(f"用户选择了: {selected_model_name}")
        model_info = loaded_models[selected_model_name]
        model_to_use = model_info["model"]
        tokenizer_to_use = model_info["tokenizer"]

        # 2. 准备 prompt
        prompt_speech_data = prompt_audio_cache.get(prompt_speech)
        if task_choice == "自主输入文本":
            input_prompt = custom_prompt + "_self_prompt"
        elif task_choice == "TTS任务":
            input_prompt = tts_text + "_TTS"
        elif task_choice == "T2T任务":
            input_prompt = t2t_text + "_T2T"
        else:
            input_prompt = TASK_PROMPT_MAPPING.get(task_choice, "未知任务类型")

        # 3. 调用重构后的推理函数，传入选择的模型
        output_res = do_decode(model_to_use, tokenizer_to_use, input_wav_path, input_prompt)

        # 4. 处理输出 (逻辑不变)
        wav_path_output = input_wav_path
        if task_choice == "TTS任务" or "empathetic_s2s_dialogue" in task_choice:
            if isinstance(output_res, list):  # TTS case
                wav_path_output = get_wav_from_token_list(output_res, prompt_speech_data)
                output_res = "生成的token: " + str(output_res)
            elif isinstance(output_res, str) and "|" in output_res:  # S2S case
                try:
                    text_res, token_list_str = output_res.split("|")
                    token_list = json.loads(token_list_str)
                    wav_path_output = get_wav_from_token_list(token_list, prompt_speech_data)
                    output_res = text_res
                except (ValueError, json.JSONDecodeError) as e:
                    print(f"处理S2S输出时出错: {e}")
                    output_res = f"错误：无法解析模型输出 - {output_res}"

        return output_res, wav_path_output


    # --- 绑定事件 (下拉框逻辑不变) ---
    task_dropdown.change(fn=lambda choice: gr.update(visible=choice == "自主输入文本"), inputs=task_dropdown,
                         outputs=custom_prompt_input)
    task_dropdown.change(fn=lambda choice: gr.update(visible=choice == "TTS任务"), inputs=task_dropdown,
                         outputs=tts_input)
    task_dropdown.change(fn=lambda choice: gr.update(visible=choice == "T2T任务"), inputs=task_dropdown,
                         outputs=t2t_input)

    submit_button.click(
        fn=handle_submit,
        # 在 inputs 列表中添加模型选择器 `model_selector`
        inputs=[model_selector, audio_input, task_dropdown, custom_prompt_input, tts_input, t2t_input,
                prompt_speech_dropdown],
        outputs=[output_text, audio_player]
    ).then(
        fn=show_confirmation,
        inputs=[output_text, audio_input, task_dropdown],
        outputs=[confirmation_row, output_text, audio_input, task_dropdown]
    )

    download_button.click(fn=download_audio, inputs=[audio_input], outputs=[download_button])
    save_button.click(fn=save_result, inputs=[confirmation_buttons, audio_input, task_dropdown, output_text],
                      outputs=confirmation_row)

# --- 关键修改：为两个模型分别进行预热 ---
print("开始预热模型...")
warmup_wav_path = "./tts/assert/hq_1.wav"
warmup_prompt = "将这段音频的语音内容详细记录为文字稿。"

for model_name, model_info in loaded_models.items():
    print(f"正在预热 {model_name}...")
    try:
        # 使用重构后的 do_s2t 函数进行预热，传入对应的模型
        res_text = do_s2t(model_info["model"], warmup_wav_path, warmup_prompt, profile=False)
        print(f'{model_name} 预热完成。ASR推理结果: {res_text}')
    except Exception as e:
        print(f"预热 {model_name} 时发生错误: {e}")

# 启动Gradio界面
print("\nGradio 界面启动中...")
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)