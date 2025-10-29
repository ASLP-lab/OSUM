# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import codecs
import copy

import librosa
import logging
import random
import tarfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torch.nn.functional as F
from gxl_ai_utils.utils import utils_file
from torch.nn.utils.rnn import pad_sequence

from wenet.text.hugging_face_tokenizer import HuggingFaceTokenizer
torchaudio.set_audio_backend("soundfile")

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])

import re


def process_text(text):
    # 1. 删除汉字左右两侧的空格
    text = re.sub(r'\s*([\u4e00-\u9fff])\s*', r'\1', text)
    # 2. 将英文转成小写
    text = text.lower()
    # 3. 删除 < 和 > 符号两侧的空格
    text = re.sub(r'\s*<\s*', '<', text)
    text = re.sub(r'\s*>\s*', '>', text)
    return text

def process_text2(text, task_tag):
    # 1. 删除汉字左右两侧的空格
    text = re.sub(r'\s*([\u4e00-\u9fff])\s*', r'\1', text)
    # 2. 将英文转成小写
    if task_tag == "<TRANSCRIBE>":
        text = text.lower()
    # 3. 删除 < 和 > 符号两侧的空格
    text = re.sub(r'\s*<\s*', '<', text)
    text = re.sub(r'\s*>\s*', '>', text)
    return text

def insert_at_position(lst, item_str, position, is_wav:bool):
    """
    将 item_str 插入到 lst 的第 position 个位置（1-based），
    若 lst 长度不足则以 "-1" 填充至目标长度后再插入。
    """
    index = position - 1
    # 一次性计算需要补充的 "-1" 数目并批量 extend
    if len(lst) < position:
        lst.extend(["-1"] * (position - len(lst)))
    if lst[index] != "-1":
        assert isinstance(lst[index], dict), f'lst[index] is not a dict {lst[index]}'
        if is_wav:
            lst[index]['wav'] = item_str['wav']
        else:
            lst[index]['txt'] = item_str['txt']
    else:
        lst[index] = item_str
    return lst

def check_wav_format(s):
    match = re.fullmatch(r"wav_(\d+)", s)
    if match:
        return True, int(match.group(1))
    else:
        return False, -1

def check_txt_format(s):
    match = re.fullmatch(r"txt_(\d+)", s)
    if match:
        return True, int(match.group(1))
    else:
        return False, -1

def load_dict_list_from_jsonl(jsonl_file_path) -> list:
    """"""
    with codecs.open(jsonl_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines_res = []
        for line in lines:
            try:
                line = json.loads(line)
                lines_res.append(line)
            except Exception as e:
                print(e)
                continue
    return lines_res
def url_opener(data):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample['src']
        if "|" not in url:
            utils_file.logging_error(f'OSUM-EChat url_opener 错误，url格式不正确 {url}, 不含有|')
            continue
        combine_path, shard_path = url.split('|')
        if combine_path == "-":
            big_dict = None
        else:
            try:
                dict_list = load_dict_list_from_jsonl(combine_path)
            except Exception as e:
                utils_file.logging_error(f'OSUM-EChat url_opener 错误，加载combine_path {combine_path} 失败 {e}')
                dict_list = []
            big_dict = {}
            for item in dict_list:
                big_dict[item['key']] = item
        try:
            pr = urlparse(shard_path)
            # local file
            if pr.scheme == '' or pr.scheme == 'file':
                stream = open(shard_path, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f'wget -q -O - {shard_path}'
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream,big_dict=big_dict)
            yield sample
        except Exception as ex:
            logging.warning('Failed to open {}'.format(shard_path))



def tar_file_and_group_full_data(data, total_num=0):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix

        Args:
            data: Iterable[{src, stream}]

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    index = 0
    total_num = total_num
    for sample in data:
        index += 1
        # utils_file.logging_limit_print(f'OSUM-EChat 正在消化第{index}个tar包')
        assert 'stream' in sample
        stream = None
        try:
            stream = tarfile.open(fileobj=sample['stream'], mode="r:*")
            big_dict = sample['big_dict']
            prev_prefix = None
            example = {'history': []}
            valid = True
            for tarinfo in stream:
                name = tarinfo.name
                pos = name.rfind('.')
                assert pos > 0, f' pos {pos}'
                prefix, postfix = name[:pos], name[pos + 1:]
                if prev_prefix is not None and prefix != prev_prefix:
                    example['key'] = prev_prefix
                    if valid:
                        # assert 'txt' in example
                        if 'txt' not in example:
                            example['txt'] = ''
                        if 'wav' not in example:
                            example['wav'] = torch.randn(1, 160000)
                            example['sample_rate'] = 16000
                        # utils_file.logging_info(f'OSUM-EChat SHUCHU第{index}个tar包')
                        yield example
                    example = {'history': []}
                    valid = True
                with stream.extractfile(tarinfo) as file_obj:
                    try:
                        if big_dict is not None:
                            if prefix not in big_dict:
                                raise Exception(f'{prefix} not in big_dict')
                            else:
                                info_dict = big_dict[prefix]
                                if 'txt' not in info_dict or 'task' not in info_dict or 'extra' not in info_dict:
                                    raise Exception(f'info_dict {info_dict} not include txt, task, extra')
                                # utils_file.logging_limit_print(f'info dict: {info_dict}')
                                if postfix == 'txt':
                                    example['txt'] = info_dict['txt']
                                elif postfix == 'task':
                                    example['task'] = info_dict['task']
                                elif postfix == 'extra':
                                    example['extra'] = info_dict['extra']
                                elif postfix in AUDIO_FORMAT_SETS:
                                    waveform, sample_rate = torchaudio.load(file_obj)
                                    # 检查音频的维度
                                    num_channels = waveform.shape[0]
                                    # 如果音频是多通道的，则进行通道平均
                                    if num_channels > 1:
                                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                                    example['wav'] = waveform
                                    example['sample_rate'] = sample_rate
                                else:
                                    pass
                        elif check_wav_format(postfix)[0]:
                            position = check_wav_format(postfix)[1]
                            waveform, sample_rate = torchaudio.load(file_obj)
                            if sample_rate != 16000:
                                waveform = torchaudio.transforms.Resample(
                                    orig_freq=sample_rate, new_freq=16000)(waveform)
                            # feat = do_compute_log_mel_spectrogram(waveform)
                            history_item = {'wav': waveform, "txt": "", 'position': position}
                            insert_at_position(example['history'], history_item, position, is_wav=True)
                        elif check_txt_format(postfix)[0]:
                            position = check_txt_format(postfix)[1]
                            txt_str = file_obj.read().decode('utf8').strip()
                            history_item = {'wav': '', "txt": txt_str, 'position': position}
                            insert_at_position(example['history'], history_item, position, is_wav=False)
                        else:
                            if postfix == 'txt':
                                example['txt'] = file_obj.read().decode('utf8').strip()
                            elif postfix == 'task':
                                example['task'] = file_obj.read().decode('utf8').strip()
                            elif postfix == 'extra':
                                extra_str = file_obj.read().decode('utf8').strip()
                                example['extra'] = json.loads(extra_str)
                            elif postfix in AUDIO_FORMAT_SETS:
                                waveform, sample_rate = torchaudio.load(file_obj)
                                # 检查音频的维度
                                num_channels = waveform.shape[0]
                                # 如果音频是多通道的，则进行通道平均
                                if num_channels > 1:
                                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                                example['wav'] = waveform
                                example['sample_rate'] = sample_rate
                            else:
                                pass
                    except Exception as ex:
                        valid = False
                        utils_file.logging_error('error to parse ex: {}'.format(ex))
                        # 1. 基础信息：错误对象、文件名、错误类型
                        # error_msg = (
                        #     f"Failed to parse {name}! "
                        #     f"Error type: {type(ex).__name__}, "
                        #     f"Message: {str(ex)}"
                        # )
                        # # 2. 补充堆栈跟踪（完整调用链路）
                        # stack_trace = traceback.format_exc()
                        # # 3. 组合日志信息，使用warning级别输出（或error级别更合适）
                        # logging.warning(f"{error_msg}\nStack trace:\n{stack_trace}")
                prev_prefix = prefix
            if prev_prefix is not None:
                example['key'] = prev_prefix
                if 'txt' in example:
                    if 'wav' not in example:
                        example['wav'] = torch.randn(1, 160000)
                        example['sample_rate'] = 16000
                    utils_file.logging_info(f'*************OSUM-EChat SHUCHU第{index}/{total_num}个tar包')
                    yield example
        except Exception as ex:
            logging.warning(
                'In tar_file_and_group: {} when processing {}'.format(
                    ex, sample['src']))
        finally:
            if stream is not None:
                stream.close()
            if 'process' in sample:
                sample['process'].communicate()
            sample['stream'].close()



def parse_raw(data):
    """ Parse key/wav/txt from json line

        Args:
            data: Iterable[str], str is a json line has key/wav/txt

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        obj = json.loads(json_line)
        assert 'key' in obj
        assert 'wav' in obj
        assert 'txt' in obj
        key = obj['key']
        wav_file = obj['wav']
        txt = obj['txt']
        try:
            if 'start' in obj:
                assert 'end' in obj
                sample_rate = torchaudio.info(wav_file).sample_rate
                start_frame = int(obj['start'] * sample_rate)
                end_frame = int(obj['end'] * sample_rate)
                waveform, _ = torchaudio.load(filepath=wav_file,
                                              num_frames=end_frame -
                                                         start_frame,
                                              frame_offset=start_frame)
            else:
                waveform, sample_rate = torchaudio.load(wav_file)
                # 检查音频的维度
                num_channels = waveform.shape[0]
                # 如果音频是多通道的，则进行通道平均
                if num_channels > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
            example = copy.deepcopy(obj)  # copy and keep all the fields
            example['wav'] = waveform  # overwrite wav
            example['sample_rate'] = sample_rate
            yield example
        except Exception as ex:
            logging.warning('Failed to read {}'.format(wav_file))


def parse_speaker(data, speaker_table_path):
    speaker_dict = {}
    with open(speaker_table_path, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            speaker_dict[arr[0]] = int(arr[1])
    for sample in data:
        assert 'speaker' in sample
        speaker = sample['speaker']
        sample['speaker'] = speaker_dict.get(speaker, 0)
        yield sample


global_style_dict = {
    "朗读": "新闻科普",
    "科普百科": "新闻科普",
    "悬疑恐怖": "恐怖故事",
    "童话故事": "童话故事",
    "客服": "客服",
    "诗歌": "诗歌散文",
    "散文": "诗歌散文",
    "武侠评书": "有声书",
    "小说": "有声书",
    "历史": "有声书",
    "科幻": "有声书",
    "对话": "日常口语",
    "口语": "日常口语",
    "幽默": "其他",
    "其他": "其他",
}
# global_chat_dict = utils_file.load_dict_from_scp("/mnt/sfs/asr/update_data/3500_chat_asr/osum_echat_all_3500_with_asr_chat.scp")

asr_X_set = {"<TRANSCRIBE> <EMOTION>", "<TRANSCRIBE> <STYLE>", "<TRANSCRIBE> <CAPTION>", "<TRANSCRIBE> <GENDER>",
             "<TRANSCRIBE> <AGE>"}

natural_language_set = {"<TRANSCRIBE> <STYLE>",
                        "<TRANSCRIBE> <CAPTION>",
                        "<TRANSCRIBE> <GENDER>",
                        "<TRANSCRIBE> <AGE>",
                        "<TRANSCRIBE> <EMOTION>",
                        "<STYLE>",
                        "<CAPTION>",
                        "<GENDER>",
                        "<AGE>",
                        "<EMOTION>",
                        }

chat_set = set([
    "<TRANSCRIBE> <S2TCHAT>",
])
import re


def extract_first_content(s):
    # 使用正则表达式匹配尖括号中的内容, input: "dfsfs<喜喜>", output: "<喜喜>"
    match = re.search(r'<[^>]+>', s)
    if match:
        return match.group()
    else:
        return "None_in_extract_X"


def extract_all_contents(s):
    # 使用正则表达式匹配所有尖括号中的内容
    matches = re.findall(r'<[^>]+>', s)
    return matches


def extract_answer(s):
    res_items = s.strip().split("<开始回答>")
    if len(res_items) == 2:
        res = res_items[1]
    else:
        res = "None_in_extract_answer"
    return res


def replace_keys_in_brackets(input_str, key_value_dict):
    for key, value in key_value_dict.items():
        # 构造匹配 <key> 形式的正则表达式模式
        pattern = re.compile(r'<{}>'.format(key))
        input_str = pattern.sub(f"<{value}>", input_str)
    return input_str


import json


def process_tagged_string(s):
    """
    处理带标签的字符串，将标签替换为对应的内容
    Args:
        s:

    Returns:
        <tag>  <tag>xxxx
    """
    match = re.match(r'^<([^<>]+)>', s)
    if match:
        tag = match.group(1)
        return f"<{tag}>", s
    else:
        new_s = "<中立>" + s
        return "<中立>", new_s


def process_tagged_string_and_delete_tag_from_txt(s):
    """
    处理带标签的字符串，提取标签并返回标签和去除标签后的内容
    Args:
        s: 带标签的字符串，格式如 "<tag>XXXX"

    Returns:
        元组 (标签, 去除标签后的内容)，例如 ("<tag>", "XXXX")
    """
    # 匹配以<标签>开头的字符串
    match = re.match(r'^<([^<>]+)>(.*)$', s)
    if match:
        tag = match.group(1)
        content = match.group(2).lstrip()  # 去除标签后的内容（可选项：移除开头空格）
        return f"<{tag}>", content
    else:
        # 无标签时添加默认<中立>标签，返回标签和原始内容
        return "<中立>", s


def split_txt2asr_tag(text):
    # 使用正则表达式匹配标签
    match = re.match(r'^(.*?)(<[^>]+>)$', text)

    # 如果匹配到一个标签
    if match:
        return match.group(1), match.group(2)
    else:
        return None, None


# 过滤自我介绍
self_list = [
    # 原有 LLM 名称及变体
    "MOSS Assistant", "MOSS助手",
    "QWEN", "QWEN Assistant", "QWEN助手","小智机器人", "小智助手", "小智AI助手","Qwen助手"

    # 通用英文助手/AI 名称
    "VirtualAssistant", "virtual assistant",
    "Helper", "helper", "ChatBot","chat bot", "chatbot", "Chat Bot",
    "AI Agent", "AI agent",  "ChatGPT", "chatgpt","<think>"

    # 常见英文名字
    # " Alice ", " alice ", " Bob ", " bob ", " Charlie ", " charlie ",
    # " Dave ", " dave ", " Eve ", " eve ", " Grace ", " grace ", " Tom ",

    # 常见中文名字
    # "小明", "小红", "小刚", "晓明", "晓红", "阿强", "阿丽",
    # "丽丽", "婷婷", "王磊", "李娜", "张伟", "赵敏", "刘洋", "陈晨",
    # "小李", "小王", "小赵", "小周", "小吴", "小马", "小暖", "乐哥","李娜"

    # 中文名字 + 助手/AI 后缀
    "小明助手", "小红小助手", "小刚AI", "晓明助手", "阿强Bot",
    "阿丽小助手", "王磊AI助手", "李娜AI", "张伟助手", "赵敏Bot",
    "刘洋AI小助手", "陈晨智能助手", "<think>"

    # 混合中英文风格
    "MOSS小助手", "QWEN小助手", "小智Bot", "Assistant小智", "AI小智",
    "ChatBot小明", "VirtualAssistant李娜","大型语言模型","语言模型"

]
escaped = [re.escape(w) for w in self_list if w]
pattern = re.compile(r"(" + "|".join(escaped) + r")")



def if_have_other_name(text):
    matches = pattern.findall(text)
    if matches:
        print("出现了：", set(matches), text)
        return True
    else:
        return False


emotion_tags = {"<HAPPY>", "<SAD>", "<ANGRY>", "<ANGER>", "<FEAR>", "<DISGUST>", "<SURPRISE>", "<NEUTRAL>"}
answer_emotion_tags = {"<ANGER>","<FEAR>","<HAPPY>","<SURPRISE>","<SAD>","<DISGUST>","<CONFUSED>","<SARCASM>","<EMBARRASSED>","<CURIOUS>","<WORRIED>","<SHY>","<SORRY>","<NEUTRAL>",}
age_tags = {"<CHILD>", "<ADULT>", "<OLD>"}
gender_tags = {"<MALE>", "<FEMALE>"}
none_tags = {"<NONE>", "<NULL>", "<None>", "<none>", "<null>"}


def expand_dialogue_to_prefixes(data, max_turn=0, min_turn=1, keep_final=True):
    """
    Expand each multi-turn sample into a sequence of prefix sub-samples to achieve progressive training from single-turn to multi-turn.
    """
    for sample in data:
        history = sample.get('history', [])
        total_turns = len(history) + 1

        if total_turns <= 1:
            sample['turn_idx'] = 1
            sample['turn_total'] = 1
            yield sample
            continue

        max_expand = total_turns if max_turn <= 0 else min(total_turns, max_turn)
        start_turn = max(1, min_turn)

        for turn_idx in range(start_turn, max_expand + 1):
            child = copy.deepcopy(sample)
            
            if turn_idx <= len(history):
                cur = history[turn_idx - 1]
                if not isinstance(cur, dict):
                    continue
                
                if 'wav' in cur and cur['wav'] is not None:
                    child['wav'] = cur['wav']
                else:
                    continue
                
                if 'txt' in cur:
                    child['txt'] = cur['txt']
                else:
                    continue
                
                extra = child.get('extra', {})
                speech_token_key = f'speech_token_{turn_idx}'
                if speech_token_key in extra:
                    child['speech_token'] = extra[speech_token_key]
                else:
                    child['speech_token'] = []

                child['history'] = []
                for i in range(turn_idx - 1):
                    hist_item = copy.deepcopy(history[i])
                    hist_item['wav'] = do_compute_log_mel_spectrogram(hist_item['wav'])
                    speech_token_key = f'speech_token_{i + 1}'
                    if speech_token_key in extra:
                        speech_token = extra[speech_token_key]
                        speech_token_num = extra.get('speech_token_num', 4097)
                        if isinstance(speech_token, list) and len(speech_token) > 0:
                            hist_item['speech_token'] = [speech_token_num - 1] + speech_token + [speech_token_num - 1]
                        else:
                            hist_item['speech_token'] = []
                    else:
                        hist_item['speech_token'] = []
                    child['history'].append(hist_item)

                
                if 'extra' in child:
                    new_extra = {}
                    for k, v in child['extra'].items():
                        if not k.startswith('speech_token'):
                            new_extra[k] = v
                    child['extra'] = new_extra
                    
            else:
                extra = child.get('extra', {})
                if 'speech_token' in extra:
                    child['speech_token'] = extra['speech_token']
                else:
                    child['speech_token'] = []

                child['history'] = []
                for i in range(len(history)):
                    hist_item = copy.deepcopy(history[i])
                    hist_item['wav'] = do_compute_log_mel_spectrogram(hist_item['wav'])
                    speech_token_key = f'speech_token_{i + 1}'
                    if speech_token_key in extra:
                        speech_token = extra[speech_token_key]
                        speech_token_num = extra.get('speech_token_num', 4097)
                        if isinstance(speech_token, list) and len(speech_token) > 0:
                            hist_item['speech_token'] = [speech_token_num - 1] + speech_token + [speech_token_num - 1]
                        else:
                            hist_item['speech_token'] = []
                    else:
                        hist_item['speech_token'] = []
                    child['history'].append(hist_item)
        
                
                if 'extra' in child:
                    new_extra = {}
                    for k, v in child['extra'].items():
                        if k == 'speech_token' or not k.startswith('speech_token'):
                            new_extra[k] = v
                    child['extra'] = new_extra

            # 添加轮次信息
            child['turn_idx'] = turn_idx
            child['turn_total'] = total_turns
            yield child

        if not keep_final:
            continue


def tokenize(data, tokenizer: HuggingFaceTokenizer, other_tokenze_conf={}, global_prompt_dict=None, speech_token_num=1):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            tokenizer:
            data: Iterable[{key, wav, txt, sample_rate}]

        Returns:
            Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    for sample in data:

        # =========== handle extra ,将其统一为字典================================
        origin_extra = sample.get('extra', {})
        if type(origin_extra) == str:
            try:
                # utils_file.logging_limit_print(f"origin_extra is a str, try to load it as json")
                sample['extra'] = json.loads(origin_extra)
            except json.JSONDecodeError:
                # utils_file.logging_error("Error: 'extra' is not a valid JSON string.")
                sample['extra'] = {}
        elif type(origin_extra) == dict:
            sample['extra'] = origin_extra
        else:
            sample['extra'] = {}
        final_extra = sample['extra']
        # =========== handle extra end =======================================

        insert_prompt = None
        # ============handle task， txt,确保task txt标签一定存在 ====================
        if 'task' not in sample:
            sample['task'] = "<TRANSCRIBE>"

        if 'txt' not in sample:
            sample['txt'] = ""
        task_name = sample['task']
        txt = sample['txt']
        # ============handle task txt end =======================================



        # ===============做补丁处理 ==========================================
        if if_have_other_name(txt):
            print(f"txt: {txt} 存在其他名称，跳过")
            continue
        txt = txt.replace("\n", " ")

        if "<AGE>" in task_name and (
                "<YOUTH>" in sample['txt'] or "<MIDDLE_AGE>" in sample['txt'] or "<MIDDLE>" in sample['txt']):
            txt = sample['txt'].replace("<YOUTH>", "<ADULT>").replace("<MIDDLE_AGE>", "<ADULT>").replace("<MIDDLE>",
                                                                                                         "<ADULT>")
            sample['txt'] = txt
        if "<STYLE>" in sample['task']:
            txt = replace_keys_in_brackets(sample['txt'], global_style_dict)
            sample['txt'] = txt
        if task_name == "TEXT2TOKEN":
            task_name = "<TEXT2TOKEN>"
            sample['task'] = task_name

        if task_name == "<Speech2TEXTandTOKEN>" or task_name == "<S2TCHAT> <TEXT2TOKEN> <EMOTION>":
            task_name = "<S2TCHAT> <TEXT2TOKEN>"
            sample['task'] = task_name

        if task_name == "<S2TCHAT> <TEXT2TOKEN>":
            history_list = sample.get('history', [])
            history_len = len(history_list)
            if history_len > 0:
                task_name = "<S2TCHAT> <TEXT2TOKEN> <HISTORY>"
                sample['task'] = task_name

        # ================补丁处理结束 ==============================================

        unk_tag = "<&&>"  # 对应数字为 27,7672,29,...

        # =============针对理解任务做only X 的处理, 加入理解任务转换的任务===================================
        if other_tokenze_conf.get("use_50_per_change_if_only_X", False) and task_name in asr_X_set:
            # utils_file.logging_limit_print(f"task_name: {task_name}, in asr_X_set")
            # 得到一个50%的随机
            if random.random() < 0.5:
                task_name = task_name.replace("<TRANSCRIBE> ", "")
                sample['task'] = task_name
                # utils_file.logging_limit_print(f"task_name: {task_name},发生任务替换, replace to {sample['task']}")
                txt = extract_first_content(sample['txt'])
                sample['txt'] = txt
                # utils_file.logging_limit_print(f"old txt: {sample['txt']}, 发生了文本替换, replace to new txt: {txt}")
        elif other_tokenze_conf.get("use_50_per_change_if_only_X", False) and task_name in chat_set:
            # utils_file.logging_limit_print(f"task_name: {task_name}, in chat_set")
            # 得到一个50%的随机
            if random.random() < 0.5:
                task_name = task_name.replace("<TRANSCRIBE> ", "")
                sample['task'] = task_name
                # utils_file.logging_limit_print(f"task_name: {task_name},发生任务替换, replace to {sample['task']}")
                txt = extract_answer(sample['txt'])
                sample['txt'] = txt
                # utils_file.logging_limit_print(f"old txt: {sample['txt']}, 发生了文本替换, replace to new txt: {txt}")
        # =============针对理解任务做only X 的处理, 加入理解任务转换的任务 end===================================



        # =======================对tts任务做处理=======================================
        if task_name == "<TEXT2TOKEN>" and other_tokenze_conf.get("use_streaming_tts", {}).get("enable", False):
            if random.random() < other_tokenze_conf.get("use_streaming_tts", {}).get("rate", 0.5):
                task_name = "<TEXT2TOKEN> <STREAMING>"
                sample['task'] = task_name
        # =======================对tts任务处理结束=======================================

        emotion_tag, txt = process_tagged_string_and_delete_tag_from_txt(txt)
        # =======================处理s2t think========================================
        if task_name == "<S2TCHAT> <THINKER>":
            # emotion_tag, txt = process_tagged_string(txt)  # 如果开头没<中立>，则加上<中立>
            if 'think_str' in final_extra:
                think_str = final_extra['think_str']
                txt = f'<think>{think_str}<think end>{txt}'
            else:
                utils_file.logging_error(f"error: think_str is not in extra, {sample}")
                continue
        # =======================处理s2t think end=====================================

        # ===================处理s2s think============================================
        if task_name == "<S2TCHAT> <TEXT2TOKEN> <THINK>":
            # emotion_tag, txt = process_tagged_string(txt)  # 如果开头没<中立>，则加上<中立>
            if 'think_str' in final_extra:
                think_str = final_extra['think_str']
                txt = f'<think>{think_str}<think end>{txt}'
            else:
                utils_file.logging_error(f"error: think_str is not in extra, {sample}")
                continue
        # ====================处理s2s think end============================================


        # =======================得到 txt的数字化token =================================
        tokens, label = tokenizer.tokenize(process_text2(txt, sample.get("task", "<TRANSCRIBE>")))
        sample['tokens'] = tokens  # token是字符， label是数字
        if txt.endswith(unk_tag):
            sample['label'] = label
        else:
            sample['label'] = label + [tokenizer.tokenizer.eos_token_id]
        # =======================得到 txt的数字化token 结束 =================================


        # ====================处理prompt ==============================================
        try:
            if "question" in sample['extra']: #if sample['task'] == '<TEXT2TEXT>':
                question = sample['extra'].get('question', "")
                if question == "":
                    utils_file.logging_info(f"error: question is empty, {sample}")
                    continue
                prompt = question
            else:
                if insert_prompt is not None:
                    prompt = insert_prompt
                else:
                    if task_name not in global_prompt_dict:
                        prompt = "<no_prompt>"
                    else:
                        random_index = random.randint(0, len(global_prompt_dict[task_name]) - 1)
                        prompt = global_prompt_dict[task_name][random_index]
            if prompt == "<no_prompt>":
                # utils_file.logging_limit_print(f'no prompt for {task_name}')
                sample['prompt'] = []
            else:
                sample['prompt'] = tokenizer.tokenize(prompt)[1]  # labels
        except Exception as e:
            utils_file.logging_info(f"error in extract prompt, {e},task_name: {task_name}, sample: {sample}")
            continue
        # ====================处理prompt 结束 =======================================



        # ========================处理speech token ================================
        if task_name == "<S2TCHAT> <TEXT2TOKEN>" or task_name == "<S2TCHAT> <TEXT2TOKEN> <THINK>" or task_name == "<TEXT2TOKEN>" or task_name == "<TEXT2TOKEN> <STREAMING>":
            if "speech_token" in final_extra:
                speech_token_tmp = final_extra['speech_token']
                if not isinstance(speech_token_tmp, list):
                    speech_token_tmp = []
                    print(f"error: speech_token is not a list, {speech_token_tmp}")
                    continue
                speech_token = [int(x) for x in speech_token_tmp]
                if len(speech_token) == 0:
                    utils_file.logging_warning(f"error: speech_token is empty,task: {task_name}")
                    continue
                sample['speech_token'] = [speech_token_num - 1] + speech_token + [speech_token_num - 1]
            elif "speech_token" in sample:
                speech_token_tmp = sample['speech_token']
                if not isinstance(speech_token_tmp, list):
                    speech_token_tmp = []
                    print(f"error: speech_token is not a list, {speech_token_tmp}")
                    continue
                speech_token = [int(x) for x in speech_token_tmp]
                if len(speech_token) == 0:
                    utils_file.logging_warning(f"error: speech_token is empty,task: {task_name}")
                    continue
                sample['speech_token'] = [speech_token_num - 1] + speech_token + [speech_token_num - 1]
            else:
                utils_file.logging_warning(f"error: speech_token is empty,task: {task_name}")
                continue
        else:
            sample['speech_token'] = []
        # ========================处理speech token 结束 ==========================



        # =====================处理output_type======================
        # tts
        if task_name == "<TEXT2TOKEN>":
            sample['output_type'] = "text2token"
        elif task_name == "<TEXT2TOKEN> <STREAMING>":
            sample['output_type'] = "text2token_streaming"
        elif task_name == "<S2TCHAT> <TEXT2TOKEN>" or task_name == "<S2TCHAT> <TEXT2TOKEN> <EMOTION>":
            sample['output_type'] = 'speech2text_token'
        elif task_name == "<S2TCHAT> <TEXT2TOKEN> <STREAMING>":
            sample['output_type'] = 'speech2text_token_streaming'
        elif task_name == "<S2TCHAT> <TEXT2TOKEN> <THINK>":
            sample['output_type'] = 'speech2text_token_think'
        elif task_name == "<S2TCHAT> <TEXT2TOKEN> <HISTORY>":
            sample['output_type'] = 'speech2text_token_history'
        elif task_name == "<TEXT2TEXT>":
            sample['output_type'] = 'text2text'
        elif task_name == "<S2TCHAT>":
            sample['output_type'] ='s2t_chat'
        elif task_name == "<S2TCHAT_FAKE>":
            sample['output_type'] ='s2t_chat_fake'
        elif task_name == "<S2TCHAT> <THINKER>":
            sample['output_type'] ='s2t_chat_think'
        else:
           sample['output_type'] = 'text'
        # utils_file.logging_limit_print(f"output_type: {sample['output_type']}")
            # s2t end
        # =====================处理output_type 结束======================
        yield sample


def filter(data,
           max_length=1200,
           min_length=0,
           token_max_length=250,
           token_min_length=1,
           min_output_input_ratio=0.00005,
           max_output_input_ratio=1,
           filter_no_extra_info: bool = False,
           max_seq_len=1000,
           other_filter_conf={}):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        try:
            assert 'sample_rate' in sample
            assert 'wav' in sample
            assert 'label' in sample
        except:
            utils_file.logging_error(f'sample_rate or wav or label not in sample,')
            continue

        output_type = sample["output_type"]
        if other_filter_conf.get("only_s2s", False):
            if output_type not in ['speech2text_token', 'speech2text_token_streaming', 'speech2text_token_think', 'speech2text_token_history']:
                utils_file.logging_error(
                    f"only_s2s, output_type is not speech2text_token or speech2text_token_streaming,speech2text_token_think,speech2text_token_history, continue, output_type: {output_type}")
                continue
        if other_filter_conf.get("only_s2t", False):
            if output_type not in ["text", 's2t_chat', "s2t_chat_fake", "s2t_chat_think"]:
                utils_file.logging_error(
                    f"only_s2t, output_type is not s2t, continue, output_type: {output_type}")
                continue
        if other_filter_conf.get("only_t2t", False):
            if output_type != 'text2text':
                utils_file.logging_error(
                    f"only_t2t, output_type is not text2text, continue, output_type: {output_type}")
                continue
        if other_filter_conf.get("only_t2s", False):
            if output_type not in ['text2token', 'text2token_streaming']:
                utils_file.logging_error(
                    f"only_t2s, output_type is not text2token or text2token_streaming, continue, output_type: {output_type}, sampl  e: {sample}")
                continue

        # 过滤不当文字wav比例
        if "speech_token" in sample and sample["output_type"] not in ['text','text2text', 's2t_chat', 's2t_chat_fake', 's2t_chat_think']:
            if len(sample['label']) * 0.8 >= len(sample['speech_token']):
                utils_file.logging_error(f"label 长度过长,和token长度不匹配，continue, len(sample['label']):{len(sample['label'])}, len(sample['speech_token']):{len(sample['speech_token'])},  task: {sample['task']}, output_type: {sample['output_type']}")
                continue
            # if len(sample['label'])>=5 and len(sample['speech_token']) > 125 and len(sample['label']) * 8.33 < len(sample['speech_token']): # 5s以上的音频，label长度大于5，限制用每秒至少3个文字
            #     utils_file.logging_error(f"label 长度过短,和token长度不匹配，continue, len(sample['label']):{len(sample['label'])}, len(sample['speech_token']):{len(sample['speech_token'])},len(sample['label']) * 8.33 < len(sample['speech_token'])")
            #     continue
            # elif len(sample['label']) * 10 < len(sample['speech_token']):
            #     utils_file.logging_error(f"label 长度过长,和token长度不匹配，continue, len(sample['label']):{len(sample['label'])}, len(sample['speech_token']):{len(sample['speech_token'])},len(sample['label']) * 10 < len(sample['speech_token'])")
            #     continue

        txt = sample['txt']
        if txt == "None_in_extract_answer":
            utils_file.logging_error(
                f'error , txt is None, continue, old txt: {sample["txt"]}, task: {sample["task"]}')
            continue
        if txt == "None_in_extract_X":
            utils_file.logging_error(
                f'error , txt is None, continue, old txt: {sample["txt"]}, task: {sample["task"]}')
            continue
            # if txt == "<None>" or txt == "<NONE>" or txt == "<none>" or txt == "None" or txt == "none" or txt == "NONE":
        if other_filter_conf.get("fiter_txt_is_None", False):
            if "<None>" in txt or "<NONE>" in txt or "<none>" in txt:
                utils_file.logging_error(
                    f'error , txt is None, continue, old txt: {sample["txt"]}, task: {sample["task"]}')
                continue

        history_list = sample.get('history', [])
        if "-1" in history_list:
            history_list = []
            sample['history'] = history_list
        history_len = 0
        history_err = False
        for item in history_list:
            if item['wav'] is None or isinstance(item['wav'], str):
                history_err = True
                break
            wav_len_itm = item['wav'].size(0)
            txt_len_itm = len(item['txt'])
            history_len += wav_len_itm + txt_len_itm
        if history_err:
            utils_file.logging_error(f"error: history_list item['wav'] is None, {sample}, continue")
            continue

        # sample['wav'] is torch.Tensor, we have 100 frames every second
        num_frames = sample['wav'].size(1) / sample['sample_rate'] * 100 + history_len

        # filter for shard_in_common
        if filter_no_extra_info:
            # if 'lang' not in sample:
            #     utils_file.logging_limit_print(f'filter_no_extra_info, lang not in sample, ')
            #     continue
            if 'task' not in sample:
                utils_file.logging_error(f'filter_no_extra_info, task not in sample, ')
                continue

        if num_frames < min_length:
            utils_file.logging_info(
                f'num_frames < min_length, continue, num_frames: {num_frames}, min_length: {min_length}, ')
            continue

        if num_frames > max_length:
            # continue
            if 'task' in sample and sample['task'] == '<CAPTION>':
                utils_file.logging_error(
                    f'num_frames > max_length, continue, num_frames: {num_frames}, max_length: {max_length}, ')
                continue
                # utils_file.logging_limit_print('进行了随机剪裁')
                # 随机选择一个起始点进行裁剪
                # start_frame = random.randint(0, int(num_frames - max_length))
                # end_frame = start_frame + max_length
                # sample['wav'] = sample['wav'][:, int(start_frame / 100 * sample['sample_rate']): int(
                #     end_frame / 100 * sample['sample_rate'])]
                # utils_file.logging_limit_print('sample[', sample['wav'].shape)
            else:
                utils_file.logging_error(
                    f'num_frames > max_length, continue, num_frames: {num_frames}, max_length: {max_length}, ')
                continue
        if len(sample['label']) < token_min_length:
            utils_file.logging_error(
                f'len(sample["label"]) < token_min_length, continue, len(sample["label"]): {len(sample["label"])}, token_min_length: {token_min_length}, ')
            continue
        if len(sample['label']) > token_max_length:
            utils_file.logging_error(
                f'len(sample["label"]) > token_max_length, continue, len(sample["label"]): {len(sample["label"])}, token_max_length: {token_max_length}, ')
            continue
        # if output_type=="text2text" and len(sample['prompt']) > token_max_length:
        #     utils_file.logging_limit_print(
        #         f'len(sample["label"]) > token_max_length, continue, len(sample["label"]): {len(sample["label"])}, token_max_length: {token_max_length}, ')
        #     continue
        # if num_frames != 0:
        #     if len(sample['label']) / num_frames < min_output_input_ratio:
        #         continue
        #     if len(sample['label']) / num_frames > max_output_input_ratio:
        #         continue


        if sample["output_type"] == "speech2text_token" or sample["output_type"] == "speech2text_token_streaming" or sample["output_type"] == "speech2text_token_think" or sample["output_type"] == "speech2text_token_history":
            seq_len = len(sample['prompt']) + num_frames / 8 + len(sample['label']) + len(sample['speech_token'])
        elif sample["output_type"] == "text2token" or sample["output_type"] == "text2token_streaming":
            seq_len = len(sample['prompt']) + len(sample['label']) + len(sample['speech_token'])
        else:
            seq_len = len(sample['prompt']) + num_frames / 8 + len(sample['label'])
        # utils_file.logging_limit_print(f'seqlen: {seq_len}, output_type:{sample["output_type"]},len(sample["prompt"]):{len(sample["prompt"])},num_frames / 8:{num_frames / 8},len(sample["label"]):{len(sample["label"])},len(sample["speech_token"]):{len(sample["speech_token"])} ')
        # for instruct llm
        seq_len = seq_len + 29*2 + history_len
        if 0 < max_seq_len < seq_len:
            utils_file.logging_error(f"seqlen: {seq_len} 超过了最大长度:{max_seq_len}，contiune")
            continue
        # utils_file.logging_limit_print(f'filter yield, task_name: {sample["task"]}, prompt:{sample["prompt"]}, label:{txt}, seq_len:{seq_len}')
        yield sample


def resample(data, resample_rate=16000):
    """ Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            sample['wav'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        yield sample


def speed_perturb(data, speeds=None):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            speeds(List[float]): optional speed

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    if speeds is None:
        speeds = [0.9, 1.0, 1.1]
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        speed = random.choice(speeds)
        if speed != 1.0:
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speed)], ['rate', str(sample_rate)]])
            sample['wav'] = wav

        yield sample


def compute_fbank(data,
                  num_mel_bins=23,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither,
                          energy_floor=0.0,
                          sample_frequency=sample_rate)
        sample['feat'] = mat
        yield sample


def compute_mfcc(data,
                 num_mel_bins=23,
                 frame_length=25,
                 frame_shift=10,
                 dither=0.0,
                 num_ceps=40,
                 high_freq=0.0,
                 low_freq=20.0):
    """ Extract mfcc

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.mfcc(waveform,
                         num_mel_bins=num_mel_bins,
                         frame_length=frame_length,
                         frame_shift=frame_shift,
                         dither=dither,
                         num_ceps=num_ceps,
                         high_freq=high_freq,
                         low_freq=low_freq,
                         sample_frequency=sample_rate)
        sample['feat'] = mat
        yield sample


def do_compute_log_mel_spectrogram(waveform,n_fft=400,
                                hop_length=160,
                                num_mel_bins=80,
                                padding=0):
    waveform = waveform.squeeze(0)  # (channel=1, sample) -> (sample,)
    # utils_file.logging_limit_print(f'wavform shape: {waveform.shape}')
    try:
        if padding > 0:
            waveform = F.pad(waveform, (0, padding))
        window = torch.hann_window(n_fft)
        stft = torch.stft(waveform,
                          n_fft,
                          hop_length,
                          window=window,
                          return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        filters = torch.from_numpy(
            librosa.filters.mel(sr=16000,
                                n_fft=n_fft,
                                n_mels=num_mel_bins))
        mel_spec = filters @ magnitudes

        # NOTE(xcsong): https://github.com/openai/whisper/discussions/269
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        feat = log_spec.transpose(0, 1)
        return feat
    except Exception as e:
        utils_file.logging_error(f'do_compute_log_mel_spectrogram error: {e}')
        return None
def compute_log_mel_spectrogram(data,
                                n_fft=400,
                                hop_length=160,
                                num_mel_bins=80,
                                padding=0):
    """ Extract log mel spectrogram, modified from openai-whisper, see:
        - https://github.com/openai/whisper/blob/main/whisper/audio.py
        - https://github.com/wenet-e2e/wenet/pull/2141#issuecomment-1811765040

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav'].squeeze(0)  # (channel=1, sample) -> (sample,)
        # utils_file.logging_limit_print(f'wavform shape: {waveform.shape}')

        if len(waveform) < n_fft:
            utils_file.logging_error(f"Audio too short: {len(waveform)} < {n_fft}")
            continue

        try:
            if padding > 0:
                waveform = F.pad(waveform, (0, padding))
            window = torch.hann_window(n_fft)
            stft = torch.stft(waveform,
                              n_fft,
                              hop_length,
                              window=window,
                              return_complex=True)
            magnitudes = stft[..., :-1].abs() ** 2

            filters = torch.from_numpy(
                librosa.filters.mel(sr=sample_rate,
                                    n_fft=n_fft,
                                    n_mels=num_mel_bins))
            mel_spec = filters @ magnitudes

            # NOTE(xcsong): https://github.com/openai/whisper/discussions/269
            log_spec = torch.clamp(mel_spec, min=1e-10).log10()
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
            log_spec = (log_spec + 4.0) / 4.0
            sample['feat'] = log_spec.transpose(0, 1)
            # utils_file.logging_limit_print(f'compute_log_mel_spectrogram yield, feat shape: {sample["feat"].shape}')
        except Exception as e:
            utils_file.logging_info(f'compute_log_mel_spectrogram error: {e}, continue, sample: {sample}')
            continue

        # utils_file.logging_limit_print(f'compute_log_mel_spectrogram yield, feat shape: {sample["feat"].shape}')
        yield sample


def spec_aug(data, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10, max_w=80):
    """ Do spec augmentation
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            max_w: max width of time warp

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        max_freq = y.size(1)
        # time mask
        for i in range(num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for i in range(num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        sample['feat'] = y
        yield sample


def spec_sub(data, max_t=20, num_t_sub=3):
    """ Do spec substitute
        Inplace operation
        ref: U2++, section 3.2.3 [https://arxiv.org/abs/2106.05642]

        Args:
            data: Iterable[{key, feat, label}]
            max_t: max width of time substitute
            num_t_sub: number of time substitute to apply

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        for i in range(num_t_sub):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            # only substitute the earlier time chosen randomly for current time
            pos = random.randint(0, start)
            y[start:end, :] = x[start - pos:end - pos, :]
        sample['feat'] = y
        yield sample


def spec_trim(data, max_t=20):
    """ Trim tailing frames. Inplace operation.
        ref: TrimTail [https://arxiv.org/abs/2211.00522]

        Args:
            data: Iterable[{key, feat, label}]
            max_t: max width of length trimming

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        max_frames = x.size(0)
        length = random.randint(1, max_t)
        if length < max_frames / 2:
            y = x.clone().detach()[:max_frames - length]
            sample['feat'] = y
        yield sample


def shuffle(data, shuffle_size=10000):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, feat, label}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            data: Iterable[{key, feat, label}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{key, feat, label}]
    """

    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: x['feat'].size(0))
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: x['feat'].size(0))
    for x in buf:
        yield x


def static_batch(data, batch_size=16):
    """ Static batch the data by `batch_size`

        Args:
            data: Iterable[{key, feat, label}]
            batch_size: batch size

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


# def dynamic_batch(data, max_frames_in_batch=12000, max_seq_in_batch=10000000):
#     """ Dynamic batch the data until the total frames in batch
#         reach `max_frames_in_batch`

#         Args:
#             data: Iterable[{key, feat, label}]
#             max_frames_in_batch: max_frames in one batch

#         Returns:
#             Iterable[List[{key, feat, label}]]
#     """
#     buf = []
#     longest_frames = 0
#     longest_seq = 0
#     max_frames_in_batch = max_frames_in_batch

#     # buf_speech_token = []
#     # longest_frames_token = 0
#     # longest_seq_token = 0
#     # max_frames_in_batch_token = int(max_frames_in_batch)

#     # buf_speech_token_with_text = []
#     # longest_frames_token_with_text = 0
#     # longest_seq_token_with_text = 0
#     # max_frames_in_batch_token_with_text = max_frames_in_batch

#     # buf_no_prompt = []
#     # longest_frames_no_prompt = 0
#     # longest_seq_no_prompt = 0
#     # max_frames_in_batch_no_prompt = int(max_frames_in_batch) # 没有prompt的放在一起

#     for sample in data:
#         # utils_file.logging_limit_print(f'sample in dynamic_batch: {sample}')
#         assert 'feat' in sample, f'feat not in '
#         assert isinstance(sample['feat'], torch.Tensor), f'feat is not tensor: {sample}'
#         new_sample_frames = sample['feat'].size(0)
#         new_seq = sample['feat'].size(0) / 8 + len(sample['label']) + len(sample.get('prompt', [])) + 300
#         longest_seq = max(longest_seq, new_seq)
#         longest_frames = max(longest_frames, new_sample_frames)
#         frames_after_padding = longest_frames * (len(buf)+1)
#         seq_after_padding = longest_seq * (len(buf)+1)
#         if frames_after_padding > max_frames_in_batch or seq_after_padding > max_seq_in_batch:
#             utils_file.logging_limit_print('dynamic batch yield')
#             yield buf
#             buf = [sample]
#             longest_frames = new_sample_frames
#             longest_seq = new_seq
#         else:
#             buf.append(sample)
#     if len(buf) > 0:
#         utils_file.logging_limit_print(f'dynamic batch yield last')
#         yield buf


def dynamic_batch(data, max_frames_in_batch=12000, max_seq_in_batch=10000000):
    """ Dynamic batch the data until the total frames in batch
        reach `max_frames_in_batch`

        Args:
            data: Iterable[{key, feat, label}]
            max_frames_in_batch: max_frames in one batch

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    longest_frames = 0
    longest_seq = 0
    max_frames_in_batch = max_frames_in_batch

    buf_s2s_streaming = [] # for speech 2 text token streaming
    longest_frames_s2s_streaming = 0
    longest_seq_s2s_streaming = 0
    max_frames_in_batch_s2s_streaming = int(max_frames_in_batch)

    buf_s2s_think = []  # for speech 2 text token think
    longest_frames_s2s_think = 0
    longest_seq_s2s_think = 0
    max_frames_in_batch_s2s_think = int(max_frames_in_batch)

    buf_s2s_history = []  # for speech 2 text token think
    longest_frames_s2s_history = 0
    longest_seq_s2s_history = 0
    max_frames_in_batch_s2s_history = int(max_frames_in_batch)

    buf_speech_token_s2s = []  # for speech 2 text token
    longest_frames_token_s2s = 0
    longest_seq_token_s2s = 0
    max_frames_in_batch_token_s2s = int(max_frames_in_batch)

    buf_speech_token_with_text = []
    longest_frames_token_with_text = 0
    longest_seq_token_with_text = 0
    max_frames_in_batch_token_with_text = max_frames_in_batch

    buf_speech_token_with_text_streaming = []
    longest_frames_token_with_text_streaming = 0
    longest_seq_token_with_text_streaming = 0
    max_frames_in_batch_token_with_text_streaming = max_frames_in_batch

    buf_t2t = []
    longest_frames_t2t = 0
    longest_seq_t2t = 0
    max_frames_in_batch_t2t = int(max_frames_in_batch)

    buf_no_prompt = []
    longest_frames_no_prompt = 0
    longest_seq_no_prompt = 0
    max_frames_in_batch_no_prompt = int(max_frames_in_batch)  # 没有prompt的放在一起

    # s2t_chat
    buf_s2t_chat = []
    longest_frames_s2t_chat = 0
    longest_seq_s2t_chat = 0
    max_frames_in_batch_s2t_chat = int(max_frames_in_batch)  # 没有prompt的放在一起

    # s2t_chat_fake
    buf_s2t_chat_fake = []
    longest_frames_s2t_chat_fake = 0
    longest_seq_s2t_chat_fake = 0
    max_frames_in_batch_s2t_chat_fake = int(max_frames_in_batch)  # 没有prompt的放在一起

    # s2t_chat_think
    buf_s2t_chat_think = []
    longest_frames_s2t_chat_think = 0
    longest_seq_s2t_chat_think = 0
    max_frames_in_batch_s2t_chat_think = int(max_frames_in_batch)  # 没有prompt的放在一起

    batch_nums = 0
    # % 4 ->s2t, t2t, t2s, s2s
    for sample in data:
        history_list = sample.get('history', [])
        history_len = 0
        history_err = False
        for item in history_list:
            if item['wav'] is None:
                history_err = True
                break
            wav_len_itm = item['wav'].size(0)
            txt_len_itm = len(item['txt'])
            history_len += wav_len_itm + txt_len_itm
        if history_err:
            utils_file.logging_error(f"error: history_list item['wav'] is None, {sample}, continue")
            continue
        # utils_file.logging_limit_print(f'sample in dynamic_batch: {sample}')
        assert 'feat' in sample
        assert isinstance(sample['feat'], torch.Tensor)
        new_sample_frames = sample['feat'].size(0) + history_len
        if "output_type" in sample and sample["output_type"] == "speech2text_token":
            new_seq = sample['feat'].size(0) / 8 + len(sample['label']) + len(sample.get('prompt', [])) + len(
                sample.get('speech_token', []))
            # for instruct llm
            new_seq = new_seq + 29 *2  + history_len
            longest_seq_token_s2s = max(longest_seq_token_s2s, new_seq)
            longest_frames_token_s2s = max(longest_frames_token_s2s, new_sample_frames)
            frames_after_padding_token_s2s = longest_frames_token_s2s * (len(buf_speech_token_s2s) + 1)
            seq_after_padding_token_s2s = longest_seq_token_s2s * (len(buf_speech_token_s2s) + 1)
            if frames_after_padding_token_s2s > max_frames_in_batch_token_s2s or seq_after_padding_token_s2s > max_seq_in_batch * 0.8:
                yield buf_speech_token_s2s
                buf_speech_token_s2s = [sample]
                longest_frames_token_s2s = new_sample_frames
                longest_seq_token_s2s = new_seq
            else:
                buf_speech_token_s2s.append(sample)
        elif "output_type" in sample and sample["output_type"] =="speech2text_token_streaming":
            new_seq = sample['feat'].size(0) / 8 + len(sample['label']) + len(sample.get('prompt', [])) + len(
                sample.get('speech_token', []))
            # for instruct llm
            new_seq = new_seq + 29 *2  + history_len
            longest_seq_s2s_streaming = max(longest_seq_s2s_streaming, new_seq)
            longest_frames_s2s_streaming = max(longest_frames_s2s_streaming, new_sample_frames)
            frames_after_padding_token = longest_frames_s2s_streaming * (len(buf_s2s_streaming) + 1)
            seq_after_padding_token = longest_seq_s2s_streaming * (len(buf_s2s_streaming) + 1)
            if frames_after_padding_token > max_frames_in_batch_s2s_streaming or seq_after_padding_token > max_seq_in_batch * 0.6:
                yield buf_s2s_streaming
                buf_s2s_streaming = [sample]
                longest_frames_s2s_streaming = new_sample_frames
                longest_seq_s2s_streaming = new_seq
            else:
                buf_s2s_streaming.append(sample)
        elif "output_type" in sample and sample["output_type"] == "speech2text_token_think":
            new_seq = sample['feat'].size(0) / 8 + len(sample['label']) + len(sample.get('prompt', [])) + len(
                sample.get('speech_token', []))
            # for instruct llm
            new_seq = new_seq + 29 * 2 + history_len
            longest_seq_s2s_think = max(longest_seq_s2s_think, new_seq)
            longest_frames_s2s_think = max(longest_frames_s2s_think, new_sample_frames)
            frames_after_padding_token = longest_frames_s2s_think * (len(buf_s2s_think) + 1)
            seq_after_padding_token = longest_seq_s2s_think * (len(buf_s2s_think) + 1)
            if frames_after_padding_token > max_frames_in_batch_s2s_think or seq_after_padding_token > max_seq_in_batch * 0.6:
                yield buf_s2s_think
                buf_s2s_think = [sample]
                longest_frames_s2s_think = new_sample_frames
                longest_seq_s2s_think = new_seq
            else:
                buf_s2s_think.append(sample)
        elif "output_type" in sample and sample["output_type"] == "speech2text_token_history":
            new_seq = sample['feat'].size(0) / 8 + len(sample['label']) + len(sample.get('prompt', [])) + len(
                sample.get('speech_token', []))
            # for instruct llm
            new_seq = new_seq + 29 * 2 + history_len
            longest_seq_s2s_history = max(longest_seq_s2s_history, new_seq)
            longest_frames_s2s_history = max(longest_frames_s2s_history, new_sample_frames)
            frames_after_padding_token = longest_frames_s2s_history * (len(buf_s2s_history) + 1)
            seq_after_padding_token = longest_seq_s2s_history * (len(buf_s2s_history) + 1)
            if frames_after_padding_token > max_frames_in_batch_s2s_history or seq_after_padding_token > max_seq_in_batch * 0.6:
                yield buf_s2s_history
                buf_s2s_history = [sample]
                longest_frames_s2s_history = new_sample_frames
                longest_seq_s2s_history = new_seq
            else:
                buf_s2s_history.append(sample)
        elif "output_type" in sample and sample["output_type"] == "text2token":
            new_seq = len(sample['label']) + len(sample.get('prompt', [])) + len(
                sample.get('speech_token', []))
            # for instruct llm
            new_seq = new_seq + 29 *2  + history_len
            longest_seq_token_with_text = max(longest_seq_token_with_text, new_seq)
            longest_frames_token_with_text = max(longest_frames_token_with_text, new_sample_frames)
            frames_after_padding_token_with_text = longest_frames_token_with_text * (
                        len(buf_speech_token_with_text) + 1)
            seq_after_padding_token_with_text = longest_seq_token_with_text * (len(buf_speech_token_with_text) + 1)
            if frames_after_padding_token_with_text > max_frames_in_batch_token_with_text or seq_after_padding_token_with_text > max_seq_in_batch:
                # utils_file.logging_limit_print('输出了t2s的batch')
                yield buf_speech_token_with_text
                buf_speech_token_with_text = [sample]
                longest_frames_token_with_text = new_sample_frames
                longest_seq_token_with_text = new_seq
            else:
                buf_speech_token_with_text.append(sample)
        elif "output_type" in sample and sample["output_type"] == "text2token_streaming":
            new_seq = len(sample['label']) + len(sample.get('prompt', [])) + len(
                sample.get('speech_token', []))
            # for instruct llm
            new_seq = new_seq + 29 *2  + history_len
            longest_seq_token_with_text_streaming = max(longest_seq_token_with_text_streaming, new_seq)
            longest_frames_token_with_text_streaming = max(longest_frames_token_with_text_streaming, new_sample_frames)
            frames_after_padding_token_with_text_streaming = longest_frames_token_with_text_streaming * (
                        len(buf_speech_token_with_text_streaming) + 1)
            seq_after_padding_token_with_text_streaming = longest_seq_token_with_text_streaming * (len(buf_speech_token_with_text_streaming) + 1)
            if frames_after_padding_token_with_text_streaming > max_frames_in_batch_token_with_text_streaming or seq_after_padding_token_with_text_streaming > max_seq_in_batch:
                # utils_file.logging_limit_print('输出了t2s的batch streaming')
                yield buf_speech_token_with_text_streaming
                buf_speech_token_with_text_streaming = [sample]
                longest_frames_token_with_text_streaming = new_sample_frames
                longest_seq_token_with_text_streaming = new_seq
            else:
                buf_speech_token_with_text_streaming.append(sample)
        elif "output_type" in sample and sample["output_type"] == "text2text":
            new_seq = len(sample['label']) + len(sample.get('prompt', [])) + len(
                sample.get('speech_token', []))
            # for instruct llm
            new_seq = new_seq + 29 *2  + history_len
            longest_seq_t2t = max(longest_seq_t2t, new_seq)
            longest_frames_t2t = max(longest_frames_t2t, new_sample_frames)
            frames_after_padding_t2t = longest_frames_t2t * (len(buf_t2t) + 1)
            seq_after_padding_t2t = longest_seq_t2t * (len(buf_t2t) + 1)
            if frames_after_padding_t2t > max_frames_in_batch_t2t or seq_after_padding_t2t > max_seq_in_batch * 0.6: # t2t没有受到frames限制, 过长的t2t数据引入会导致爆显存，所以给总长度进行限制
                yield buf_t2t
                buf_t2t = [sample]
                longest_frames_t2t = new_sample_frames
                longest_seq_t2t = new_seq
            else:
                buf_t2t.append(sample)
        elif "output_type" in sample and sample["output_type"] == "s2t_chat":
            new_seq = sample['feat'].size(0) / 8 + len(sample['label']) + len(sample.get('prompt', []))
            new_seq = new_seq + 29 *2  + history_len
            longest_seq_s2t_chat = max(longest_seq_s2t_chat, new_seq)
            longest_frames_s2t_chat = max(longest_frames_s2t_chat, new_sample_frames)
            frames_after_padding_s2t_chat = longest_frames_s2t_chat * (len(buf_s2t_chat) + 1)
            seq_after_padding_s2t_chat = longest_seq_s2t_chat * (len(buf_s2t_chat) + 1)
            if frames_after_padding_s2t_chat > max_frames_in_batch_s2t_chat or seq_after_padding_s2t_chat > max_seq_in_batch:
                yield buf_s2t_chat
                buf_s2t_chat = [sample]
                longest_frames_s2t_chat = new_sample_frames
                longest_seq_s2t_chat = new_seq
            else:
                buf_s2t_chat.append(sample)
        elif "output_type" in sample and sample["output_type"] == "s2t_chat_fake":
            new_seq = sample['feat'].size(0) / 8 + len(sample['label']) + len(sample.get('prompt', []))
            new_seq = new_seq + 29 *2  + history_len
            longest_seq_s2t_chat_fake = max(longest_seq_s2t_chat_fake, new_seq)
            longest_frames_s2t_chat_fake = max(longest_frames_s2t_chat_fake, new_sample_frames)
            frames_after_padding_s2t_chat_fake = longest_frames_s2t_chat_fake * (len(buf_s2t_chat_fake) + 1)
            seq_after_padding_s2t_chat_fake = longest_seq_s2t_chat_fake * (len(buf_s2t_chat_fake) + 1)
            if frames_after_padding_s2t_chat_fake > max_frames_in_batch_s2t_chat_fake or seq_after_padding_s2t_chat_fake > max_seq_in_batch:
                yield buf_s2t_chat_fake
                buf_s2t_chat_fake = [sample]
                longest_frames_s2t_chat_fake = new_sample_frames
                longest_seq_s2t_chat_fake = new_seq
            else:
                buf_s2t_chat_fake.append(sample)
        elif "output_type" in sample and sample["output_type"] == "s2t_chat_think":
            new_seq = sample['feat'].size(0) / 8 + len(sample['label']) + len(sample.get('prompt', []))
            new_seq = new_seq + 29 *2  + history_len
            longest_seq_s2t_chat_think = max(longest_seq_s2t_chat_think, new_seq)
            longest_frames_s2t_chat_think = max(longest_frames_s2t_chat_think, new_sample_frames)
            frames_after_padding_s2t_chat_think = longest_frames_s2t_chat_think * (len(buf_s2t_chat_think) + 1)
            seq_after_padding_s2t_chat_think = longest_seq_s2t_chat_think * (len(buf_s2t_chat_think) + 1)
            if frames_after_padding_s2t_chat_think > max_frames_in_batch_s2t_chat_think or seq_after_padding_s2t_chat_think > max_seq_in_batch:
                yield buf_s2t_chat_think
                buf_s2t_chat_think = [sample]
                longest_frames_s2t_chat_think = new_sample_frames
                longest_seq_s2t_chat_think = new_seq
            else:
                buf_s2t_chat_think.append(sample)
        else:
            if len(sample.get('prompt', [])) == 0:
                # 没有prompt的text任务的放在一起
                new_seq = sample['feat'].size(0) / 8 + len(sample['label']) + len(sample.get('prompt', []))
                # for instruct llm
                new_seq = new_seq + 29 *2  + history_len
                longest_seq_no_prompt = max(longest_seq_no_prompt, new_seq)
                longest_frames_no_prompt = max(longest_frames_no_prompt, new_sample_frames)
                frames_after_padding_no_prompt = longest_frames * (len(buf_no_prompt) + 1)
                seq_after_padding_no_prompt = longest_seq_no_prompt * (len(buf_no_prompt) + 1)
                if frames_after_padding_no_prompt > max_frames_in_batch_no_prompt or seq_after_padding_no_prompt > max_seq_in_batch:
                    yield buf_no_prompt
                    buf_no_prompt = [sample]
                    longest_frames_no_prompt = new_sample_frames
                    longest_seq_no_prompt = new_seq
                else:
                    buf_no_prompt.append(sample)
            else:
                new_seq = sample['feat'].size(0) / 8 + len(sample['label']) + len(sample.get('prompt', []))
                # for instruct llm
                new_seq = new_seq + 29 *2  + history_len
                longest_seq = max(longest_seq, new_seq)
                longest_frames = max(longest_frames, new_sample_frames)
                frames_after_padding = longest_frames * (len(buf) + 1)
                seq_after_padding = longest_seq * (len(buf) + 1)
                if frames_after_padding > max_frames_in_batch or seq_after_padding > max_seq_in_batch:
                    yield buf
                    buf = [sample]
                    longest_frames = new_sample_frames
                    longest_seq = new_seq
                else:
                    buf.append(sample)
    if len(buf) > 0:
        yield buf


def batch(data, batch_type='static', batch_size=16, max_frames_in_batch=12000, max_seq_in_batch=10000000):
    """ Wrapper for static/dynamic batch
    """
    if batch_type == 'static':
        return static_batch(data, batch_size)
    elif batch_type == 'dynamic':
        return dynamic_batch(data, max_frames_in_batch, max_seq_in_batch=max_seq_in_batch)
    else:
        logging.fatal('Unsupported batch type {}'.format(batch_type))


def padding(data):
    """ Padding the data into training data

        Args:
            data: Iterable[List[{key, feat, label}]]

        Returns:
            Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:
        assert isinstance(sample, list)
        feats_length = torch.tensor([x['feat'].size(0) for x in sample],
                                    dtype=torch.int32)
        order = torch.argsort(feats_length, descending=True)
        feats_lengths = torch.tensor(
            [sample[i]['feat'].size(0) for i in order], dtype=torch.int32)
        sorted_feats = [sample[i]['feat'] for i in order]
        sorted_keys = [sample[i]['key'] for i in order]
        sorted_labels = [
            torch.tensor(sample[i]['label'], dtype=torch.int64) for i in order
        ]
        sorted_speech_tokens = [
            torch.tensor(sample[i]['speech_token'], dtype=torch.int64) for i in order
        ]

        sorted_wavs = [sample[i]['wav'].squeeze(0) for i in order]
        label_lengths = torch.tensor([x.size(0) for x in sorted_labels],
                                     dtype=torch.int32)
        speech_token_lengths = torch.tensor([x.size(0) for x in sorted_speech_tokens],
                                            dtype=torch.int32)
        wav_lengths = torch.tensor([x.size(0) for x in sorted_wavs],
                                   dtype=torch.int32)
        # utils_file.logging_limit_print('------------------')
        # for feat_item in sorted_feats:
        #     utils_file.logging_limit_print(feat_item.shape)
        # utils_file.logging_limit_print('------------------')

        if len(sorted_feats)==0:
            utils_file.logging_info(f'empty feats, output_type')
            continue
        padded_feats = pad_sequence(sorted_feats,
                                    batch_first=True,
                                    padding_value=0)
        padding_labels = pad_sequence(sorted_labels,
                                      batch_first=True,
                                      padding_value=-100)

        padding_speech_tokens = pad_sequence(sorted_speech_tokens,
                                             batch_first=True,
                                             padding_value=-100)
        padded_wavs = pad_sequence(sorted_wavs,
                                   batch_first=True,
                                   padding_value=0)

        sorted_lang = [
            sample[i].get('lang', 'cn') for i in order
        ]

        sorted_speaker = [
            sample[i].get('speaker', 'None') for i in order
        ]

        sorted_emotion = [
            sample[i].get('emotion', 'None') for i in order
        ]
        sorted_gender = [
            sample[i].get('gender', 'None') for i in order
        ]
        # sorted_duration = [
        #     sample[i]['duration'] for i in order
        # ]
        sorted_task = [
            sample[i].get('task', '<TRANSCRIBE>') for i in order
        ]
        sorted_txts = [
            sample[i].get('txt', '') for i in order
        ]

        batch = {
            "keys": sorted_keys,
            "feats": padded_feats,
            "target": padding_labels,
            "feats_lengths": feats_lengths,
            "target_lengths": label_lengths,
            "pcm": padded_wavs,
            "pcm_length": wav_lengths,
            "speech_tokens": padding_speech_tokens,
            "speech_tokens_length": speech_token_lengths,
            "lang": sorted_lang,
            "speaker": sorted_speaker,
            "emotion": sorted_emotion,
            "gender": sorted_gender,
            "task": sorted_task,
            'txts': sorted_txts
        }
        if 'prompt' in sample[0] and len(sample[0]['prompt']) > 0:
            sorted_prompts = [
                torch.tensor(sample[i]['prompt'], dtype=torch.int64
                             ) for i in order
            ]
            prompt_lengths = torch.tensor([x.size(0) for x in
                                           sorted_prompts], dtype=torch.int32)
            padding_prompts = pad_sequence(sorted_prompts,
                                           batch_first=True,
                                           padding_value=-1)
            batch['prompt'] = padding_prompts
            batch['prompt_lengths'] = prompt_lengths

        if 'output_type' in sample[0]:
            batch['output_type'] = sample[0]['output_type']
        else:
            batch['output_type'] = 'text'

        history_batch = []
        for i in order:
            if 'history' in sample[i]:
                history_batch.append(sample[i].get("history",[]))
            else:
                history_batch.append([])
        batch['history'] = history_batch
        if 'extra' in sample[0]:
            batch['extra'] = [sample[i].get('extra', {}) for i in order]
        yield batch
