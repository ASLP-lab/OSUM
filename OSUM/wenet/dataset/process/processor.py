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

import copy
import librosa
import logging
import json
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
from wenet.text.base_tokenizer import BaseTokenizer

# torchaudio.utils.sox_utils.set_buffer_size(16500)
torchaudio.set_audio_backend("soundfile")

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


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
        try:
            pr = urlparse(url)
            # local file
            if pr.scheme == '' or pr.scheme == 'file':
                stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f'wget -q -O - {url}'
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning('Failed to open {}'.format(url))


def tar_file_and_group(data):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix

        Args:
            data: Iterable[{src, stream}]

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    for sample in data:
        assert 'stream' in sample
        stream = None
        try:
            stream = tarfile.open(fileobj=sample['stream'], mode="r:*")
            prev_prefix = None
            example = {}
            valid = True
            for tarinfo in stream:
                name = tarinfo.name
                pos = name.rfind('.')
                assert pos > 0
                prefix, postfix = name[:pos], name[pos + 1:]
                if prev_prefix is not None and prefix != prev_prefix:
                    example['key'] = prev_prefix
                    if valid:
                        yield example
                    example = {}
                    valid = True
                with stream.extractfile(tarinfo) as file_obj:
                    try:
                        if postfix == 'txt':
                            example['txt'] = file_obj.read().decode(
                                'utf8').strip()
                        elif postfix in AUDIO_FORMAT_SETS:
                            waveform, sample_rate = torchaudio.load(file_obj)
                            example['wav'] = waveform
                            example['sample_rate'] = sample_rate
                        else:
                            example[postfix] = file_obj.read()
                    except Exception as ex:
                        valid = False
                        logging.warning('error to parse {}'.format(name))
                prev_prefix = prefix
            if prev_prefix is not None:
                example['key'] = prev_prefix
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


def tar_file_and_group_full_data(data):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix

        Args:
            data: Iterable[{src, stream}]

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    for sample in data:
        assert 'stream' in sample
        stream = None
        try:
            stream = tarfile.open(fileobj=sample['stream'], mode="r:*")
            prev_prefix = None
            example = {}
            valid = True
            for tarinfo in stream:
                name = tarinfo.name
                pos = name.rfind('.')
                assert pos > 0
                prefix, postfix = name[:pos], name[pos + 1:]
                if prev_prefix is not None and prefix != prev_prefix:
                    example['key'] = prev_prefix
                    if valid:
                        # assert 'txt' in example
                        if 'txt' not in example:
                            example['txt'] = ''
                        yield example
                    example = {}
                    valid = True
                with stream.extractfile(tarinfo) as file_obj:
                    try:
                        if postfix == 'txt':
                            example['txt'] = file_obj.read().decode(
                                'utf8').strip()
                        elif postfix == 'lang':
                            example['lang'] = file_obj.read().decode(
                                'utf8').strip()
                        elif postfix == 'speaker':
                            try:
                                example['speaker'] = file_obj.read().decode(
                                    'utf8').strip()
                            except Exception as ex:
                                example['speaker'] = "none"
                        elif postfix == 'emotion':
                            example['emotion'] = file_obj.read().decode(
                                'utf8').strip()
                        elif postfix == 'gender':
                            example['gender'] = file_obj.read().decode(
                                'utf8').strip()
                        elif postfix == 'task':
                            example['task'] = file_obj.read().decode(
                                'utf8').strip()
                        elif postfix == 'speech_token':
                            example['speech_token'] = file_obj.read()
                        elif postfix == 'duration':
                            duration_str = file_obj.read().decode(
                                'utf8').strip()
                            try:
                                duration_float = float(duration_str)
                                example['duration'] = duration_float
                            except Exception as ex:
                                logging.warning(f'error to parse duration {duration_str}')
                                example['duration'] = 0

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
                            example[postfix] = file_obj.read()
                    except Exception as ex:
                        valid = False
                        # logging.warning('error to parse {}'.format(name))
                prev_prefix = prefix
            if prev_prefix is not None:
                example['key'] = prev_prefix
                if 'txt' in example:
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


def filter(data,
           max_length=1200,
           min_length=10,
           token_max_length=250,
           token_min_length=1,
           min_output_input_ratio=0.00005,
           max_output_input_ratio=1,
           filter_no_extra_info: bool = False,
           max_seq_len=1000):
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
            continue

        txt = sample['txt_new']
        if txt == "None_in_extract_answer":
            print(f'error , txt is None, continue, old txt: {sample["txt"]}, task: {sample["task"]}')
            continue
        if txt == "None_in_extract_X":
            print(f'error , txt is None, continue, old txt: {sample["txt"]}, task: {sample["task"]}')
            continue 
        #if txt == "<None>" or txt == "<NONE>" or txt == "<none>" or txt == "None" or txt == "none" or txt == "NONE":
        if "<None>" in txt or "<NONE>" in txt or "<none>" in txt:
            print(f'error , txt is None, continue, old txt: {sample["txt"]}, task: {sample["task"]}')
            continue

        # sample['wav'] is torch.Tensor, we have 100 frames every second
        num_frames = sample['wav'].size(1) / sample['sample_rate'] * 100

        # filter for shard_in_common
        if filter_no_extra_info:
            if 'task' not in sample:
                continue

        if num_frames < min_length:
            continue
        if num_frames > max_length:
            # continue
            if 'task' in sample and sample['task'] == '<CAPTION>':
                start_frame = random.randint(0, int(num_frames - max_length))
                end_frame = start_frame + max_length
                sample['wav'] = sample['wav'][:, int(start_frame / 100 * sample['sample_rate']): int(
                    end_frame / 100 * sample['sample_rate'])]
            else:
                continue
        if len(sample['label']) < token_min_length:
            continue
        if len(sample['label']) > token_max_length:
            continue
        if sample["output_type"] == "speech2text_token":
            seq_len = len(sample['prompt']) + num_frames / 8 + len(sample['label']) + len(sample['speech_token'])
        elif sample["output_type"] == "text2token":
            seq_len = len(sample['prompt']) + len(sample['label']) + len(sample['speech_token'])
        else:
            seq_len =  len(sample['prompt']) + num_frames / 8 + len(sample['label'])
        # utils_file.logging_limit_print(f'seqlen: {seq_len}, output_type:{sample["output_type"]},len(sample["prompt"]):{len(sample["prompt"])},num_frames / 8:{num_frames / 8},len(sample["label"]):{len(sample["label"])},len(sample["speech_token"]):{len(sample["speech_token"])} ')
        if max_seq_len > 0 and max_seq_len < seq_len:
            # utils_file.logging_limit_print(f"seqlen: {seq_len} 超过了最大长度:{max_seq_len}，contiune")
            continue
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
        # print(f'wavform shape: {waveform.shape}')
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
        yield sample


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


def replace_keys_in_brackets(input_str, key_value_dict):
    for key, value in key_value_dict.items():
        # 构造匹配 <key> 形式的正则表达式模式
        pattern = re.compile(r'<{}>'.format(key))
        input_str = pattern.sub(f"<{value}>", input_str)
    return input_str


def tokenize(data, tokenizer: BaseTokenizer, global_prompt_dict=None):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]

        Returns:
            Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    for sample in data:
        try:
            assert 'txt' in sample
        except:
            print(f'tokenize: {sample}')
            exit()
        if 'task' in sample:
            task_name = sample['task']
            if "<STYLE>" in sample['task']:
                txt = replace_keys_in_brackets(sample['txt'], global_style_dict)
            else:
                txt = sample['txt']
        else:
            txt = sample['txt']

        tokens, label = tokenizer.tokenize(process_text(txt))
        sample['tokens'] = tokens  # token是字符， label是数字
        sample['label'] = label + [tokenizer.eod_id]
        sample['txt_new'] = txt
        if 'task' in sample:
            task_name = sample['task']
            try:
                random_index = random.randint(0, len(global_prompt_dict[task_name]) - 1)
                prompt = global_prompt_dict[task_name][random_index]
                if prompt == "<no_prompt>":
                    utils_file.logging_limit_print(f'no prompt for {task_name}')
                    sample['prompt'] = []
                else:
                    sample['prompt'] = tokenizer.tokenize(prompt)[1]  # labels
            except:
                pass
        else:
            task_name = '<TRANSCRIBE>'
            try:
                random_index = random.randint(0, len(global_prompt_dict[task_name]) - 1)
                prompt = global_prompt_dict[task_name][random_index]
                sample['prompt'] = tokenizer.tokenize(prompt)[1]  # labels
            except:
                pass

        if False and 'speech_token' in sample: # 先不考虑token任务
            old_task_name = sample['task']
            if old_task_name == "<TRANSCRIBE>":
                task_name = '<TEXT2SPEECH_TOKEN>'
                sample['output_type'] = 'text2token'
            elif old_task_name == "<S2TCHAT>":
                task_name = '<SPEECH2TEXT_SPEECH_TOKEN>'
                sample['output_type'] = 'speech2text_token'
            else:
                task_name = old_task_name
            try:
                random_index = random.randint(0, len(global_prompt_dict[task_name]) - 1)
                prompt = global_prompt_dict[task_name][random_index]
                sample['prompt'] = tokenizer.tokenize(prompt)[1]  # labels
            except:
                pass
            speech_token = [int(x) for x in sample['speech_token']]
            sample['speech_token'] = [4096] + speech_token + [4096]
        else:
            sample['output_type'] = 'text'
            sample['speech_token'] = [4096]
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

    buf_speech_token = []
    longest_frames_token = 0
    longest_seq_token = 0
    max_frames_in_batch_token = int(max_frames_in_batch)

    buf_speech_token_with_text = []
    longest_frames_token_with_text = 0
    longest_seq_token_with_text = 0
    max_frames_in_batch_token_with_text = max_frames_in_batch

    buf_no_prompt = []
    longest_frames_no_prompt = 0
    longest_seq_no_prompt = 0
    max_frames_in_batch_no_prompt = int(max_frames_in_batch) # 没有prompt的放在一起

    for sample in data:
        assert 'feat' in sample
        assert isinstance(sample['feat'], torch.Tensor)
        new_sample_frames = sample['feat'].size(0)
        if "output_type" in sample and sample["output_type"] == "speech2text_token":
            new_seq = sample['feat'].size(0) / 8 + len(sample['label']) + len(sample.get('prompt', [])) + len(
                sample.get('speech_token', []))
            longest_seq_token = max(longest_seq_token, new_seq)
            longest_frames_token = max(longest_frames_token, new_sample_frames)
            frames_after_padding_token = longest_frames_token * (len(buf_speech_token)+1)
            seq_after_padding_token = longest_seq_token * (len(buf_speech_token)+1)
            if frames_after_padding_token > max_frames_in_batch_token or seq_after_padding_token > max_seq_in_batch:
                yield buf_speech_token
                buf_speech_token = [sample]
                longest_frames_token = new_sample_frames
                longest_seq_token = new_seq
            else:
                buf_speech_token.append(sample)
        elif "output_type" in sample and sample["output_type"] == "text2token":
            new_seq = len(sample['label']) + len(sample.get('prompt', [])) + len(
                sample.get('speech_token', []))
            longest_seq_token_with_text = max(longest_seq_token_with_text, new_seq)
            longest_frames_token_with_text = max(longest_frames_token_with_text, new_sample_frames)
            frames_after_padding_token_with_text = longest_frames_token_with_text * (len(buf_speech_token_with_text)+1)
            seq_after_padding_token_with_text = longest_seq_token_with_text * (len(buf_speech_token_with_text)+1)
            if frames_after_padding_token_with_text > max_frames_in_batch_token_with_text or seq_after_padding_token_with_text > max_seq_in_batch:
                yield buf_speech_token_with_text
                buf_speech_token_with_text = [sample]
                longest_frames_token_with_text = new_sample_frames
                longest_seq_token_with_text = new_seq
            else:
                buf_speech_token_with_text.append(sample)
        else:
            if len(sample.get('prompt', []))== 0:
                # 没有prompt的text任务的放在一起
                new_seq = sample['feat'].size(0) / 8 + len(sample['label']) + len(sample.get('prompt', []))
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
                longest_seq = max(longest_seq, new_seq)
                longest_frames = max(longest_frames, new_sample_frames)
                frames_after_padding = longest_frames * (len(buf)+1)
                seq_after_padding = longest_seq * (len(buf)+1)
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
        # print('------------------')
        # for feat_item in sorted_feats:
        #     print(feat_item.shape)
        # print('------------------')

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
            "task": sorted_task
        }
        if 'prompt' in sample[0] and len(sample[0]['prompt'])>0:
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

        if 'output_type' in sample[0] and sample[0]['output_type'] == 'speech2text_token':
            batch['output_type'] = 'speech2text_token'
        elif 'output_type' in sample[0] and sample[0]['output_type'] == 'text2token':
            batch['output_type'] = 'text2token'
        else:
            batch['output_type'] = 'text'

        if 'extra' in sample[0]:
            batch['extra'] = [sample[i].get('extra',{}) for i in order]
        yield batch
