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

import logging
import json
import random
import tarfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse

import torch
import torchaudio
from gxl_ai_utils.utils import utils_file

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
            sample.update(stream=stream, file_path=url)
            yield sample
        except Exception as ex:
            utils_file.logging_warning('Failed to open {}'.format(url))


def tar_file_and_group(data, total_num = 0, num_workers=1):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix

        Args:
            data: Iterable[{src, stream}]

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
            :param total_num:
    """
    tar_index = 0
    now =utils_file.do_get_now_time()
    total_num = total_num//num_workers
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        worker_id = 0
    else:
        worker_id = worker_info.id
    for sample in data:
        assert 'stream' in sample
        assert  'file_path' in sample
        # stream = tarfile.open(fileobj=sample['stream'], mode="r|*")
        stream = tarfile.open(fileobj=sample['stream'], mode="r:*")
        file_path = sample['file_path']
        prev_prefix = None
        example = {"file_path": file_path}
        valid = True
        cost_time = utils_file.do_get_elapsed_time(now)
        now = utils_file.do_get_now_time()
        utils_file.logging_info(f'开始处理{file_path}, index: {tar_index}/{total_num}，worker_id: {worker_id}/{num_workers}, 耗时: {cost_time:.2f}s')
        tar_index += 1
        try:
            for tarinfo in stream:
                name = tarinfo.name
                pos = name.rfind('.')
                assert pos > 0
                prefix, postfix = name[:pos], name[pos + 1:]
                if prev_prefix is not None and prefix != prev_prefix:
                    example['key'] = prev_prefix
                    if valid:
                        yield example
                    example = {"file_path": file_path}
                    valid = True
                with stream.extractfile(tarinfo) as file_obj:
                    try:
                        if postfix == 'txt':
                            txt = file_obj.read().decode('utf8').strip()
                            example['txt'] = txt  # .replace(' ', '')
                        elif postfix in AUDIO_FORMAT_SETS:
                            # example['wav'] = file_obj.read()
                            pass
                        elif postfix == 'extra':
                            example['extra'] = file_obj.read().decode('utf8').strip()
                        else:
                            example[postfix] = file_obj.read().decode('utf8').strip()
                    except Exception as ex:
                        valid = False
                        utils_file.logging_warning('error to parse {}'.format(name))
                prev_prefix = prefix
            if prev_prefix is not None:
                example['key'] = prev_prefix
                yield example
            stream.close()
            if 'process' in sample:
                sample['process'].communicate()
            sample['stream'].close()
        except Exception as ex:
            utils_file.logging_warning(f'处理{file_path}失败')
            utils_file.logging_warning('Failed to parse {}'.format(file_path))



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
        txt = obj['txt']
        wav = obj['wav']
        try:
            example = dict(key=key,
                           wav=wav,
                           txt=txt)
            yield example
        except Exception as ex:
            utils_file.logging_warning('Failed to read {}'.format(wav))



def middle_handler(data,):
    for sample in data:
        # assert 'label' in sample
        # sample['wav'] is torch.Tensor, we have 100 frames every second
        # if len(sample['label']) < token_min_length:
        #     continue
        # if len(sample['label']) > token_max_length:
        #     continue
        origin_extra = sample.get('extra', {})
        if type(origin_extra) == str:
            try:
                # utils_file.logging_limit_print_info(f"origin_extra is a str, try to load it as json")
                sample['extra'] = json.loads(origin_extra)
            except json.JSONDecodeError:
                # utils_file.logging_error("Error: 'extra' is not a valid JSON string.")
                sample['extra'] = {}
        elif type(origin_extra) == dict:
            sample['extra'] = origin_extra
        else:
            sample['extra'] = {}

        if 'age' in sample['extra']:
            age = sample['extra']['age']
            if isinstance(age, str) and not age.startswith('<') and not age.endswith('>'):
                age = '<' + age + '>'
                age = age.upper()
                sample['extra']['age'] = age


        gender = sample.get('gender', '<NONE>')
        if gender in ['male', 'female', 'MALE', 'FEMALE', '<male>', '<female>', '<MALE>', '<FEMALE>', ]:
            if not gender.startswith('<') and not gender.endswith('>'):
                gender = '<' + gender + '>'
            gender = gender.upper()
            sample['extra']['gender'] = gender
        else:
            pass


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
            buf.sort(key=lambda x: len(x['label']) + len(x['wav']))
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: len(x['label']) + len(x['wav']))
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


def dynamic_batch(data, max_frames_in_batch=12000):
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
    for sample in data:
        new_sample_frames = len(sample['label']) + len(sample['wav'])
        longest_frames = max(longest_frames, new_sample_frames)
        frames_after_padding = longest_frames * (len(buf) + 1)
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_sample_frames
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf


def batch(data, batch_type='static', batch_size=16, max_frames_in_batch=12000):
    """ Wrapper for static/dynamic batch
    """
    if batch_type == 'static':
        return static_batch(data, batch_size)
    elif batch_type == 'dynamic':
        return dynamic_batch(data, max_frames_in_batch)
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
        # print_info(sample)
        assert isinstance(sample, list)
        label_lengths = torch.tensor([len(x['txt']) for x in sample],
                                    dtype=torch.int32)
        order = torch.argsort(label_lengths, descending=True)
        sorted_keys = [sample[i]['key'] for i in order]
        # print_info(f'sorted_keys: {sorted_keys}')
        sorted_labels = [sample[i]['txt'] for i in order]
        # print_info(f'sorted_labels: {sorted_labels}')
        # sorted_wavs = [sample[i]['wav'] for i in order]
        sorted_wavs = ["-" for i in order]
        extra_dict_list = [sample[i]['extra'] for i in order]
        sorted_file_path = [sample[i]['file_path'] for i in order]
        batch_dict = {
            'keys': sorted_keys,
            'labels': sorted_labels,
            'wavs': sorted_wavs,
            'extra_dicts': extra_dict_list,
            'tar_files': sorted_file_path
        }
        yield batch_dict
