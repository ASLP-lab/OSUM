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
import random
from typing import List

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

import wenet.dataset.process.processor as processor
from wenet.text.base_tokenizer import BaseTokenizer
from wenet.utils.file_utils import read_lists


class Processor(IterableDataset):

    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:

    def __init__(self, shuffle=True, partition=True, split_num=1,multi_num=1):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition
        self.split_num = split_num
        self.multi_num = multi_num

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def split_data(self, total_num):
        data = list(range(total_num))
        sub_epoch = self.epoch + 1
        full_epoch = sub_epoch // self.split_num
        num_per_sub_epochs = total_num // self.split_num
        random.Random(full_epoch).shuffle(data)

        split_index = sub_epoch - full_epoch * self.split_num
        begin = split_index * num_per_sub_epochs
        end = (begin + num_per_sub_epochs 
                if (split_index + 1) < self.split_num else
                total_num)
        
        # print(f'begin: {begin}, end: {end}, world_size: {self.world_size}')
        return data[begin:end]

    def sample(self, data, split_num=1):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        if self.split_num == 1 and self.multi_num == 1:
            data = list(range(len(data)))
        elif self.split_num != 1:
            assert self.multi_num == 1
            data = self.split_data(len(data))
        else: 
            assert self.split_num ==1
            data = list(range(len(data*self.multi_num)))
        # TODO(Binbin Zhang): fix this
        # We can not handle uneven data for CV on DDP, so we don't
        # sample data by rank, that means every GPU gets the same
        # and all the CV data
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
            # print(f'num dataset: {len(data)}')
        data = data[self.worker_id::self.num_workers]
        self.epoch += 1
        return data

    def pre_sample(self, data, split_num=1):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        if self.split_num == 1 and self.multi_num == 1:
            data = list(range(len(data)))
        elif self.split_num != 1:
            assert self.multi_num == 1
            data = self.split_data(len(data))
        else:
            assert self.split_num ==1
            data = list(range(len(data*self.multi_num)))
        # TODO(Binbin Zhang): fix this
        # We can not handle uneven data for CV on DDP, so we don't
        # sample data by rank, that means every GPU gets the same
        # and all the CV data
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
            # print(f'num dataset: {len(data)}')
        data = data[self.worker_id::self.num_workers]
        return data


class DataList(IterableDataset):

    def __init__(self, lists, shuffle=True, partition=True, split_num=1):
        self.lists = lists
        self.sampler = DistributedSampler(shuffle, partition, split_num)
        self.true_lists = self.sampler.pre_sample(self.lists)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        for index in indexes:
            # yield dict(src=src)
            data = dict(src=self.lists[index])
            data.update(sampler_info)
            yield data
from gxl_ai_utils.utils import utils_file

class BigDataList(IterableDataset):

    def __init__(self,s2t_dataset,t2s_dataset,s2s_dataset,t2t_dataset, weight_num:List[int]):
        self.s2t_dataset = s2t_dataset
        self.t2s_dataset = t2s_dataset
        self.s2s_dataset = s2s_dataset
        self.t2t_dataset = t2t_dataset
        self.batch_index = 0
        self.weight_num = weight_num
        utils_file.logging_info(f"weight_num:{weight_num}")
    def set_epoch(self, epoch):
        self.s2t_dataset.set_epoch(epoch)
        self.t2s_dataset.set_epoch(epoch)
        self.s2s_dataset.set_epoch(epoch)
        self.t2t_dataset.set_epoch(epoch)

    def __iter__(self):
        datasets = [iter(d) for d in [self.s2t_dataset, self.t2s_dataset, self.s2s_dataset, self.t2t_dataset]]
        while True:
            self.batch_index += 1
            selected_iter = self.do_select_iter(datasets)
            try:
                yield next(selected_iter)
            except StopIteration:
                # 移除已耗尽的数据源
                datasets = [it for it in datasets if it is not selected_iter]
                if not datasets:  # 所有数据源耗尽时终止
                    break

    def do_select_iter(self, datasets):
        # 检查各迭代器是否有效（未耗尽）
        valid_indices = [i for i, it in enumerate(datasets) if it is not None]
        if not valid_indices:
            raise StopIteration
        # 保存当前随机状态
        original_state = random.getstate()

        # 临时设置随机种子为batch_index
        random.seed(self.batch_index)
        # 根据weight_num计算有效数据源的权重
        valid_weights = [self.weight_num[i] for i in valid_indices]

        # 按权重随机选择（使用random.choices）
        selected_idx = random.choices(valid_indices, weights=valid_weights, k=1)[0]
        # 恢复原始随机状态
        random.setstate(original_state)
        return datasets[selected_idx]


def get_dataset(data_type,
                data_list_file,
                tokenizer: BaseTokenizer,
                conf,
                partition=True):
    lists = read_lists(data_list_file)
    shuffle = conf.get('shuffle', True)
    split_num = conf.get('split_num', 1)
    multi_num = conf.get('multi_num', 1)
    lists = lists * multi_num
    if_data_recover = conf.get('data_recover', False)
    data_recover_conf = conf.get('data_recover_conf', {})
    if if_data_recover:
        print(f"recover data old list len:{len(lists)}")
        start_idx = data_recover_conf.get('start_idx', 0)
        if start_idx >= len(lists):
            start_idx = 0
        lists = lists[start_idx:]
        print(f"recover data from {start_idx}, new list len:{len(lists)}")
    dataset = DataList(lists, shuffle=shuffle, partition=partition, split_num=split_num)
    true_list = dataset.true_lists
    if data_type == 'shard':
        dataset = Processor(dataset, processor.url_opener)
        dataset = Processor(dataset, processor.tar_file_and_group_full_data, total_num=len(true_list))
    else:
        dataset = Processor(dataset, processor.parse_raw)

    speaker_conf = conf.get('speaker_conf', None)
    if speaker_conf is not None:
        dataset = Processor(dataset, processor.parse_speaker, **speaker_conf)

    if conf.get('eod_id', None) is not None:
        tokenizer.eod_id = conf['eod_id']
    # prompt dict
    from gxl_ai_utils.utils import utils_file
    other_tokenze_conf = conf.get('other_tokenze_conf', {})
    global_prompt_dict = utils_file.load_dict_from_yaml(conf.get('prompt_conf_path', "conf/promp,t_config.yaml"))
    speech_token_num = conf.get('speech_token_num', 1)
    dataset = Processor(dataset, processor.tokenize, tokenizer, other_tokenze_conf=other_tokenze_conf,
                        global_prompt_dict=global_prompt_dict, speech_token_num=speech_token_num)
    filter_conf = conf.get('filter_conf', {})
    dataset = Processor(dataset, processor.filter, **filter_conf)

    resample_conf = conf.get('resample_conf', {})
    dataset = Processor(dataset, processor.resample, **resample_conf)

    speed_perturb = conf.get('speed_perturb', False)
    if speed_perturb:
        dataset = Processor(dataset, processor.speed_perturb)

    feats_type = conf.get('feats_type', 'fbank')
    assert feats_type in ['fbank', 'mfcc', 'log_mel_spectrogram']
    if feats_type == 'fbank':
        fbank_conf = conf.get('fbank_conf', {})
        dataset = Processor(dataset, processor.compute_fbank, **fbank_conf)
    elif feats_type == 'mfcc':
        mfcc_conf = conf.get('mfcc_conf', {})
        dataset = Processor(dataset, processor.compute_mfcc, **mfcc_conf)
    elif feats_type == 'log_mel_spectrogram':
        log_mel_spectrogram_conf = conf.get('log_mel_spectrogram_conf', {})
        dataset = Processor(dataset, processor.compute_log_mel_spectrogram,
                            **log_mel_spectrogram_conf)

    spec_aug = conf.get('spec_aug', True)
    spec_sub = conf.get('spec_sub', False)
    spec_trim = conf.get('spec_trim', False)
    if spec_aug:
        spec_aug_conf = conf.get('spec_aug_conf', {})
        dataset = Processor(dataset, processor.spec_aug, **spec_aug_conf)
    if spec_sub:
        spec_sub_conf = conf.get('spec_sub_conf', {})
        dataset = Processor(dataset, processor.spec_sub, **spec_sub_conf)
    if spec_trim:
        spec_trim_conf = conf.get('spec_trim_conf', {})
        dataset = Processor(dataset, processor.spec_trim, **spec_trim_conf)
    # for emotion-only task
    # dataset = Processor(dataset, processor.add_ssl_vec)
    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = Processor(dataset, processor.shuffle, **shuffle_conf)

    sort = conf.get('sort', True)
    if sort:
        sort_conf = conf.get('sort_conf', {})
        dataset = Processor(dataset, processor.sort, **sort_conf)

    batch_conf = conf.get('batch_conf', {})
    dataset = Processor(dataset, processor.batch, **batch_conf)
    dataset = Processor(dataset, processor.padding)
    return dataset

def do_get_fake_file():
    temp_path = f'~/.cache/.temp/{random.randint(10000, 99999)}.txt'
    utils_file.makedir_for_file(temp_path)
    return temp_path

def BigDataset(data_type,
            data_list_file_s2t,
            data_list_file_t2s,
            data_list_file_s2s,
            data_list_file_t2t,
            tokenizer: BaseTokenizer,
            conf,
            partition=True):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shard tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_type(str): raw/shard
            bpe_model(str): model for english bpe part
            partition(bool): whether to do data partition in terms of rank
    """
    assert data_type in ['raw', 'shard']
    # 深度复制conf
    s2t_conf = copy.deepcopy(conf)
    s2t_conf['other_tokenze_conf']["use_s2s_convert_s2t"]['enable'] = True
    s2t_conf['filter_conf']['other_filter_conf']['only_s2t'] = True
    s2t_conf['other_tokenze_conf']["only_info"]["only_s2t"] = True

    t2s_conf = copy.deepcopy(conf)
    t2s_conf['filter_conf']['other_filter_conf']['only_t2s'] = True
    t2s_conf['other_tokenze_conf']["only_info"]['only_t2s'] = True
    s2s_conf = copy.deepcopy(conf)
    s2s_conf['filter_conf']['other_filter_conf']['only_s2s'] = True
    s2s_conf['other_tokenze_conf']["only_info"]['only_s2s'] = True
    t2t_conf = copy.deepcopy(conf)
    t2t_conf['filter_conf']['other_filter_conf']['only_t2t'] = True
    t2t_conf['other_tokenze_conf']["only_info"]['only_t2t'] = True

    tmp_file_s2t = do_get_fake_file()
    s2s_list = utils_file.load_list_file_clean(data_list_file_s2s)
    # s2s_list_little = s2s_list[::3]
    s2s_list_little = []
    s2t_list = utils_file.load_list_file_clean(data_list_file_s2t)
    s2t_full_list = s2t_list + s2s_list_little
    utils_file.write_list_to_file(s2t_full_list, tmp_file_s2t)


    s2t_dataset = get_dataset(data_type, tmp_file_s2t, tokenizer, s2t_conf, partition=partition)
    t2s_dataset = get_dataset(data_type, data_list_file_t2s, tokenizer, t2s_conf, partition=partition)
    s2s_dataset = get_dataset(data_type, data_list_file_s2s, tokenizer, s2s_conf, partition=partition)
    t2t_dataset = get_dataset(data_type, data_list_file_t2t, tokenizer, t2t_conf, partition=partition)
    dataset = BigDataList(s2t_dataset, t2s_dataset, s2s_dataset, t2t_dataset,
                          weight_num=[len(read_lists(tmp_file_s2t)),
                                      len(read_lists(data_list_file_t2s)),
                                      len(read_lists(data_list_file_s2s)),
                                      len(read_lists(data_list_file_t2t))
    ])
    return dataset

def Dataset(data_type,
            data_list_file,
            tokenizer: BaseTokenizer,
            conf,
            partition=True):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shard tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_type(str): raw/shard
            bpe_model(str): model for english bpe part
            partition(bool): whether to do data partition in terms of rank
    """
    assert data_type in ['raw', 'shard', 'shard_full_data']
    dataset = get_dataset(data_type, data_list_file, tokenizer, conf, partition=partition)
    return dataset
