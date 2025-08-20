import copy
from typing import Optional
from wenet.dataset.dataset import Dataset, BigDataset

from wenet.text.base_tokenizer import BaseTokenizer


def init_asr_dataset(data_type,
                     data_list_file,
                     tokenizer: Optional[BaseTokenizer] = None,
                     conf=None,
                     partition=True):
    return Dataset(data_type, data_list_file, tokenizer, conf, partition)

def init_asr_big_dataset(data_type,
                     data_list_file_s2t,
                     data_list_file_t2s,
                     data_list_file_s2s,
                     data_list_file_t2t,
                     tokenizer: Optional[BaseTokenizer] = None,
                     conf=None,
                     partition=True):
    return BigDataset(data_type, data_list_file_s2t, data_list_file_t2s, data_list_file_s2s, data_list_file_t2t, tokenizer, conf, partition)

def init_dataset(dataset_type,
                 data_type,
                 data_list_file,
                 tokenizer: Optional[BaseTokenizer] = None,
                 conf=None,
                 partition=True,
                 split='train'):
    assert dataset_type in ['asr', 'ssl']

    if split != 'train':
        cv_conf = copy.deepcopy(conf)
        cv_conf['cycle'] = 1
        cv_conf['speed_perturb'] = False
        cv_conf['spec_aug'] = False
        cv_conf['spec_sub'] = False
        cv_conf['spec_trim'] = False
        cv_conf['shuffle'] = False
        cv_conf['list_shuffle'] = False
        conf = cv_conf

    if dataset_type == 'asr':
        return init_asr_dataset(data_type, data_list_file, tokenizer, conf,
                                partition)
    else:
        from wenet.ssl.init_dataset import init_dataset as init_ssl_dataset
        return init_ssl_dataset(data_type, data_list_file, conf, partition)


def init_big_dataset(dataset_type,
                 data_type,
                 data_list_file_s2t,
                 data_list_file_t2s,
                 data_list_file_s2s,
                 data_list_file_t2t,
                 tokenizer: Optional[BaseTokenizer] = None,
                 conf=None,
                 partition=True,
                 split='train'):
    assert dataset_type in ['asr', 'ssl']

    if split != 'train':
        cv_conf = copy.deepcopy(conf)
        cv_conf['cycle'] = 1
        cv_conf['speed_perturb'] = False
        cv_conf['spec_aug'] = False
        cv_conf['spec_sub'] = False
        cv_conf['spec_trim'] = False
        cv_conf['shuffle'] = False
        cv_conf['list_shuffle'] = False
        conf = cv_conf

    if dataset_type == 'asr':
        return init_asr_big_dataset(data_type,
                                data_list_file_s2t, data_list_file_t2s, data_list_file_s2s, data_list_file_t2t,
                                tokenizer, conf, partition)
    else:
        pass
        # from wenet.ssl.init_dataset import init_dataset as init_ssl_dataset
        # return init_ssl_dataset(data_type, data_list_file, conf, partition)