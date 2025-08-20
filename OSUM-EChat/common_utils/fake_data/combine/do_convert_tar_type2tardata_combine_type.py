import os

import tqdm
from gxl_ai_utils.utils import utils_file
from gxl_ai_utils.utils.utils_file import do_get_now_time, do_get_file_pure_name_from_path, do_get_elapsed_time, \
    write_dict_list_to_jsonl
from torch.utils.data import DataLoader
import torch
import sys
sys.path.insert(0, 'common_utils/fake_data/combine')
from dataset.dataset_no_wav import Dataset


def do_convert_tartype2tardata_combine_type(shards_list_file, output_dir_path=None, tar_dir=None, task_tag=None, batch_size=50, num_workers=10):
    """
    10 batch , 1 worker : 15.9s, nvidia-smi 在8卡使用
    10batch, 10worker: 16.02s, nvidia-smi 在8卡使用
    10 batch ,10worker , 310tar: 17.46s，
    10 batch 50workt , 310tar: 11.81s，
    :param shards_list_file:
    :param output_dir_path:
    :param batch_size:
    :param num_workers:
    :return:
    """
    if output_dir_path is None:
        output_dir_path = os.path.dirname(shards_list_file)
    dataset_conf = {'batch_conf': {'batch_size': batch_size, 'batch_type': 'static', 'max_frames_in_batch': 0},
                    'fbank_conf': {'dither': 0.1, 'frame_length': 25, 'frame_shift': 10, 'num_mel_bins': 80},
                    'filter_conf': {'max_length': 1600, 'max_output_input_ratio': 0.125, 'min_length': 100,
                                    'token_max_length': 200, 'token_min_length': 1}, 'pitch_shift': False,
                    'resample_conf': {'resample_rate': 16000}, 'shuffle': True, 'shuffle_conf': {'shuffle_size': 10240},
                    'sort': True, 'sort_conf': {'sort_size': 512}, 'spec_aug': False,
                    'spec_aug_conf': {'max_f': 10, 'max_t': 50, 'num_f_mask': 2, 'num_t_mask': 2},
                    'speed_perturb': False,
                    'split_with_space': True, 'token_mask': False, 'token_mask_conf': {'p': 0.2},
                    'volume_perturb': False}

    test_dataset = Dataset("shard",
                           shards_list_file,
                           # "./tmp.list",
                           {},
                           None,
                           dataset_conf,
                           partition=False,
                           num_workers=num_workers)
    generator = torch.Generator()
    generator.manual_seed(1234)
    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=num_workers, generator=generator,prefetch_factor = 20)
    time_now = do_get_now_time()
    output_dir = output_dir_path
    os.makedirs(output_dir, exist_ok=True)
    data_list_path = os.path.join(output_dir, "data.list")
    data_list = []
    for i, batch in enumerate(test_data_loader):
        sorted_key = batch['keys']
        sorted_txt = batch['labels']
        sorted_wav = batch['wavs']
        sorted_extra_dicts = batch['extra_dicts']
        sorted_file_path = batch['tar_files']
        j = 0
        for key, txt, wav ,extra_dict,file_path in zip(sorted_key, sorted_txt, sorted_wav,sorted_extra_dicts,sorted_file_path):
            j += 1
            file_path_key = do_get_file_pure_name_from_path(file_path)
            if task_tag is not None:
                item_dict = {'tas_tag': task_tag, 'key': key, 'txt': txt, 'wav': "only-meta", 'extra': extra_dict,
                             "tar_path_key": file_path_key}
            else:
                item_dict = {'key': key, 'txt': txt, 'wav': "only-meta", 'extra': extra_dict, "tar_path_key":file_path_key}
            data_list.append(item_dict)
            # torchaudio.save(wav_path, wav, 16000)
            # with open(wav_path, "wb") as wav_file:
            #     wav_file.write(wav)
    time_elapsed = do_get_elapsed_time(time_now)
    print(f'time_elapsed: {time_elapsed}')  # 10个tar包用时13.5s
    write_dict_list_to_jsonl(data_list, data_list_path)
    if tar_dir is None:
        tar_dir = os.path.dirname(shards_list_file)
    # 分发little data.list
    do_distribute_datalist_for_conbine_type(data_list, output_dir, task_tag=task_tag, tar_dir=tar_dir)

def do_distribute_datalist_for_conbine_type(data_list_path_or_list, output_dir, task_tag=None,tar_dir = None):
    if isinstance(data_list_path_or_list, str):
        data_list = utils_file.load_dict_list_from_jsonl(data_list_path_or_list)
    elif isinstance(data_list_path_or_list, list):
        data_list = data_list_path_or_list
    else:
        raise ValueError(f'data_list_path_or_list 类型错误, 应该是str或list')
    if tar_dir is None:
        tar_dir = output_dir
    print(f'outputdir: {output_dir}')
    print(f'tar_dir: {tar_dir}')
    dict_for_distribute = {}
    res_list = []
    for item in tqdm.tqdm(data_list, total=len(data_list), desc='分发little data.list'):
        if task_tag is not None:
            item['task'] = task_tag
        if 'extra' in item:
            if 'question' in item['extra'] and 'q_txt' not in item['extra']:
                item['extra']['q_txt'] = item['extra']['question']
        tar_key = item['tar_path_key']
        if tar_key not in dict_for_distribute:
            dict_for_distribute[tar_key] = []
        dict_for_distribute[tar_key].append(item)
        if len(dict_for_distribute[tar_key]) == 1000:
            tar_file_path = os.path.join(output_dir, f'{tar_key}.list')
            res_list.append(tar_file_path)
            write_dict_list_to_jsonl(dict_for_distribute[tar_key], tar_file_path)
            dict_for_distribute.pop(tar_key)
    if len(dict_for_distribute) > 0:
        for tar_key in dict_for_distribute:
            tar_file_path = os.path.join(output_dir, f'{tar_key}.list')
            res_list.append(tar_file_path)
            write_dict_list_to_jsonl(dict_for_distribute[tar_key], tar_file_path)
    if task_tag is not None and isinstance(data_list_path_or_list, str):
        write_dict_list_to_jsonl(data_list,  data_list_path_or_list)
    list_combine_path = os.path.join(output_dir, 'combines_list.txt')
    utils_file.write_list_to_file(res_list, list_combine_path)
    tar_root_dir_info = os.path.join(output_dir, 'combines_tar_root.txt')
    all_dict_path = os.path.join(output_dir, 'data.list')
    write_dict_list_to_jsonl(data_list, all_dict_path)
    utils_file.write_list_to_file([tar_dir], tar_root_dir_info)



if __name__ == '__main__':
    shards_list_path = "common_utils/fake_data/shard/shards_list.txt"
    output_dir = "common_utils/fake_data/combine"
    do_convert_tartype2tardata_combine_type(shards_list_path, output_dir, task_tag='<TRANSCRIBE>', batch_size=1, num_workers=1)

