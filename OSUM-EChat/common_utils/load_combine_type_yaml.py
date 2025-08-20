import os
import random
import time

from gxl_ai_utils.utils import utils_file

data_config_path, tmp_file_path = utils_file.do_get_commandline_param(2)
# random.seed(10086)# 老的
# 把当前时间戳作为随机种子
random.seed(int(time.time()))
# random.seed(7891)# 尝试一下新的顺序  #7890
data_info_dict = utils_file.load_dict_from_yaml(data_config_path)
if data_info_dict is None:
    data_info_dict = {}
total_list = []
for data_info in data_info_dict.values():
    if "path" not in data_info:
        print(f"path or weight not in data_info:{data_info}")
        continue
    if "weight" not in data_info:
        data_weight = 1
    else:
        data_weight = int(float(data_info['weight']))
    data_path_i = data_info['path']
    utils_file.logging_info(f'path:{data_path_i} ')

    if data_weight == 0:
        data_weight = float(data_info['weight'])
        if data_weight >= 0:
            utils_file.logging_info(f'data {data_path_i} weight is {data_weight}, will be used as a list')
        final_data_list_i_tmp = utils_file.load_list_file_clean(data_path_i)
        true_num = int(len(final_data_list_i_tmp)*data_weight)
        final_data_list_i = utils_file.do_get_random_sublist(final_data_list_i_tmp, true_num)
    else:
        final_data_list_i = utils_file.load_list_file_clean(data_path_i) * data_weight
    # 判断数据类型
    if "combines_list.txt" in data_path_i:
        print(f'是 combine类型的数据')
        tar_root_path = data_path_i.replace('combines_list.txt', 'combines_tar_root.txt')
        if not os.path.exists(tar_root_path):
            utils_file.logging_info(f'combine_list.txt:{data_path_i} 对应的 combines_tar_root.txt:{tar_root_path} 不存在')
            continue
        tar_root = utils_file.load_first_row_clean(tar_root_path)
        if tar_root.endswith('/'):
            tar_root = tar_root[:-1]
        utils_file.logging_info(f' tar_root:{tar_root}')
        new_final_data_list_i = []
        for data_path_j in final_data_list_i:
            # "combine_path|shard_path"
            tmp_lines = f'{data_path_j}|{tar_root}/{utils_file.do_get_file_pure_name_from_path(data_path_j)}.tar'
            new_final_data_list_i.append(tmp_lines)
    else:
        print(f'不是 combine类型的数据,是传统shard类型的数据')
        new_final_data_list_i = [f'-|{data_path_j}' for data_path_j in final_data_list_i]

    utils_file.logging_info(f'true load num is : {len(new_final_data_list_i)}')
    total_list.extend(new_final_data_list_i)
random.shuffle(total_list)
utils_file.write_list_to_file(total_list, tmp_file_path)
