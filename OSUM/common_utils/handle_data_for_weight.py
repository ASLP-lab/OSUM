import random

from gxl_ai_utils.utils import utils_file

data_config_path, tmp_file_path = utils_file.do_get_commandline_param(2)
random.seed(7890)
data_info_dict = utils_file.load_dict_from_yaml(data_config_path)
total_list = []
for data_info in data_info_dict.values():
    data_path_i = data_info['path']
    utils_file.logging_info(f'path:{data_path_i} ')
    data_weight = int(float(data_info['weight']))
    if data_weight == 0:
        data_weight = float(data_info['weight'])
        if data_weight >= 0:
            utils_file.logging_info(f'data {data_path_i} weight is {data_weight}, will be used as a list')
        final_data_list_i_tmp = utils_file.load_list_file_clean(data_path_i)
        true_num = int(len(final_data_list_i_tmp)*data_weight)
        final_data_list_i = utils_file.do_get_random_sublist(final_data_list_i_tmp, true_num)
    else:
        final_data_list_i = utils_file.load_list_file_clean(data_path_i) * data_weight
    utils_file.logging_info(f'true load num is : {len(final_data_list_i)}')
    total_list.extend(final_data_list_i)
random.shuffle(total_list)
total_list = total_list[::-1]
utils_file.write_list_to_file(total_list, tmp_file_path)
