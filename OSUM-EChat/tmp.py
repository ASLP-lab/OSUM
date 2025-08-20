import codecs

from gxl_ai_utils.utils import utils_file
input_file = "./infer_res.scp"
dict_info = utils_file.load_dict_from_scp(input_file)
new_dict = {}
def write_dict_to_scp(dic: dict, scp_file_path: str):
    # logging_print("开始write_dict_to_scp()，数据总条数为:", len(dic))
    # os.makedirs(os.path.dirname(scp_file_path), exist_ok=True)
    with codecs.open(scp_file_path, 'w', encoding='utf-8') as f:
        for k, v in dic.items():
            f.write(f"{k} {v}\n")
for key, value in dict_info.items():
    new_value = value.split("<think end>")[-1]
    new_dict[key] = new_value
write_dict_to_scp(new_dict, "infer_res_new.scp")