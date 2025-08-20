# 负责wav & text instruct 的处理
import codecs
import json
import random

from gxl_ai_utils.utils import utils_file
import torch
import os

# asr+X
task_names = [
    "age",
    "gender",
    "style",
    "emotion",
    "caption"
]
map_dict = {
    "<TRANSCRIBE> <STYLE>": "style",
    "<TRANSCRIBE> <AGE>": "age",
    "<TRANSCRIBE> <GENDER>": "gender",
    "<TRANSCRIBE> <EMOTION>": "emotion",
    "<TRANSCRIBE> <CAPTION>": "caption",
    "<STYLE>": "style_only_X",
    "<AGE>": "age_only_X",
    "<GENDER>": "gender_only_X",
    "<EMOTION>": "emotion_only_X",
    "<CAPTION>": "caption_only_X",
}
question_info_dir = "/mnt/sfs/asr/code/osum_osum_echat/examples/wenetspeech/whisper/conf/language_follow_prompt/Q_with_asr"
# question_info_dir = "D:\osum_echat_workspace_2\osum_using_ofical\osum_osum_echat\examples\wenetspeech\whisper\conf\language_follow_prompt\Q_with_asr"
question_big_dict = {}
question_wav_big_dict = {}

# for task_name in task_names:
#     task_json_path = os.path.join(question_info_dir, f"{task_name}.json")
#     dict_i = utils_file.load_dict_from_json(task_json_path)
#     question_strs = list(dict_i.values())
#     question_big_dict[task_name] = question_strs
#
#
# for task_name in task_names:
#     task_json_path = os.path.join(question_info_dir, f"{task_name}.json")
#     dict_i = utils_file.load_dict_from_json(task_json_path)
#     question_strs = list(dict_i.keys())  # 获取字典的键作为问题字符串列表
#     question_wav_big_dict[task_name] =  []
#     for i in range(1, 21):
#         new_question_strs = [f"/mnt/sfs/asr/update_data/Q_with_asr_added_by_20250406/{task_name}_{x}_speaker_{i}.wav" for x in question_strs]
#         question_wav_big_dict[task_name].extend(new_question_strs)

question_info_dir = "/mnt/sfs/asr/code/osum_osum_echat/examples/wenetspeech/whisper/conf/language_follow_prompt/Q"
# question_info_dir = "D:\osum_echat_workspace_2\osum_using_ofical\osum_osum_echat\examples\wenetspeech\whisper\conf\language_follow_prompt\Q"
# for task_name in task_names:
#     task_json_path = os.path.join(question_info_dir, f"{task_name}.json")
#     dict_i = utils_file.load_dict_from_json(task_json_path)
#     question_strs = list(dict_i.values())
#     question_big_dict[task_name+"_only_X"] = question_strs
# for task_name in task_names:
#     task_json_path = os.path.join(question_info_dir, f"{task_name}.json")
#     dict_i = utils_file.load_dict_from_json(task_json_path)
#     question_strs = list(dict_i.keys())  # 获取字典的键作为问题字符串列表
#     question_wav_big_dict[task_name+"_only_X"] =  []
#     for i in range(1,21):
#         new_question_strs = [f"/mnt/sfs/asr/update_data/Q_only_X_added_by_20250406/{task_name}_{x}_speaker_{i}.wav" for x in question_strs]
#         question_wav_big_dict[task_name+"_only_X"].extend(new_question_strs)


# for answer
answer_info_dir = "/mnt/sfs/asr/code/osum_osum_echat/examples/wenetspeech/whisper/conf/language_follow_prompt/Q_with_asr/answer"
# answer_info_dir = "D:\osum_echat_workspace_2\osum_using_ofical\osum_osum_echat\examples\wenetspeech\whisper\conf\language_follow_prompt\Q_with_asr\\answer"
answer_big_dict = {}
# for task_name in task_names:
#     task_json_path = os.path.join(answer_info_dir, f"{task_name}.json")
#     dict_i = utils_file.load_dict_from_json(task_json_path)
#     answer_big_dict[task_name] = dict_i

answer_info_dir = "/mnt/sfs/asr/code/osum_osum_echat/examples/wenetspeech/whisper/conf/language_follow_prompt/Q/answer"
# answer_info_dir = "D:\osum_echat_workspace_2\osum_using_ofical\osum_osum_echat\examples\wenetspeech\whisper\conf\language_follow_prompt\Q\\answer"
# for task_name in task_names:
#     task_json_path = os.path.join(answer_info_dir, f"{task_name}.json")
#     dict_i = utils_file.load_dict_from_json(task_json_path)
#     answer_big_dict[task_name+"_only_X"] = dict_i



def get_question_prompt_by_task(task_tag):
    key_str = map_dict.get(task_tag, None)
    if key_str is None:
        return None
    question_strs = question_big_dict.get(key_str, [])
    if len(question_strs) == 0:
        return None
    # 随机算一个句子
    question_str = random.choice(question_strs)
    return question_str

def get_question_wav_path_by_task(task_tag):
    key_str = map_dict.get(task_tag, None)
    if key_str is None:
        return None
    question_strs = question_wav_big_dict.get(key_str, [])
    if len(question_strs) == 0:
        return None
    # 随机算一个句子
    question_str = random.choice(question_strs)
    return question_str

def get_answer_prompt_by_task(task_tag, answer_tag, asr_txt=None):
    key_str = map_dict.get(task_tag, None)
    if key_str is None:
        utils_file.logging_error(f"task_tag {task_tag} not in map_dict")
        return None
    answer_dict = answer_big_dict.get(key_str, {})
    answer_strs = answer_dict.get(answer_tag, [])
    if len(answer_strs) == 0:
        utils_file.logging_error(f"answer_tag {answer_tag} not in answer_dict of {task_tag}")
        return None
    # 随机算一个句子
    answer_str = random.choice(answer_strs)
    if asr_txt is not None:
        if "{}" not in answer_str:
            utils_file.logging_error(f"answer_str {answer_str} not contain","{}", 'but asr_txt is not None')
            return None
        answer_str = answer_str.format(asr_txt)
    else:
        if "{}" in answer_str:
            utils_file.logging_error(f"answer_str {answer_str} contain","{}", 'but asr_txt is None')
            return None
    return answer_str

if __name__ == '__main__':
    task_tag = "<CAPTION>"
    type_tag = "church_bells"
    question_str = get_question_wav_path_by_task(task_tag)
    print(question_str)
    question_str = get_question_wav_path_by_task(task_tag)
    print(question_str)
    question_str = get_question_wav_path_by_task(task_tag)
    print(question_str)
    question_str = get_question_wav_path_by_task(task_tag)
    print(question_str)

    question_str = get_question_prompt_by_task(task_tag)
    print(question_str)
    question_str = get_question_prompt_by_task(task_tag)
    print(question_str)
    question_str = get_question_prompt_by_task(task_tag)
    print(question_str)
    question_str = get_question_prompt_by_task(task_tag)
    print(question_str)

    question_str = get_answer_prompt_by_task(task_tag, type_tag)
    print(question_str)
    question_str = get_answer_prompt_by_task(task_tag, type_tag)
    print(question_str)
    question_str = get_answer_prompt_by_task(task_tag, type_tag)
    print(question_str)
    question_str = get_answer_prompt_by_task(task_tag, type_tag)
    print(question_str)

    question_str = get_answer_prompt_by_task(task_tag, type_tag, "I am a child")
    print(question_str)
    question_str = get_answer_prompt_by_task(task_tag, type_tag, "I am a child")
    print(question_str)
    question_str = get_answer_prompt_by_task(task_tag, type_tag, "I am a child")
    print(question_str)
    question_str = get_answer_prompt_by_task(task_tag, type_tag, "I am a child")
    print(question_str)

    # caption_json_path = "D:\osum_echat_workspace_2\osum_using_ofical\osum_osum_echat\examples\wenetspeech\whisper\conf\language_follow_prompt\Q_with_asr\\answer\caption.json"
    # caption_dict = utils_file.load_dict_from_json(caption_json_path)
    # new_caption_dict = {}
    # utils_file.print_list(list(caption_dict.keys()))
    # for k, v in caption_dict.items():
    #     new_k = k.lower()
    #     new_caption_dict[new_k] = v
    # with codecs.open(caption_json_path, 'w', encoding='utf-8') as f:
    #     json.dump(new_caption_dict, f, ensure_ascii=False, indent=4)




