import re

from gxl_ai_utils.utils import utils_file

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Compute accuracy')
    parser.add_argument('--ref_path', type=str, default='config.yaml', help='config file path')
    parser.add_argument('--hyp_path', type=str, default='model.pth', help='model file path')
    parser.add_argument('--output_path', type=str, default='data', help='data file path')

    args = parser.parse_args()
    return args

def do_get_first_tag_from_str(input_str, upper=True):
    """
    得到第一个<>标签的内容，并将其转为大写
    >>> input : "<tag1>content1<tag2>content2<tag3>content3"
    >>> output: "<TAG1>"
    :param upper: if false, return the lower case of the tag
    :param input_str:
    :return:
    """
    # 使用正则表达式提取所有 <> 中的内容
    matches = re.findall(r'<.*?>', input_str)
    # 输出结果
    if len(matches)==0:
        return '<--no_tag-->'
    if upper:
        return matches[0].upper()
    else:
        return matches[0].lower()

def do_showing_confusion_matrix(labels, matrix,title='', output_fig_path: str=None):
    """
    可视化混淆矩阵的函数
    :param labels: 标签序列，类型为列表等可迭代对象
    :param matrix: 代表混淆矩阵的二维列表，元素为整数，形状应为(len(labels), len(labels))
    """
    import matplotlib.pyplot as plt
    import numpy as np
    # 设置中文字体为黑体，以支持中文显示（确保系统中已安装黑体字体）
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 解决负号显示问题（有时候会出现负号显示异常，这一步是为了保证显示正常）
    plt.rcParams['axes.unicode_minus'] = False
    num_classes = len(labels)
    fig, ax = plt.subplots()
    # 使用imshow来绘制热力图展示混淆矩阵
    im = ax.imshow(np.array(matrix), cmap=plt.cm.Blues)

    # 设置x轴和y轴的刻度以及对应的标签
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # 旋转x轴刻度标签，让其更美观显示
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 添加每个方格中的数值标签
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax.text(j, i, matrix[i][j],
                           ha="center", va="center", color="black")

    ax.set_title(f"{title} Confusion Matrix")
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    fig.tight_layout()
    if output_fig_path is not None:
        plt.savefig(output_fig_path)
    else:
        plt.show()


def do_compute_acc(ref_file, hyp_file, output_path):
    """
    计算正确率和错误率，并列出混淆矩阵
    :return:
    """
    output_dir = os.path.dirname(output_path)
    utils_file.makedir_sil(output_dir)
    utils_file.logging_info('开始计算正确率，acc')
    acc_path = output_path
    hyp_dict = utils_file.load_dict_from_scp(hyp_file)
    ref_dict = utils_file.load_dict_from_scp(ref_file)
    tag_hyp_dict = {}
    for key, value in hyp_dict.items():
        tag_hyp_dict[key] = do_get_first_tag_from_str(value)
    tag_ref_dict = {}
    for key, value in ref_dict.items():
        tag_ref_dict[key] = do_get_first_tag_from_str(value)
    output_acc_f = open(acc_path, 'w', encoding='utf-8')
    same_num = 0
    all_tags = set()
    # utils_file.print_dict(utils_file.do_get_random_subdict(tag_ref_dict, 10))
    for tag in tag_ref_dict.values():
        all_tags.add(tag)
    labels = sorted(list(all_tags))
    utils_file.logging_info(f'标签种类为：{labels}')
    num_classes = len(labels)
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for key, hyp_tags in tag_hyp_dict.items():
        if key not in tag_ref_dict:
            continue
        ref_tags = tag_ref_dict[key]
        # 判断标签是否相同
        if ref_tags == hyp_tags:
            if_same = True
            same_num += 1
            index = labels.index(ref_tags)
            matrix[index][index] += 1
        else:
            if_same = False
            # 标签不同的情况，找到真实标签和预测标签对应的索引，在相应的位置加1
            ref_index = labels.index(ref_tags)
            if hyp_tags in labels:
                hyp_index = labels.index(hyp_tags)
                matrix[ref_index][hyp_index] += 1
        # 向文件写入数据
        output_acc_f.write(f"key: {key}\n")
        output_acc_f.write(f"ref_tag: {ref_tags}\n")
        output_acc_f.write(f"hyp_tag: {hyp_tags}\n")
        output_acc_f.write(f"if_same: {if_same}\n")
        output_acc_f.write("\n")  # 添加空行分隔不同的条目
    acc_num = same_num / len(tag_hyp_dict)
    output_acc_f.write(f'正确率为：{acc_num}')
    output_acc_f.flush()
    output_acc_f.close()
    figure_path = os.path.join(output_dir, 'confusion_matrix.png')
    do_showing_confusion_matrix(labels, matrix, output_fig_path=figure_path)
    return {
        'acc': acc_num,
    }

if __name__ == '__main__':
    args = parse_args()
    ref_file = args.ref_path
    hyp_file = args.hyp_path
    output_path = args.output_path
    do_compute_acc(ref_file, hyp_file, output_path)
