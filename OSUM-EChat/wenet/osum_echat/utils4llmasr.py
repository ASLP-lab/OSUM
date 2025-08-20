import random
from typing import Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from wenet.utils.common import pad_list
from gxl_ai_utils.utils import utils_file


def add_sos_eos4speech_llm(ys_pad: torch.Tensor, sos: int, eos: int,
                           ignore_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add <sos> and <eos> labels.
    为out后接一个eos. in基本保持不变

    Args:
        ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
        sos (int): index of <sos>
        eos (int): index of <eeos>
        ignore_id (int): index of padding

    Returns:
        ys_in (torch.Tensor) : (B, Lmax)
        ys_out (torch.Tensor) : (B, Lmax + 1)

    Examples:
        >>> sos_id = 10
        >>> eos_id = 11
        >>> ignore_id = -1
        >>> ys_pad
        tensor([[ 1,  2,  3,  4,  5],
                [ 4,  5,  6, -1, -1],
                [ 7,  8,  9, -1, -1]], dtype=torch.int32)
        >>> ys_in,ys_out=add_sos_eos(ys_pad, sos_id , eos_id, ignore_id)
        >>> ys_in
        tensor([[ 1,  2,  3,  4,  5],
                [ 4,  5,  6, 11, 11],
                [ 7,  8,  9, 11, 11]])
        >>> ys_out
        tensor([[ 1,  2,  3,  4,  5, 11],
                [ 4,  5,  6, 11, -1, -1],
                [ 7,  8,  9, 11, -1, -1]])
    """
    _sos = torch.tensor([sos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    _eos = torch.tensor([eos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    # ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_in = [y for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)


global_prompt_dict = None


def get_prompt_by_task(task_name):
    """
    根据task给定指定的prompt, 并实现prompt的多样随意性
    Args:
        task_name:

    Returns:

    """
    global global_prompt_dict
    if global_prompt_dict is None:
        global_prompt_dict = utils_file.load_dict_from_yaml('conf/prompt.yaml')
    random_index = random.randint(0, len(global_prompt_dict[task_name]) - 1)
    return global_prompt_dict[task_name][random_index]


import torch


def merge_labels_with_valid_adjacent(
        labels_embeds1, labels_target1, labels_mask1,
        labels_embeds2, labels_target2, labels_mask2,
        pad_value=0, ignore_id=-100
):
    """
    合并两组标签，有效特征紧邻拼接，无效特征后移
    Args:
        labels_embeds1 (Tensor): 标签1嵌入，形状 (B, L1, D)
        labels_target1 (Tensor): 标签1目标，形状 (B, L1)
        labels_mask1 (Tensor):  标签1掩码，形状 (B, L1)
        labels_embeds2 (Tensor): 标签2嵌入，形状 (B, L2, D)
        labels_target2 (Tensor): 标签2目标，形状 (B, L2)
        labels_mask2 (Tensor):  标签2掩码，形状 (B, L2)
        pad_value (int): 嵌入填充值
        ignore_id (int): 目标填充值（如IGNORE_ID）
    Returns:
        merged_embeds (Tensor): 合并嵌入，形状 (B, L1+L2, D)
        merged_target (Tensor): 合并目标，形状 (B, L1+L2)
        merged_mask (Tensor):  合并掩码，形状 (B, L1+L2)
    """
    batch_size = labels_embeds1.size(0)
    max_len = labels_embeds1.size(1) + labels_embeds2.size(1)

    merged_embeds = []
    merged_target = []
    merged_mask = []

    for i in range(batch_size):
        # 提取有效特征索引
        valid_indices1 = torch.where(labels_mask1[i])[0]
        valid_indices2 = torch.where(labels_mask2[i])[0]

        # 合并有效特征段
        valid_embeds = torch.cat([
            labels_embeds1[i, valid_indices1],
            labels_embeds2[i, valid_indices2]
        ], dim=0)

        valid_target = torch.cat([
            labels_target1[i, valid_indices1],
            labels_target2[i, valid_indices2]
        ], dim=0)

        valid_mask = torch.cat([
            labels_mask1[i, valid_indices1],
            labels_mask2[i, valid_indices2]
        ], dim=0)

        # 填充无效部分
        pad_length = max_len - len(valid_embeds)
        padded_embeds = torch.cat([
            valid_embeds,
            torch.full((pad_length, labels_embeds1.size(2)), pad_value, device=labels_embeds1.device)
        ], dim=0)

        padded_target = torch.cat([
            valid_target,
            torch.full((pad_length,), ignore_id, device=labels_target1.device)
        ], dim=0)

        padded_mask = torch.cat([
            valid_mask,
            torch.zeros(pad_length, dtype=torch.bool, device=labels_mask1.device)
        ], dim=0)

        merged_embeds.append(padded_embeds)
        merged_target.append(padded_target)
        merged_mask.append(padded_mask)

    # 堆叠批次结果
    merged_embeds = torch.stack(merged_embeds, dim=0).to(labels_embeds1.device)
    merged_target = torch.stack(merged_target, dim=0).to(labels_target1.device)
    merged_mask = torch.stack(merged_mask, dim=0).to(labels_mask1.device)

    return merged_embeds, merged_target, merged_mask


def make_streaming_mode_from_s2s_old(text_tokens_padded, text_tokens_lens, speech_tokens_padded, speech_tokens_lens, ):
    """

    Args:
        text_tokens_padded: (B, Lmax)
        text_tokens_lens: (B,)
        speech_tokens_padded: (B, Lmax2)
        speech_tokens_lens: (B,)

    Returns:
        streaming_mode_tokens_padded: (B, Lmax+Lmax2+1)
        streaming_mode_tokens_lens: (B,)

    首先assert每个单元的文字有效token的数量的3倍是少于该单元的speech token的数量。
    然后做如下排列：对于batch内的每个item, 先排6个文字有效token,然后再排18个speech 有效token,然后再排6个文字token,然后排18个speech token,以此类推，直到有效文本token用尽。
    """
    text_tokens_padded = text_tokens_padded.to(torch.int64)
    speech_tokens_padded = speech_tokens_padded.to(torch.int64)
    batch_size = text_tokens_padded.size(0)
    device = text_tokens_padded.device

    # 验证文字token数量不超过语音token的1/3
    for i in range(batch_size):
        assert text_tokens_lens[i] * 3 <= speech_tokens_lens[i], \
            f"Batch {i}: Text tokens * 3 should be less than speech tokens"

    streaming_mode_tokens_list = []
    streaming_mode_lens = []

    for i in range(batch_size):
        text_tokens = text_tokens_padded[i, :text_tokens_lens[i]]
        speech_tokens = speech_tokens_padded[i, :speech_tokens_lens[i]].to(torch.int64)

        streaming_tokens = []
        text_idx = 0
        speech_idx = 0

        while text_idx < text_tokens_lens[i]:
            # 处理文本token（6个一组），防止越界
            chunk_size = min(6, text_tokens_lens[i] - text_idx)
            streaming_tokens.extend(text_tokens[text_idx:text_idx + chunk_size].tolist())
            text_idx += chunk_size

            # 如果文本token不足6个，添加999标记
            if chunk_size < 6:
                streaming_tokens.append(999)

            # 处理语音token（18个一组），防止越界
            speech_chunk = min(18, speech_tokens_lens[i] - speech_idx)
            streaming_tokens.extend(speech_tokens[speech_idx:speech_idx + speech_chunk].tolist())
            speech_idx += speech_chunk

        # 如果文本token正好用完，添加999标记
        if text_idx == text_tokens_lens[i] and text_tokens_lens[i] % 6 == 0:
            streaming_tokens.append(999)

        # 添加剩余的语音token
        streaming_tokens.extend(speech_tokens[speech_idx:].tolist())

        # 转换为BFLOAT16张量
        streaming_mode_tokens_list.append(torch.tensor(streaming_tokens, dtype=torch.int64, device=device))
        streaming_mode_lens.append(len(streaming_tokens))

    streaming_mode_tokens_padded = pad_sequence(streaming_mode_tokens_list, batch_first=True, padding_value=0).to(
        device)
    streaming_mode_tokens_lens = torch.tensor(streaming_mode_lens, device=device)
    return streaming_mode_tokens_padded, streaming_mode_tokens_lens

def make_streaming_mode_from_s2s(text_tokens_padded, text_tokens_lens, speech_tokens_padded, speech_tokens_lens, ):
    """

    Args:
        text_tokens_padded: (B, Lmax)
        text_tokens_lens: (B,)
        speech_tokens_padded: (B, Lmax2)
        speech_tokens_lens: (B,)

    Returns:
        streaming_mode_tokens_padded: (B, Lmax+Lmax2+1)
        streaming_mode_tokens_lens: (B,)

    首先assert每个单元的文字有效token的数量的3倍是少于该单元的speech token的数量。
    然后做如下排列：对于batch内的每个item, 先排6个文字有效token,然后再排18个speech 有效token,然后再排6个文字token,然后排18个speech token,以此类推，直到有效文本token用尽。
    <think_end> : [13708,   766,   835,    29]
    """
    text_tokens_padded = text_tokens_padded.to(torch.int64)
    speech_tokens_padded = speech_tokens_padded.to(torch.int64)
    batch_size = text_tokens_padded.size(0)
    device = text_tokens_padded.device

    # 验证文字token数量不超过语音token的1/3
    for i in range(batch_size):
        assert text_tokens_lens[i] * 3 <= speech_tokens_lens[i], \
            f"Batch {i}: Text tokens * 3 should be less than speech tokens"

    streaming_mode_tokens_list = []
    streaming_mode_lens = []

    for i in range(batch_size):
        text_tokens = text_tokens_padded[i, :text_tokens_lens[i]]
        speech_tokens = speech_tokens_padded[i, :speech_tokens_lens[i]].to(torch.int64)

        streaming_tokens = []
        text_idx = 0
        speech_idx = 0

        while text_idx < text_tokens_lens[i]: # 这里的指针指的是左指针，肯定不能等于 len(text_tokens)
            # 处理文本token（6个一组），防止越界
            chunk_size = min(6, text_tokens_lens[i] - text_idx)
            streaming_tokens.extend(text_tokens[text_idx:text_idx + chunk_size].tolist())
            text_idx += chunk_size

            # 处理语音token（18个一组），防止越界
            speech_chunk = min(18, speech_tokens_lens[i] - speech_idx)
            streaming_tokens.extend(speech_tokens[speech_idx:speech_idx + speech_chunk].tolist())
            speech_idx += speech_chunk

        # 添加剩余的语音token
        streaming_tokens.extend(speech_tokens[speech_idx:].tolist())

        streaming_mode_tokens_list.append(torch.tensor(streaming_tokens, dtype=torch.int64, device=device))
        streaming_mode_lens.append(len(streaming_tokens))

    streaming_mode_tokens_padded = pad_sequence(streaming_mode_tokens_list, batch_first=True, padding_value=0).to(
        device)
    streaming_mode_tokens_lens = torch.tensor(streaming_mode_lens, device=device)
    return streaming_mode_tokens_padded, streaming_mode_tokens_lens


def make_streaming_mode_from_s2s4think(
    text_tokens_padded, text_tokens_lens,
    speech_tokens_padded, speech_tokens_lens,
):
    """
    Args:
        text_tokens_padded: (B, Lmax)
        text_tokens_lens:   (B,)
        speech_tokens_padded: (B, Lmax2)
        speech_tokens_lens:   (B,)

    Returns:
        streaming_mode_tokens_padded: (B, Lmax+Lmax2+1)
        streaming_mode_tokens_lens:   (B,)
    """
    text_tokens_padded   = text_tokens_padded.to(torch.int64)
    speech_tokens_padded = speech_tokens_padded.to(torch.int64)
    batch_size = text_tokens_padded.size(0)
    device     = text_tokens_padded.device

    # 验证文字 token 数量不超过语音 token 的 1/3
    for i in range(batch_size):
        assert text_tokens_lens[i] * 3 <= speech_tokens_lens[i], \
            f"Batch {i}: Text tokens * 3 should be <= speech tokens"

    streaming_mode_tokens_list = []
    streaming_mode_lens        = []

    # 要检测的子序列
    target_seq = [13708, 766, 835, 29]
    seq_len = len(target_seq)

    for i in range(batch_size):
        # 取出本样本的有效文本和语音序列
        text_tokens   = text_tokens_padded[i, :text_tokens_lens[i]]
        speech_tokens = speech_tokens_padded[i, :speech_tokens_lens[i]]
        streaming_tokens = []

        # —— 新增逻辑：先在 text_tokens 中寻找整个子序列 target_seq ——
        text_list = text_tokens.tolist()
        prefix_end_idx = 0
        # 滑窗匹配
        for j in range(text_tokens_lens[i] - seq_len + 1):
            if text_list[j:j + seq_len] == target_seq:
                prefix_end_idx = j + seq_len
                break
        # 如果找到了，就先把前缀一次性输出
        if prefix_end_idx > 0:
            streaming_tokens.extend(text_list[:prefix_end_idx])
            text_idx = prefix_end_idx
        else:
            text_idx = 0
        # —— 新增逻辑结束 ——

        speech_idx = 0

        # 之后再从 text_idx 开始做常规的“6 文本 + 18 语音”交错
        while text_idx < text_tokens_lens[i]:
            # 文本块（最多 6）
            chunk_size = min(6, text_tokens_lens[i] - text_idx)
            streaming_tokens.extend(text_list[text_idx:text_idx + chunk_size])
            text_idx += chunk_size

            # 语音块（最多 18）
            speech_chunk = min(18, speech_tokens_lens[i] - speech_idx)
            streaming_tokens.extend(speech_tokens[speech_idx:speech_idx + speech_chunk].tolist())
            speech_idx += speech_chunk

        # 最后再把剩余的所有语音 token 全部补上
        streaming_tokens.extend(speech_tokens[speech_idx:].tolist())

        # 收集本样本结果
        streaming_mode_tokens_list.append(
            torch.tensor(streaming_tokens, dtype=torch.int64, device=device)
        )
        streaming_mode_lens.append(len(streaming_tokens))

    # padding 到同样长度
    streaming_mode_tokens_padded = pad_sequence(
        streaming_mode_tokens_list,
        batch_first=True,
        padding_value=0
    ).to(device)
    streaming_mode_tokens_lens = torch.tensor(streaming_mode_lens, device=device)

    return streaming_mode_tokens_padded, streaming_mode_tokens_lens


def do_embedding_for_two_embeds(input_token_ids, dividing_id, embedding1, embedding2):
    """

    Args:
        input_token_ids: (B, Lmax) ,其词表范围是[0, vocab_size1+vocab_size2)
        dividing_id: int, 第一个词表的个数
        embedding1: nn.Embedding(vocab_size1, embedding_dim)
        embedding2: nn.Embedding(vocab_size2, embedding_dim)

    Returns:
        embedding1_output: (B, Lmax, D)

    把两个embeddings 虚拟成一个大的词向量
    """
    input_token_ids = input_token_ids.to(torch.int64)
    mask4embedding1 = input_token_ids < dividing_id
    mask4embedding2 = input_token_ids >= dividing_id
    embedding1_output = embedding1(input_token_ids[mask4embedding1]).to(embedding1.weight.dtype)
    embedding2_output = embedding2(input_token_ids[mask4embedding2] - dividing_id).to(embedding1.weight.dtype)
    res_output = torch.zeros(input_token_ids.size(0), input_token_ids.size(1), embedding1.embedding_dim,dtype=embedding1.weight.dtype,
                             device=embedding1.weight.device)
    res_output[mask4embedding1] = embedding1_output
    res_output[mask4embedding2] = embedding2_output
    return res_output

def do_convert_num2text(num_str: str):
    """
    将数字字符串转换为中文数字
    Args:
        num_str: 数字字符串
    Returns:
        转换后的中文数字字符串
    """
    import cn2an
    num_str = num_str.strip()
    output = cn2an.transform(num_str, "an2cn")
    return output


def _do_test_for_streaming_chat():
    # test make_streaming_mode_from_s2s
    text_tokens_padded = torch.randint(0, 100, (3, 10)).to(torch.device('npu:0'))
    text_tokens_lens = torch.tensor([5, 7, 3]).to(torch.device('npu:0'))
    speech_tokens_padded = torch.randint(100, 200, (3, 150)).to(torch.device('npu:0'))
    speech_tokens_lens = torch.tensor([100, 120, 80]).to(torch.device('npu:0'))
    streaming_mode_tokens_padded, streaming_mode_tokens_lens = make_streaming_mode_from_s2s(text_tokens_padded,
                                                                                            text_tokens_lens,
                                                                                            speech_tokens_padded,
                                                                                            speech_tokens_lens)
    print(streaming_mode_tokens_padded.shape)
    print(streaming_mode_tokens_padded.device)
    print(streaming_mode_tokens_lens)
    print(streaming_mode_tokens_lens.device)

    # test do_embedding_for_two_embeds
    input_token_ids = torch.randint(0, 100, (3, 10)).to(torch.device('npu:0'))
    dividing_id = 50
    embedding1 = torch.nn.Embedding(50, 10).to(torch.device('npu:0'))
    embedding2 = torch.nn.Embedding(50, 10).to(torch.device('npu:0'))
    res_output = do_embedding_for_two_embeds(input_token_ids, dividing_id, embedding1, embedding2)
    print(res_output.shape)
    print(res_output.device)
    a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, ]).to(torch.device('npu:0'))
    print(a[3:1000])
if __name__ == '__main__':
    """"""
