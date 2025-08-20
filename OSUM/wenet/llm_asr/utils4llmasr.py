import random
from typing import Tuple

import torch

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
    random_index = random.randint(0, len(global_prompt_dict[task_name])-1)
    return global_prompt_dict[task_name][random_index]
